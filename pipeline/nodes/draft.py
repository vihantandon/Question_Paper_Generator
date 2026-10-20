"""
Step 3 -- draft (the language model writes the question).

    retrieve gave:  grounding text (relevant_chunks, role "main" | "blend")
                    and real past exam questions (style_examples)
    draft makes:    ONE question, as checked JSON

The sources are numbered [S1] [S2] ... in the prompt, and the model must say
which ones it used. parse_draft() then checks the reply before it is accepted:

  * the question and an answer outline are there;
  * "marks" is exactly target_marks, and the parts (a), (b) ... add up to it;
  * it cites at least one source about the main topic -- and, when there is
    a prerequisite to blend in, one about that too, and says where it is used.

A reply that fails a check goes back to the model with the reason (that is
done by pipeline/llm.py), so a bad reply never reaches the state.

Not checked here: whether every fact really is in the sources, and whether
the question has one clear reading. That is the next step, quality_check.
When it rejects a draft it sets revision_feedback, and this node then writes
a new draft that fixes the problem (the feedback goes into the prompt).

A topic with no grounding text is NOT sent to the model: the draft would be
made up. draft stays None and draft_error says why.

Reads:  current_topic, blend_topic, topic_cos, target_marks, target_difficulty,
        prompt_instruction, relevant_chunks, style_examples, grounding_warning,
        revision_feedback (+ the previous draft, when there is feedback)
Writes: draft, draft_error, draft_attempts
"""
import re

from pipeline.llm import LLMError, LLMFatalError

MAX_EXAMPLE_CHARS = 1200      # a past question with a long program is cut here in the prompt
PART_LABELS = "abcdefghij"

SYSTEM_PROMPT = """\
You set exam questions for a B.Tech computer science course at an Indian university.

Rules:
1. Use ONLY facts from the numbered SOURCES. Every definition, rule, number, piece of code \
or behaviour the question or its answer depends on must be stated or directly shown in a \
source you cite. Do not add anything the sources do not support.
2. The question must have one correct answer and only one reasonable reading.
3. The PAST EXAM QUESTIONS show this university's style, wording and depth. Copy the style; \
never copy their content, data, code or wording.
4. Marks set the depth: a 2-mark question needs a few lines; a 6- or 10-mark question needs \
working, a trace, code or a derivation.
5. If the question has parts (a), (b), ..., the parts' marks must add up to the total.
6. Do not number the question ("Q1.") and do not write marks, CO or Bloom labels inside it; \
they are added when the paper is put together.
7. Reply with ONE JSON object and nothing else."""


def number_sources(chunks):
    """{"S1": chunk, "S2": chunk, ...} in the order retrieve gave them.
    The same state always gives the same numbering, so a later step can
    look a label up again."""
    return {f"S{i}": c for i, c in enumerate(chunks, start=1)}


def describe_source(chunk, topic, blend):
    """The header line of one source in the prompt."""
    if chunk.get("role") == "blend" and blend:
        about = f'prerequisite "{blend["name"]}"'
    else:
        about = f'main topic "{topic["name"]}"'
    kind = chunk.get("type") or "text"
    section = chunk.get("section")
    where = f', section "{section}"' if section and kind == "book" else ""
    return f"{about} -- {kind}{where}"


def shorten(text, limit=MAX_EXAMPLE_CHARS):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


# "**Q3.**", "Q.3", "Q1)" at the start of a past question
QUESTION_NUMBER_RE = re.compile(r"^\W*Q(?:uestion)?\s*\.?\s*\d+\s*[.):\-]?\**\s*", re.IGNORECASE)
# a line of only asterisks: what is left where the splitter took out "[CO2, 5 Marks]"
EMPTY_BOLD_RE = re.compile(r"^[ \t]*\*{2,}[ \t]*\n?", re.MULTILINE)


def clean_example(text):
    """A past question as the model should see it: without its number (the
    model copied "Q3." into its own question) and without the "****" lines
    the PYQ splitter leaves behind; long ones are cut."""
    text = QUESTION_NUMBER_RE.sub("", (text or "").strip(), count=1)
    return shorten(EMPTY_BOLD_RE.sub("", text))


def reply_format(marks, blend):
    blend_line = (f'  "blend_used": "one sentence: where a correct answer needs {blend["name"]}"'
                  if blend else '  "blend_used": ""')
    return "\n".join([
        "Reply with ONLY this JSON object:",
        "{",
        '  "question": "the question as printed: the whole question, or the common part '
        'before its sub-parts (include any code or data it needs)",',
        '  "parts": [{"label": "a", "text": "a sub-question", "marks": 2}],',
        f'  "marks": {marks},',
        '  "answer_outline": ["each step, fact or result a full-marks answer must contain"],',
        '  "sources_used": ["S1", "S2"],',
        blend_line,
        "}",
        f'"parts" is [] when the question has no sub-parts; with parts, their marks add up to '
        f'{marks}. "marks" must be exactly {marks}. "sources_used" lists every source the '
        f'question and its answer rely on.',
    ])


def build_messages(state):
    """The two messages sent to the model: the fixed rules (system) and this
    question's task, sources and examples (user)."""
    topic, blend = state["current_topic"], state.get("blend_topic")
    marks = state["target_marks"]
    lines = ["TASK", state["prompt_instruction"],
             f"Total marks: {marks}.  Difficulty: {state.get('target_difficulty', 'not given')}."]

    cos = state.get("topic_cos") or []
    if cos:
        lines += ["", "Course outcome(s) the question should test:"]
        lines += [f"- {c.get('short_id') or c.get('co_id')}: {c.get('description', '')}" for c in cos]

    if blend:
        lines += ["", f'PREREQUISITE TO BLEND IN: "{blend["name"]}" (semester {blend["semester"]}). '
                      f'A correct answer must genuinely need it. Use a source about it, and say '
                      f'where it is needed in "blend_used".']

    lines += ["", "SOURCES -- the only facts you may use"]
    for label, chunk in number_sources(state.get("relevant_chunks") or []).items():
        lines += ["", f"[{label}] {describe_source(chunk, topic, blend)}", chunk["text"].strip()]

    examples = state.get("style_examples") or []
    if examples:
        lines += ["", "PAST EXAM QUESTIONS -- copy the style and depth, never the content"]
        for q in examples:
            lines += ["", f"({q.get('marks') or '?'} marks, {q.get('bloom_level') or 'Bloom level not given'})",
                      clean_example(q.get("text"))]

    feedback = state.get("revision_feedback")
    if feedback:
        lines += ["", "YOUR PREVIOUS DRAFT WAS REJECTED", f"Reason: {feedback}"]
        previous = (state.get("draft") or {}).get("question")
        if previous:
            lines += ["Previous draft:", previous]
        lines += ["Write a new question that fixes this."]

    lines += ["", reply_format(marks, blend)]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


# ---------------------------------------------------------------- checking the reply
def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _whole_number(value):
    """10, 10.0, "10" and "10 marks" -> 10; anything else -> None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(\d+)\s*(?:marks?)?\s*", value, re.IGNORECASE)
        return int(m.group(1)) if m else None
    return None


def _source_label(value):
    """"S1", "[S1]", "s1", "Source 1" -> "S1"; anything else -> None."""
    m = re.fullmatch(r"\s*\[?\s*s(?:ource)?\s*\[?(\d+)\]?\s*\]?\s*", str(value), re.IGNORECASE)
    return f"S{int(m.group(1))}" if m else None


def parse_draft(obj, *, marks, sources, blend=None):
    """Check the model's JSON and return the draft in a fixed shape:

        {"question", "parts": [{"label", "text", "marks"}], "marks",
         "answer_outline": [...], "sources_used": ["S1", ...],
         "source_ids": [chunk ids], "blend_used"}

    Raises ValueError, with a reason written for the model, when a check fails."""
    question = _text(obj.get("question"))
    if not question:
        raise ValueError('"question" is empty')

    total = _whole_number(obj.get("marks"))
    if total != marks:
        raise ValueError(f'"marks" is {obj.get("marks")!r}, but the question must be worth exactly {marks} marks')

    raw_parts = obj.get("parts") or []
    if not isinstance(raw_parts, list):
        raise ValueError('"parts" must be a list ([] when there are no sub-parts)')
    parts = []
    for i, part in enumerate(raw_parts):
        if not isinstance(part, dict) or not _text(part.get("text")):
            raise ValueError(f'part {i + 1} has no "text"')
        part_marks = _whole_number(part.get("marks"))
        if not part_marks or part_marks < 0:
            raise ValueError(f'part {i + 1} needs whole-number "marks" above 0')
        label = _text(part.get("label")).strip("().").lower()
        parts.append({"label": label or PART_LABELS[i % len(PART_LABELS)],
                      "text": _text(part["text"]), "marks": part_marks})
    if parts and sum(p["marks"] for p in parts) != marks:
        raise ValueError(f"the parts add up to {sum(p['marks'] for p in parts)} marks, not {marks}")

    outline = obj.get("answer_outline")
    outline = [outline] if isinstance(outline, str) else outline if isinstance(outline, list) else []
    outline = [_text(step) for step in outline if _text(step)]
    if not outline:
        raise ValueError('"answer_outline" is empty: list what a full-marks answer must contain')

    cited = obj.get("sources_used") or []
    cited = re.split(r"[,;]", cited) if isinstance(cited, str) else cited if isinstance(cited, list) else []
    used = []
    for value in cited:
        label = _source_label(value)
        if label in sources and label not in used:
            used.append(label)
    if not used:
        raise ValueError(f'"sources_used" names none of the given sources ({", ".join(sources)})')

    roles = {sources[label].get("role") for label in used}
    if "main" not in roles:
        main = [label for label, c in sources.items() if c.get("role") == "main"]
        raise ValueError(f"the question must use at least one source about the main topic ({', '.join(main)})")

    blend_used = _text(obj.get("blend_used"))
    blend_sources = [label for label, c in sources.items() if c.get("role") == "blend"]
    if blend and blend_sources:
        if "blend" not in roles:
            raise ValueError(f'the question must also use a source about the prerequisite '
                             f'"{blend["name"]}" ({", ".join(blend_sources)}) and list it in "sources_used"')
        if not blend_used:
            raise ValueError(f'"blend_used" is empty: say where a correct answer needs "{blend["name"]}"')

    return {"question": question, "parts": parts, "marks": total, "answer_outline": outline,
            "sources_used": used, "source_ids": [sources[label].get("id") for label in used],
            "blend_used": blend_used if blend else ""}


# ---------------------------------------------------------------- the node
def make_draft_node(llm, temperature=0.7):
    """Build the node. `llm` is a pipeline.llm.LLMClient (or anything with the
    same complete_json method)."""

    def draft_node(state):
        chunks = state.get("relevant_chunks") or []
        if not any(c.get("role") == "main" for c in chunks):
            reason = state.get("grounding_warning") or "no grounding text for this topic"
            return {"draft": None, "draft_error": f"not drafted: {reason}"}

        sources = number_sources(chunks)
        attempt = state.get("draft_attempts", 0) + 1

        def check(obj):
            return parse_draft(obj, marks=state["target_marks"], sources=sources,
                               blend=state.get("blend_topic"))

        try:
            draft = llm.complete_json(build_messages(state), temperature=temperature, validate=check)
        except LLMFatalError:
            raise                       # wrong settings / no credit: stop, every question would fail
        except LLMError as exc:
            return {"draft": None, "draft_error": str(exc), "draft_attempts": attempt}

        draft["model"] = getattr(llm, "model", "?")
        return {"draft": draft, "draft_error": None, "draft_attempts": attempt}

    return draft_node
