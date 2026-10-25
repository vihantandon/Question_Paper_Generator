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
Writes: draft, draft_error, draft_attempts, drafts (every draft so far)
"""
import re

from pipeline.llm import LLMError, LLMFatalError

MAX_EXAMPLE_CHARS = 1200      # a past question with a long program is cut here in the prompt
PART_LABELS = "abcdefghij"
# a result left open in the answer key: "R[8]=?", "= ???", "TBD", "to be computed"
PLACEHOLDER_RE = re.compile(r"=\s*\?+|\?{2,}|\bTBD\b|\bto be (?:computed|calculated|determined|found)\b",
                            re.IGNORECASE)

SYSTEM_PROMPT = """\
You set exam questions for a B.Tech computer science course at an Indian university.

Rules:
1. Use ONLY facts from the numbered SOURCES. Every definition, rule, number, piece of code \
or behaviour the question or its answer depends on must be stated or directly shown in a \
source you cite. Do not add anything the sources do not support. This holds for every line \
of the answer outline too, including any advantage, reason or comparison it gives. Steps the \
student works out (a design, a computation, a trace, a derivation) are fine when they follow \
from facts in the sources.
2. The question must have one reasonable reading, and each part one answer that two examiners \
would mark the same way. Avoid open parts such as "how could X be adapted" or "discuss".
3. Do not state in the question what a part asks the student to explain, derive or find.
4. The PAST EXAM QUESTIONS show this university's style, wording and depth. Copy the style; \
never copy their content, data, code or wording.
5. Marks set the depth: a 2-mark question needs a few lines; a 6- or 10-mark question needs \
working, a trace, code or a derivation.
6. The answer outline is the answer key: work every part out to its final answer -- the \
actual numbers, output, order or plan, never "?", "the value" or "compute it". When the \
question gives data to work on, choose data where the method matters: the answer must not \
be obvious without carrying the method out (not a case where doing nothing is already best).
7. If the question has parts (a), (b), ..., the parts' marks must add up to the total.
8. Do not number the question ("Q1.") and do not write marks, CO or Bloom labels inside it; \
they are added when the paper is put together.
9. Reply with ONE JSON object and nothing else."""

# GROUNDING=balanced (the default): the question must be built on the sources,
# but may also use correct, standard facts of the course that they do not state.
_STRICT_RULE_1 = SYSTEM_PROMPT[SYSTEM_PROMPT.index("1. "):SYSTEM_PROMPT.index("2. ")]
BALANCED_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_STRICT_RULE_1, """\
1. Build the question on the numbered SOURCES: its central idea must come from a source about \
the main topic, and you must cite it. You may also use correct, standard facts that a course at \
this level teaches, even when no source states them -- never a fact you are unsure of, and \
nothing beyond the course level. Steps the student works out (a design, a computation, a trace, \
a derivation) must follow correctly from these facts.
""")


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
        '  "answer_outline": ["each step, fact or result a full-marks answer must contain, '
        'with the actual values worked out"],',
        '  "sources_used": ["S1", "S2"],',
        blend_line,
        "}",
        f'"parts" is [] when the question has no sub-parts; with parts, their marks add up to '
        f'{marks}. "marks" must be exactly {marks}. "sources_used" lists every source the '
        f'question and its answer rely on.',
    ])


def build_messages(state, grounding="balanced"):
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

    lines += ["", "SOURCES -- build the question on these" if grounding == "balanced"
              else "SOURCES -- the only facts you may use"]
    for label, chunk in number_sources(state.get("relevant_chunks") or []).items():
        lines += ["", f"[{label}] {describe_source(chunk, topic, blend)}", chunk["text"].strip()]

    examples = state.get("style_examples") or []
    if examples:
        lines += ["", "PAST EXAM QUESTIONS -- copy the style and depth, never the content"]
        for q in examples:
            lines += ["", f"({q.get('marks') or '?'} marks, {q.get('bloom_level') or 'Bloom level not given'})",
                      clean_example(q.get("text"))]

    earlier = state.get("earlier_questions") or []
    if earlier:
        lines += ["", "QUESTIONS ALREADY SET ON THIS TOPIC -- write a DIFFERENT problem: not the same "
                      "task or situation, not the same problem with new numbers"]
        lines += [f"- {_short(q['text'])}" for q in earlier]

    feedback = state.get("revision_feedback")
    if feedback:
        lines += ["", "YOUR PREVIOUS DRAFT WAS REJECTED", f"Reason: {feedback}"]
        previous = (state.get("draft") or {}).get("question")
        if previous:
            lines += ["Previous draft:", previous]
        lines += ["Write a new question that fixes this."]

    lines += ["", reply_format(marks, blend)]
    return [{"role": "system", "content": BALANCED_SYSTEM_PROMPT if grounding == "balanced" else SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


def _short(text, limit=300):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


# ---------------------------------------------------------------- checking the reply
def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _real_line_breaks(text):
    """A model sometimes writes "\\n" (backslash, n) where it means a new line:
    "...by length.\\n\\n(a) Sort..." came out with the backslashes showing.
    Only in text with no real line break, and never inside "double quotes",
    where it may be C code: printf("%d\\n", x)."""
    if "\n" in text or "\\n" not in text:
        return text
    pieces = text.split('"')
    pieces[::2] = [p.replace("\\n", "\n") for p in pieces[::2]]
    return '"'.join(pieces)


def _repeats_parts(question, parts):
    """Does the question text also write out every part, each on a line of
    its own starting "(a)" ...? ("... created using (a) array and (b) tree"
    inside a sentence does not count.)"""
    return bool(parts) and all(
        re.search(rf"(?m)^\s*\(?{re.escape(p['label'])}[).]\s", question) for p in parts)


CITATION_RE = re.compile(r"[(\[]\s*(S\d{1,2}(?:\s*(?:[,;/&]|and)\s*S\d{1,2})*)\s*[)\]]")


def _cited_in(texts):
    """Source labels cited in brackets: "(S6)", "[S2]", "(S4, S1)". A bare
    "S1" is not counted: it may be a string's name in the question."""
    out = []
    for text in texts:
        for group in CITATION_RE.findall(text or ""):
            out += [f"S{int(n)}" for n in re.findall(r"S(\d{1,2})", group) if f"S{int(n)}" not in out]
    return out


def _strip_repeated_parts(question, parts):
    """The question with the parts it writes out again cut off its end -- when
    what follows the first "(a)" line is only the parts (as "parts" gives
    them, with marks). None when there is more there, or no text before it."""
    first = re.search(rf"(?m)^\s*\(?{re.escape(parts[0]['label'])}[).]\s", question)
    if not first or not question[:first.start()].strip():
        return None
    def words(text):
        return set(re.findall(r"[a-z0-9]+", text.lower())) - {"marks", "mark"}
    tail = words(question[first.start():])
    extra = tail - words(" ".join(f"{p['label']} {p['text']} {p['marks']}" for p in parts))
    return question[:first.start()].rstrip() if len(extra) <= max(2, len(tail) // 10) else None


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


def _fold_headings(lines):
    """A heading line ("DP table computation:") joins the line after it: it
    states nothing itself, and the checker cannot list a claim for it."""
    out = []
    for line in lines:
        if out and out[-1].endswith(":") and len(out[-1].split()) <= 6:
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out


def parse_draft(obj, *, marks, sources, blend=None):
    """Check the model's JSON and return the draft in a fixed shape:

        {"question", "parts": [{"label", "text", "marks"}], "marks",
         "answer_outline": [...], "sources_used": ["S1", ...],
         "source_ids": [chunk ids], "blend_used"}

    Raises ValueError, with a reason written for the model, when a check fails."""
    question = _real_line_breaks(_text(obj.get("question")))
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
                      "text": _real_line_breaks(_text(part["text"])), "marks": part_marks})
    if parts and sum(p["marks"] for p in parts) != marks:
        raise ValueError(f"the parts add up to {sum(p['marks'] for p in parts)} marks, not {marks}")
    if _repeats_parts(question, parts):
        stem = _strip_repeated_parts(question, parts)
        if stem:                    # cut off in code: asking the model again costs a whole request
            question = stem
    if _repeats_parts(question, parts):
        labels = ", ".join(f"({p['label']})" for p in parts)
        raise ValueError(f'"question" also writes out the parts {labels}: put only the common text '
                         f'before the parts in "question", and each part only in "parts"')
    # task lines ("1. The sorted list ...") -- not numbered code lines ("1. int x = 5;")
    numbered = re.findall(r"(?m)^\s*\(?[1-9][.)]\s+[A-Z][a-z]+\b[^;{}=<>#\n]*$", question)
    if not parts and len(numbered) >= 2:
        raise ValueError('"question" lists numbered tasks (1., 2., ...) with no marks: write them as '
                         '"parts" (a), (b), ... each with its marks, adding up to the total')
    if len(parts) == 1:
        # one part worth all the marks is not a part: "(a) Perform ... [10]"
        question, parts = f"{question}\n{parts[0]['text']}", []

    outline = obj.get("answer_outline")
    outline = [outline] if isinstance(outline, str) else outline if isinstance(outline, list) else []
    outline = _fold_headings([_text(step) for step in outline if _text(step)])
    if not outline:
        raise ValueError('"answer_outline" is empty: list what a full-marks answer must contain')
    for n, step in enumerate(outline, start=1):
        m = PLACEHOLDER_RE.search(step)
        if m:
            raise ValueError(f'answer_outline line {n} leaves its result open ("{m.group().strip()}"): '
                             f'work it out and write the actual value')

    cited = obj.get("sources_used") or []
    cited = re.split(r"[,;]", cited) if isinstance(cited, str) else cited if isinstance(cited, list) else []
    used = []
    for value in cited:
        label = _source_label(value)
        if label in sources and label not in used:
            used.append(label)
    # a source the answer outline or blend_used cites -- "(S6)", "(S4, S1)" -- but
    # "sources_used" forgot: added here, instead of a whole new request
    for label in _cited_in(outline + [_text(obj.get("blend_used"))]):
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
# the writer would not use the prerequisite's sources
BLEND_PROBLEMS = ("must also use a source about the prerequisite", '"blend_used" is empty')


def without_blend(state):
    """The state for a question on the main topic alone: no prerequisite, its
    sources left out (main sources come first, so S1, S2 ... keep their numbers)."""
    from pipeline.difficulty_blend import build_question_prompt
    topic = state["current_topic"]
    return dict(state, blend_topic=None,
                relevant_chunks=[c for c in state.get("relevant_chunks") or [] if c.get("role") == "main"],
                prompt_instruction=build_question_prompt(topic, None, state.get("target_difficulty"),
                                                         state["target_marks"]))


def make_draft_node(llm, temperature=0.7, grounding="balanced"):
    """Build the node. `llm` is a pipeline.llm.LLMClient (or anything with the
    same complete_json method).

    When the writer will not use the prerequisite's sources in any of its
    tries, it is asked once more for a question on the main topic alone: a
    paper with a plain question beats a paper with a gap. The draft then says
    "blend_dropped", and the check does not demand the prerequisite.
    A reply that could not be used at all ends this draft (draft_error); the
    graph then starts a fresh draft, up to MAX_DRAFTS (pipeline/graph.py)."""

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

        dropped = None
        try:
            draft = llm.complete_json(build_messages(state, grounding), temperature=temperature,
                                      validate=check)
        except LLMFatalError:
            raise                       # wrong settings / no credit: stop, every question would fail
        except LLMError as exc:
            blend = state.get("blend_topic")
            if not (blend and any(p in str(exc) for p in BLEND_PROBLEMS)):
                return {"draft": None, "draft_error": str(exc), "draft_attempts": attempt,
                        "revision_feedback": f"Your last reply could not be used ({exc}). Follow the reply "
                                             f"format and the rules exactly."}
            plain = without_blend(state)
            try:
                draft = llm.complete_json(build_messages(plain, grounding), temperature=temperature,
                                          validate=lambda obj: parse_draft(obj, marks=state["target_marks"],
                                                                           sources=number_sources(
                                                                               plain["relevant_chunks"])))
            except LLMFatalError:
                raise
            except LLMError as again:
                return {"draft": None, "draft_error": str(again), "draft_attempts": attempt}
            dropped = (f'no link to "{blend["name"]}": the writer did not use its sources, so this is a '
                       f'question on the main topic alone')

        draft["model"] = getattr(llm, "model", "?")
        if dropped:
            draft["blend_dropped"] = dropped
        return {"draft": draft, "draft_error": None, "draft_attempts": attempt,
                "drafts": (state.get("drafts") or []) + [draft]}

    return draft_node
