"""
Step 4 -- quality_check (is the draft grounded, answerable and clear?).

    draft gave:          a question + answer outline + the sources it cites
    quality_check makes: a report, pass / fail, and -- on a fail -- the
                         feedback draft uses to write a better question

A language model (the "checker") reads the draft and the same numbered
sources [S1] [S2] ... the writer had. It lists the facts the question text
relies on, and then goes through the answer outline LINE BY LINE, listing
the facts each line relies on. For every fact it names the source that
states it AND copies the words that state it. Then this code -- not the
model -- decides:

  * a fact is SUPPORTED only if the copied words really are in that source
    (quote_in_source). A checker that says "S4 supports this" without being
    able to show where is not believed. This is what catches drafts like
    "Extract-Max: move A[n] to A[0], then max-heapify", which the model knew
    but none of the sources said.
  * not every claim is a fact to look up. KNOWLEDGE -- what a student must
    have learned (a definition, rule, formula, complexity, a known algorithm)
    -- needs the quote above. WORKING -- what the student produces by
    applying that knowledge to THIS question (the steps of an algorithm they
    are asked to design, a computed value, a recurrence derived for a
    variant) needs no quote: it must name the sources it builds on, and the
    checker must judge that it follows from them correctly. Asking for a
    quote for working rejected a good rod-cutting question because "keep the
    larger price in P[length]" -- the very thing the student was asked to
    design -- is in no book. A wrong step still fails, as "does not follow".
  * every line of the answer outline must have its facts listed. When the
    checker could pick which facts to list, it skipped some: an accepted
    draft's outline said "array storage gives better cache locality", which
    no source says and the checker never listed.
  * the question must also be answerable from the sources, have one
    reasonable reading and one answer two examiners would mark alike (an
    open "how could X be adapted?" part fails this), not give its own answer
    away in the stem, ask for work that fits its marks, and -- when there is
    a prerequisite to blend in -- really need it.

TWO GROUNDING MODES (GROUNDING in .env; pipeline/graph.py passes it in)

  balanced (default) -- strict on the QUESTION, light on the ANSWER:
    * a fact the question text states must be quoted from a source, or be
      judged by the checker both correct and standard for this course and
      semester ("model knowledge"). Wrong, or beyond the course level: fail.
    * the answer outline is only evidence that the question works: its facts
      need no quote; only one the checker finds wrong fails. It is kept as a
      draft answer key, to check before use.
    * anchor: at least one fact must be quoted from a source about the main
      topic, so every question is built on the course material.
  strict -- every knowledge claim, in the question and in the answer, needs
    a verified quote (and the anchor too).

Each claim gets a status: grounded | model knowledge | working (all fine) or
not in sources | wrong | beyond level | unconfirmed | wrong step | no basis.
The model-knowledge ones are listed in the report so a reviewer checks just those.

Pass -> the graph goes on. Fail -> revision_feedback says what was wrong, and
the graph sends the question back to draft (pipeline/graph.py decides how
many times).

The checker can be a different model from the writer (LLM_CHECK_MODEL in
.env): a model checking its own draft tends to agree with itself.

Reads:  draft, relevant_chunks, current_topic, blend_topic, target_marks
Writes: quality (the report), quality_passed, revision_feedback,
        quality_reports (every report so far, for this question)
"""
import re

from pipeline.llm import LLMError, LLMFatalError
from pipeline.nodes.draft import number_sources

GROUNDING_MODES = ("balanced", "strict")
OK_STATUSES = ("grounded", "model knowledge", "working")
QUOTE_MIN_WORDS = 2           # a shorter "quote" proves nothing
QUOTE_MAX_WORDS = 40

SYSTEM_PROMPT = """\
You check exam questions written by someone else against the source texts \
they were allowed to use. Be strict: a fact counts as supported only if a source \
states it or shows it directly -- not if it is merely true, or well known, or \
implied. Copy quotes exactly, character for character. Reply with ONE JSON object \
and nothing else."""

BALANCED_SYSTEM_PROMPT = """\
You check exam questions written by someone else. Be strict about the QUESTION: \
everything it states must be correct, it must be answerable with one clear answer, \
and it must suit the course and semester. Check the answer outline only to see \
whether the question really works: its facts must be correct, but need not be \
quoted. Copy quotes exactly, character for character. Reply with ONE JSON object \
and nothing else."""

CLAIM_SHAPE = ('{"claim": "...", "kind": "knowledge", "source": "S1", "quote": "exact words from S1", '
               '"correct": true, "standard": true}, '
               '{"claim": "...", "kind": "working", "builds_on": ["S1", "S4"], "follows": true}')


# ---------------------------------------------------------------- the prompt
def draft_as_text(draft):
    lines = [draft["question"]]
    lines += [f"({p['label']}) {p['text']}   [{p['marks']} marks]" for p in draft.get("parts", [])]
    return "\n".join(lines)


def build_messages(state, grounding="balanced"):
    draft, blend = state["draft"], state.get("blend_topic")
    topic = state.get("current_topic") or {}
    outline = draft["answer_outline"]
    balanced = grounding == "balanced"
    lines = [f'COURSE: semester {topic.get("semester", "?")}, topic "{topic.get("name", "?")}"', "",
             f"QUESTION TO CHECK ({draft['marks']} marks)", draft_as_text(draft), "",
             f"ANSWER OUTLINE ({len(outline)} lines: what the writer says a full-marks answer contains)"]
    lines += [f"{n}. {step}" for n, step in enumerate(outline, start=1)]
    if blend:
        lines += ["", f'PREREQUISITE THE QUESTION MUST NEED: "{blend["name"]}"']
    lines += ["", "SOURCES -- the course material the question is built on" if balanced
              else "SOURCES -- the only texts the question may rely on"]
    for label, chunk in number_sources(state.get("relevant_chunks") or []).items():
        lines += ["", f"[{label}]", chunk["text"].strip()]
    lines += [
        "", "WHAT TO DO",
        "A claim is anything the question or its answer depends on. Each claim is one of two kinds:",
        "- KNOWLEDGE: something a student must have learned -- a definition, rule, property, formula, "
        "complexity, advantage, a known algorithm or one of its steps. Examples: \"counting sort runs "
        "in O(n + k) time\", \"HEAP-EXTRACT-MAX moves the last element to the root\", \"r_n = max(p_i + "
        "r_{n-i})\". Give the source that states it (\"S1\", \"S2\", ...) and copy the words of that "
        f"source that state it (at most {QUOTE_MAX_WORDS} words, exactly as written). If no source "
        "states it, give source null and quote \"\". Also give correct: is it true? and standard: is "
        f"it taught in a typical semester-{topic.get('semester', '?')} course on this topic (not "
        "research level, not from a later course)?",
        "- WORKING: what the student produces by applying knowledge to THIS question -- the steps of "
        "an algorithm the question asks them to design, a value they compute, a trace, a recurrence "
        "they derive for a variant the question describes. Example: for \"design a way to keep the "
        "best price for each length\", the step \"scan the pieces and keep the larger price in "
        "P[length]\". Give builds_on: the sources whose knowledge it uses, and follows: true only if "
        "it correctly follows from that knowledge and the question's data.",
        "A fact a student is expected to KNOW is knowledge even when it appears in the answer "
        "outline. When unsure, it is knowledge.",
        *(["The question may use correct, standard facts that no source states (give source null). "
           "Its central idea must still come from a source about the main topic."] if balanced else []),
        "1. question_claims: the claims the question text itself states or takes for granted "
        "([] if none).",
        f"2. outline: ONE entry for EACH of the {len(outline)} outline lines, in order, listing "
        "EVERY claim in that line -- including any advantage, reason or comparison it gives. Never "
        "leave a line out and never give a line no claims.",
        ("3. answerable: can every part be answered fully from the sources and standard knowledge of "
         "this course, plus ordinary reasoning?") if balanced else
        "3. answerable: can every part be answered fully from the sources plus ordinary reasoning?",
        "4. single_reading: does every part have exactly one reasonable reading AND one answer that "
        "two examiners would mark the same way? An open part ('how could X be adapted', 'suggest "
        "a way', 'discuss') with many acceptable answers is false. A design part ('propose a data "
        "structure for ...') is fine only when the sources make one answer clearly the best.",
        "5. gives_away: does the question text already state what a part asks the student to "
        "explain, derive or find? (e.g. the stem says 'deletion takes O(n)' and part (a) asks "
        "why deletion takes O(n))",
        "6. marks_fit: is the work asked for right for the marks given?",
        "7. blend_needed: does a correct answer genuinely need the prerequisite? "
        "(null when no prerequisite is given)",
        "",
        "Reply with ONLY this JSON object:",
        "{",
        f'  "question_claims": [{CLAIM_SHAPE}],',
        f'  "outline": [{{"line": 1, "claims": [{CLAIM_SHAPE}]}}],',
        '  "answerable": true, "answerable_reason": "",',
        '  "single_reading": true, "ambiguity": "",',
        '  "gives_away": false, "gives_away_reason": "",',
        '  "marks_fit": true, "marks_reason": "",',
        '  "blend_needed": true, "blend_reason": "",',
        '  "feedback": "if anything is wrong: what the writer should change"',
        "}"]
    return [{"role": "system", "content": BALANCED_SYSTEM_PROMPT if balanced else SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


# ---------------------------------------------------------------- checking the checker
def normalise(text):
    """Lower case, every run of non-letters/digits -> one space, and a word
    broken at a line end ("an-\\nalyze") joined again."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text or "")
    return " ".join(re.findall(r"[^\W_]+", text.lower()))


# CLRS prints pseudocode line numbers on lines of their own, so the chunk text
# of "if n == 0 / return 0" is "1 / if n == 0 / 2 / return 0"
LINE_NUMBER_RE = re.compile(r"^[ \t]*\d{1,2}[ \t]*$", re.MULTILINE)


def quote_in_source(quote, source_text):
    """Are these words really in the source? Small copying differences
    (case, spacing, punctuation, a hyphenated line break, a pseudocode line
    number in between) do not matter; "..." may stand for left-out words, but
    every piece must be there, in order."""
    pieces = [normalise(p) for p in re.split(r"\.\.\.|…|\[\.\.\.\]", quote or "")]
    pieces = [p for p in pieces if p]
    if sum(len(p.split()) for p in pieces) < QUOTE_MIN_WORDS:
        return False
    for source in {normalise(source_text), normalise(LINE_NUMBER_RE.sub("", source_text or ""))}:
        at = 0
        for piece in pieces:
            found = source.find(piece, at)
            if found < 0:
                break
            at = found + len(piece)
        else:
            return True
    return False


def _flag(obj, key):
    value = obj.get(key)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return value if isinstance(value, bool) or value is None else bool(value)


def _label(value):
    m = re.fullmatch(r"\s*\[?\s*s(?:ource)?\s*(\d+)\s*\]?\s*", str(value or ""), re.IGNORECASE)
    return f"S{int(m.group(1))}" if m else None


def _status(claim, raw, sources, mode):
    """grounded | model knowledge | working (fine), or why the claim fails."""
    in_question = claim["line"] is None
    if claim["kind"] == "working":
        if claim["follows"] is False:
            return "wrong step"
        if mode == "strict":
            if not claim["builds_on"]:
                return "no basis"
            return "working" if claim["follows"] is True else "wrong step"
        return "working"                          # balanced: only a wrong step fails
    if claim["source"] and quote_in_source(claim["quote"], sources[claim["source"]]["text"]):
        return "grounded"
    if mode == "strict":
        return "not in sources"
    correct, standard = _flag(raw, "correct"), _flag(raw, "standard")
    if correct is False:
        return "wrong"
    if not in_question:
        return "model knowledge"                  # balanced, answer outline: only a wrong fact fails
    if standard is False:
        return "beyond level"
    return "model knowledge" if correct is True and standard is True else "unconfirmed"


def _claims(raw_claims, where, sources, line=None, mode="strict"):
    """Read and verify a list of claims. `where` names the list in errors."""
    if not isinstance(raw_claims, list):
        raise ValueError(f"{where} must be a list of claims")
    out = []
    for i, raw in enumerate(raw_claims):
        if not isinstance(raw, dict) or not str(raw.get("claim") or "").strip():
            raise ValueError(f'{where}, claim {i + 1} has no "claim" text')
        claim = {"claim": str(raw["claim"]).strip(),
                 "line": line}                # outline line number; None = the question text
        if str(raw.get("kind") or "").strip().lower() == "working":
            builds_on = raw.get("builds_on") or []
            builds_on = re.split(r"[,;\s]+", builds_on) if isinstance(builds_on, str) else builds_on
            labels = [lb for lb in map(_label, builds_on if isinstance(builds_on, list) else [])
                      if lb in sources]
            follows = _flag(raw, "follows")
            claim.update(kind="working", builds_on=sorted(set(labels), key=labels.index),
                         follows=follows if isinstance(follows, bool) else None)
        else:                                 # knowledge -- also when "kind" is missing or unclear
            label = _label(raw.get("source"))
            claim.update(kind="knowledge", source=label if label in sources else None,
                         quote=str(raw.get("quote") or "").strip())
        # a quote is believed only when its words are found in the source it names
        claim["status"] = _status(claim, raw, sources, mode)
        claim["supported"] = claim["status"] in OK_STATUSES
        out.append(claim)
    return out


def parse_report(obj, *, sources, outline_lines, blend=None, mode="strict"):
    """The checker's JSON in a fixed shape, with every claim's quote checked
    against the source it names and every outline line accounted for.
    Raises ValueError (the checker is asked again, with the reason) when the
    reply is not a usable report."""
    claims = _claims(obj.get("question_claims") or [], '"question_claims"', sources, mode=mode)

    outline = obj.get("outline")
    if not isinstance(outline, list) or len(outline) != outline_lines:
        got = len(outline) if isinstance(outline, list) else "no"
        raise ValueError(f'"outline" must have one entry for each of the {outline_lines} answer-outline '
                         f'lines, in order (got {got})')
    for n, entry in enumerate(outline, start=1):
        if not isinstance(entry, dict) or not entry.get("claims"):
            raise ValueError(f"outline line {n} has no claims: list every claim in it "
                             f"(for pure working, the rule it applies)")
        claims += _claims(entry["claims"], f"outline line {n}", sources, line=n, mode=mode)

    for key in ("answerable", "single_reading", "marks_fit", "gives_away"):
        if not isinstance(_flag(obj, key), bool):
            raise ValueError(f'"{key}" must be true or false')
    blend_needed = _flag(obj, "blend_needed") if blend else None
    if blend and not isinstance(blend_needed, bool):
        raise ValueError('"blend_needed" must be true or false: a prerequisite was given')
    main = [label for label, c in sources.items() if c.get("role") == "main"]
    return {
        "mode": mode,
        "claims": claims,
        # built on the course material: some fact is quoted from a main-topic source
        "anchored": any(c["status"] == "grounded" and c["source"] in main for c in claims),
        "main_sources": main,
        "counts": {status: sum(c["status"] == status for c in claims) for status in OK_STATUSES},
        "answerable": _flag(obj, "answerable"),
        "answerable_reason": str(obj.get("answerable_reason") or "").strip(),
        "single_reading": _flag(obj, "single_reading"),
        "ambiguity": str(obj.get("ambiguity") or "").strip(),
        "gives_away": _flag(obj, "gives_away"),
        "gives_away_reason": str(obj.get("gives_away_reason") or "").strip(),
        "marks_fit": _flag(obj, "marks_fit"),
        "marks_reason": str(obj.get("marks_reason") or "").strip(),
        "blend_needed": blend_needed,
        "blend_reason": str(obj.get("blend_reason") or "").strip(),
        "checker_feedback": str(obj.get("feedback") or "").strip(),
    }


def problems(report, blend=None):
    """What is wrong with the draft, one sentence each, written for the
    writer. An empty list means the draft passes."""
    out = []

    def where(c):
        return "in the question" if c["line"] is None else f"outline line {c['line']}"

    def having(*statuses):
        return [c for c in report["claims"] if c["status"] in statuses]

    def listed(claims):
        return "; ".join(f'"{c["claim"]}" ({where(c)})' for c in claims)

    unsupported = having("not in sources")
    if unsupported:
        text = "; ".join(
            f'"{c["claim"]}" ({where(c)}: '
            + (f'{c["source"]} does not say this, the quoted words are not in it)' if c["source"]
               else "in no source)")
            for c in unsupported)
        out.append(f"These facts are not stated in the sources: {text}. Use only what the "
                   f"sources state, or change the question so it does not need them.")
    if having("wrong"):
        out.append(f"These facts are wrong: {listed(having('wrong'))}. Correct them.")
    if having("beyond level"):
        out.append(f"These facts go beyond what this course teaches at this semester: "
                   f"{listed(having('beyond level'))}. Keep to what the course teaches.")
    if having("unconfirmed"):
        out.append(f"These facts in the question are neither quoted from a source nor confirmed "
                   f"correct and standard for this course: {listed(having('unconfirmed'))}.")
    wrong = having("wrong step", "no basis")
    if wrong:
        text = "; ".join(
            f'"{c["claim"]}" ({where(c)}: '
            + ("builds on no source)" if c["status"] == "no basis"
               else "does not follow" + (f' from {", ".join(c["builds_on"])})' if c["builds_on"] else ")"))
            for c in wrong)
        out.append(f"These steps are wrong, or do not follow from the sources: {text}. Fix them, "
                   f"or change the question.")
    if not report.get("anchored", True):
        main = ", ".join(report.get("main_sources") or []) or "none were given"
        out.append(f"The question is not built on the course material: none of its facts is quoted "
                   f"from a source about the main topic ({main}). Build it around what they say.")
    if not report["answerable"]:
        out.append("It cannot be fully answered from the sources" +
                   (f": {report['answerable_reason']}" if report["answerable_reason"] else "."))
    if not report["single_reading"]:
        out.append("It does not have one clear reading and one answer every examiner would mark alike" +
                   (f": {report['ambiguity']}" if report["ambiguity"] else "."))
    if report["gives_away"]:
        out.append("The question text already gives away what a part asks for" +
                   (f": {report['gives_away_reason']}" if report["gives_away_reason"] else "."))
    if not report["marks_fit"]:
        out.append("The work asked for does not fit the marks" +
                   (f": {report['marks_reason']}" if report["marks_reason"] else "."))
    if blend and report["blend_needed"] is False:
        out.append(f'A correct answer does not really need "{blend["name"]}"' +
                   (f": {report['blend_reason']}" if report["blend_reason"] else "."))
    return out


# ---------------------------------------------------------------- the node
def make_quality_check_node(checker, grounding="balanced"):
    """`checker` is a pipeline.llm.LLMClient (it may be a different model
    from the one that drafts). grounding: "balanced" or "strict" (see top)."""
    if grounding not in GROUNDING_MODES:
        raise ValueError(f"grounding must be one of {GROUNDING_MODES}, not {grounding!r}")

    def quality_check_node(state):
        draft = state.get("draft")
        if not draft:                       # nothing was drafted: nothing to check
            return {"quality": None, "quality_passed": False}

        sources = number_sources(state.get("relevant_chunks") or [])
        blend = state.get("blend_topic")
        # the prerequisite is only demanded when there was text about it to use
        if not any(c.get("role") == "blend" for c in sources.values()):
            blend = None

        try:
            report = checker.complete_json(
                build_messages(dict(state, blend_topic=blend), grounding), temperature=0.0,
                validate=lambda obj: parse_report(obj, sources=sources, blend=blend, mode=grounding,
                                                  outline_lines=len(draft["answer_outline"])))
        except LLMFatalError:
            raise
        except LLMError as exc:
            # the checker failed, not the draft: do not loop on it, and do not pass it either
            report = {"error": str(exc)}
            return {"quality": report, "quality_passed": False, "revision_feedback": None,
                    "quality_reports": (state.get("quality_reports") or []) + [report]}

        found = problems(report, blend)
        report["problems"] = found
        report["passed"] = not found
        report["model"] = getattr(checker, "model", "?")
        feedback = None
        if found:
            feedback = " ".join(found)
            if report["checker_feedback"]:
                feedback += f" Checker's advice: {report['checker_feedback']}"
        return {"quality": report, "quality_passed": not found, "revision_feedback": feedback,
                "quality_reports": (state.get("quality_reports") or []) + [report]}

    return quality_check_node
