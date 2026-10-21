"""
Step 5 -- classify (which CO, which Bloom level?).

    quality_check gave: an accepted question
    classify makes:     its course outcome and Bloom level, each with a reason.
                        Its difficulty is the one the teacher asked for
                        (target_difficulty): the blend of earlier topics
                        already makes it so, and asking the model for it again
                        cost tokens for a guess (it said "medium" for 51 of 53
                        PYQs, then "easy" for 34).

A language model reads the question (with its marks), the subject's course
outcomes -- ALL of them, not only the ones its topic is mapped to -- and a
few real questions from the college's papers with the CO the examiners gave
them (one per CO: the one sharing most words with this question). It answers:
  co               the one CO the question mainly assesses
  bloom_read       the Bloom level of what the student must DO for most of
                   the marks, read from the question itself

It is told nothing else: not the topic, not the difficulty that was asked
for, not the level the college gives each CO. That keeps it a fair, separate
opinion, and it is the same classifier pipeline/eval_classify.py scores
against the labels printed on the real papers (the PYQs).

THE LABELS THIS CODE MAKES FROM THE ANSWER
  bloom       the Bloom level the college gives the chosen CO (co_levels in
              the subject YAML). The papers print a question's CO level as
              its Bloom level (52 of the 53 PYQs), so this is the label the
              paper would carry. bloom_read stays as a second opinion; the
              model's own reading matched only 27 of 53 paper labels, the CO
              level 37. Without a co_level the reading is used (bloom_source
              says which).
  co (in the pipeline)
              a generated question was written for a topic, and the syllabus
              maps that topic to its COs. The model picks from ALL the
              subject's COs (co_free; that is what the evaluation scores),
              and when its pick is not one of the topic's COs the topic wins:
              its only CO, or -- for a topic with several -- the one whose
              level is closest to how the question reads. On the PYQs the
              free pick was right for only 2 of 8 ALGO questions, while a
              generated question's topic is known for certain.
  notes       when the topic's CO replaced the model's pick, and when the
              question reads as a different level than its CO's. For the reviewer (and, later, the
              paper planner); they do not reject the question.

keyword_classify() is a rule-based stand-in (Bloom verbs, CO word overlap),
used by the evaluation as the score to beat.

Reads:  draft, current_topic, topic_cos, target_difficulty (copied as the difficulty)
Writes: classification
"""
import json
import re
from pathlib import Path

from pipeline.curriculum import BLOOM_LEVELS, ROOT, bloom_rank, load_subjects, normalize_bloom, subject_of_topic
from pipeline.llm import LLMError, LLMFatalError
from pipeline.nodes.quality_check import draft_as_text

# change it when the prompt changes: pipeline/eval_classify.py keeps answers
# only from the same version
PROMPT_VERSION = 3
PYQ_DIR = ROOT / "Extracting" / "Processed" / "pyq_structured"
EXAMPLE_CHARS = 350           # each example question is cut to this length

SYSTEM_PROMPT = """\
You are an experienced university examiner. You label exam questions for the \
course file: which course outcome a question assesses and its Bloom's taxonomy \
level. Reply with ONE JSON object and nothing else."""

BLOOM_GUIDE = (
    ("Remember", "recall a definition, fact, list or syntax as it was taught."),
    ("Understand", "explain, describe or illustrate an idea in their own words; give an example; "
                   "say what something means or why it is used."),
    ("Apply", "carry out a method they were taught on the data given: trace or run an algorithm, "
              "insert / delete / sort step by step, compute a result, write code for a clearly "
              "specified task with constructs they learned."),
    ("Analyze", "take apart given code or a given algorithm: work out its output, find its errors, "
                "derive its running time or recurrence; compare methods by how they work."),
    ("Evaluate", "judge and justify: decide which method suits a situation and defend the choice, "
                 "prove or disprove a claim."),
    ("Create", "design something new for a stated problem when no method is given: an algorithm, "
               "a program's structure, a class diagram, a data structure."),
)


# ---------------------------------------------------------------- examples from the papers
QUESTION_NUMBER_RE = re.compile(r"^\s*\**\s*Q(?:uestion)?\s*\.?\s*\d+\s*[.):\-]?\s*\**\s*", re.IGNORECASE)


def clean_question(text):
    """Drop the "Q.2 " / "**Q3.**" / "Question 4:" a PYQ starts with."""
    return QUESTION_NUMBER_RE.sub("", text or "", count=1).strip()


def load_pyqs(folder=PYQ_DIR):
    """Every PYQ question with a CO and a Bloom label:
    [{"id", "subject", "text", "marks", "label_co", "label_bloom"}].
    [] when the folder is missing."""
    out = []
    for path in sorted(Path(folder).glob("*_pyq.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            meta = record.get("metadata") or {}
            if meta.get("type") != "question" or not meta.get("course_outcome") or not meta.get("bloom_level"):
                continue
            out.append({"id": record["id"], "subject": meta["subject"], "text": clean_question(record["document"]),
                        "marks": meta.get("marks"), "label_co": meta["course_outcome"],
                        "label_bloom": meta["bloom_level"]})
    return out


def pick_examples(question, subject_id, pyqs, exclude_id=None):
    """For each CO of the subject, the PYQ with that CO that shares the most
    words with `question` -- never the question itself (exclude_id, or the
    same text), so the evaluation cannot show the model its own answer."""
    q, same = _words(question), " ".join(question.split())
    best = {}
    for p in pyqs:
        if p["subject"] != subject_id or p["id"] == exclude_id or " ".join(p["text"].split()) == same:
            continue
        overlap = len(q & _words(p["text"]))
        if p["label_co"] not in best or overlap > best[p["label_co"]][0]:
            best[p["label_co"]] = (overlap, p)
    return [best[co][1] for co in sorted(best, key=lambda c: (len(c), c))]


def _short(text, limit=EXAMPLE_CHARS):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


# ---------------------------------------------------------------- the prompt
def build_messages(question, marks, subject, examples=()):
    """question: the full question text (with its parts); subject: from
    pipeline.curriculum.load_subjects(); examples: from pick_examples()."""
    semester = subject.get("semester") or "?"
    lines = [f"COURSE: {subject['name']} (semester {semester})", "", "COURSE OUTCOMES"]
    lines += [f"{co}: {text}" for co, text in subject["cos"].items()]
    if examples:
        lines += ["", "QUESTIONS FROM THIS COURSE'S PAPERS, WITH THE CO THE EXAMINERS GAVE THEM "
                      "(they show how the examiners read the COs)"]
        lines += [f"- [{e['label_co']}] ({e['marks']} marks) {_short(e['text'])}" for e in examples]
    lines += ["", "QUESTION TO LABEL" + (f" ({marks} marks)" if marks else ""), question.strip(), "",
              "WHAT TO DO",
              "1. co: the ONE course outcome this question mainly assesses: the one whose wording best "
              "matches what the student must DO for most of the marks, not just the topic it is about. "
              "Read the COs the way the examiners above did.",
              "2. bloom: the Bloom's taxonomy level of what the student must do for most of the marks:"]
    lines += [f"   {name} -- {text}" for name, text in BLOOM_GUIDE]
    lines += ["", "Reply with ONLY this JSON object:",
              '{"co": "CO1", "co_reason": "one sentence", "bloom": "Apply", "bloom_reason": "one sentence"}']
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


# ---------------------------------------------------------------- the reply
def _co(value, co_ids):
    """"CO4" / "co 4" / "ALGO:CO4" -> "CO4", if the subject has it."""
    m = re.search(r"co\s*0*(\d+)", str(value or ""), re.IGNORECASE)
    co = f"CO{int(m.group(1))}" if m else None
    if co not in co_ids:
        raise ValueError(f'"co" must be one of {", ".join(co_ids)} (got {value!r})')
    return co


def parse_classification(obj, co_ids):
    """The model's JSON in a fixed shape. Raises ValueError (the model is
    asked again, with the reason) when a label is missing or unknown."""
    bloom = normalize_bloom(obj.get("bloom"))
    if bloom is None:
        raise ValueError(f'"bloom" must be one of {", ".join(BLOOM_LEVELS)} (got {obj.get("bloom")!r})')
    return {"co": _co(obj.get("co"), co_ids),
            "co_reason": str(obj.get("co_reason") or "").strip(),
            "bloom_read": bloom,
            "bloom_reason": str(obj.get("bloom_reason") or "").strip()}


def with_course_details(result, subject):
    """Add the CO's wording and level; the Bloom label is the CO's level."""
    level = subject.get("co_levels", {}).get(result["co"])
    return dict(result, subject=subject["id"], co_text=subject["cos"].get(result["co"], ""),
                co_level=level, bloom=level or result["bloom_read"],
                bloom_source="co level" if level else "question")


def classify(llm, question, marks, subject, examples=()):
    """Ask the model; returns the labels plus co_text, co_level, bloom and
    model. LLMError when the model gives no usable answer."""
    result = llm.complete_json(build_messages(question, marks, subject, examples), temperature=0.0,
                               validate=lambda obj: parse_classification(obj, list(subject["cos"])))
    return dict(with_course_details(result, subject), model=getattr(llm, "model", "?"))


# ---------------------------------------------------------------- the rule-based stand-in
# highest level first: "design an algorithm ... and analyse it" is Create
BLOOM_WORDS = (
    ("Create", r"design|devise|formulate|propose|develop|give an? (?:efficient )?algorithm"
               r"|construct an? (?:algorithm|solution)|class diagram"),
    ("Evaluate", r"justify|evaluate|assess|critique|prove|disprove|which (?:one )?is better|defend"),
    ("Analyze", r"analy[sz]e|complexity|running time|compare|differentiate|distinguish|output of"
                r"|find the (?:error|bug)|debug|recurrence"),
    ("Apply", r"apply|compute|calculate|trace|solve|implement|insert|delete|sort|simulate|perform|use"
              r"|write (?:a|an|the)? ?(?:c\+\+ |c |sql )?(?:program|function|code|query|method|class)"),
    ("Understand", r"explain|describe|discuss|illustrate|summari[sz]e|interpret|what do you mean"
                   r"|difference between|give an example|why"),
    ("Remember", r"define|list|state|name|recall|what is"),
)
_STOP = set("that this with from into their there which while where what when have will been were "
            "given using used such each other these those about various different like".split())


def _words(text):
    """Content words, roughly stemmed: "implementing" and "implement" match."""
    out = set()
    for w in re.findall(r"[a-z]{4,}", text.lower()):
        if w not in _STOP:
            out.add(re.sub(r"(?:ing|ion|ions|ed|es|s)$", "", w) or w)
    return out


def keyword_classify(question, subject):
    """Bloom from the verbs the question uses (the highest level found;
    Apply when none), CO from the words it shares with each CO's wording
    (a CO of that Bloom level counts extra)."""
    low = question.lower()
    bloom = next((level for level, words in BLOOM_WORDS if re.search(rf"\b(?:{words})", low)), "Apply")
    q = _words(question)
    levels = subject.get("co_levels", {})
    scores = {co: len(q & _words(text)) + (1.5 if levels.get(co) == bloom else 0)
              for co, text in subject["cos"].items()}
    co = max(scores, key=lambda c: (scores[c], -list(scores).index(c)))
    result = {"co": co, "co_reason": "shares the most words with this CO", "bloom_read": bloom,
              "bloom_reason": "the highest Bloom verb in the question"}
    return dict(with_course_details(result, subject), model="keywords")


# ---------------------------------------------------------------- the node
def topic_cos_of(state, subject):
    """The COs the question's topic is mapped to (from graph_lookup, else the YAML)."""
    cos = [c.get("short_id") or str(c.get("co_id", "")).split(":")[-1] for c in state.get("topic_cos") or []]
    topic_id = (state.get("current_topic") or {}).get("id")
    if not cos and topic_id in subject["topics"]:
        cos = subject["topics"][topic_id]["cos"]
    return [co for co in cos if co in subject["cos"]]


def choose_topic_co(result, topic_cos, subject):
    """The model's pick when the topic is mapped to it (or the topic's COs
    are unknown); otherwise the topic's CO whose level is closest to how the
    question reads (the first one listed on a tie)."""
    free = result["co"]
    if not topic_cos or free in topic_cos:
        return free
    levels, read = subject.get("co_levels", {}), bloom_rank(result["bloom_read"])
    return min(topic_cos, key=lambda co: (abs(bloom_rank(levels[co]) - read) if co in levels else 9,
                                          topic_cos.index(co)))


def notes_for(result, state):
    """What a reviewer should look at. Not reasons to reject."""
    notes = []
    if result.get("co_source") == "topic":
        topic_id = (state.get("current_topic") or {}).get("id")
        notes.append(f"the classifier would pick {result['co_free']}, but {topic_id} is mapped to "
                     f"{', '.join(result['topic_cos'])}, so {result['co']} is used: check that the "
                     f"question kept to its topic")
    if result.get("co_level") and result["bloom_read"] != result["co_level"]:
        notes.append(f"the paper would print {result['co_level']} ({result['co']}'s level), but the "
                     f"question itself reads as {result['bloom_read']}")
    return notes


def make_classify_node(llm, subjects=None, pyqs=None):
    """`llm`: a pipeline.llm.LLMClient. `subjects`: from load_subjects();
    `pyqs`: the labelled papers for the examples (load_pyqs()). Both are read
    from disk when not given."""
    subjects = load_subjects() if subjects is None else subjects
    pyqs = load_pyqs() if pyqs is None else pyqs

    def classify_node(state):
        draft = state.get("draft")
        if not draft:
            return {"classification": None}
        subject = subject_of_topic(subjects, state["current_topic"]["id"])
        question = draft_as_text(draft)
        try:
            result = classify(llm, question, draft.get("marks"), subject,
                              pick_examples(question, subject["id"], pyqs))
        except LLMFatalError:
            raise
        except LLMError as exc:          # no labels for this question; it is still a good question
            return {"classification": {"error": str(exc)}}
        topic_cos = topic_cos_of(state, subject)
        co = choose_topic_co(result, topic_cos, subject)
        result = dict(with_course_details(dict(result, co=co), subject), co_free=result["co"],
                      co_source="model" if co == result["co"] else "topic", topic_cos=topic_cos,
                      difficulty=state.get("target_difficulty"))      # as the teacher asked
        result["notes"] = notes_for(result, state)
        return {"classification": result}

    return classify_node
