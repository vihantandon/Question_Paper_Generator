"""
pipeline/blueprint.py -- what the teacher asks for (Phase 2, steps 1 and 2).

A blueprint is a small YAML file, one per paper:

    subject: DS
    exam: T1                 # T1 | T2 | T3 (T3 = end semester)
    total_marks: 20
    time: 1 hour
    questions:
      - topic: DS_03         # a topic id from prerequisite_graph/subjects/ds.yaml
        marks: 5
        difficulty: medium   # easy | medium | hard
      - ...

The teacher does NOT give the CO: after a question is written, classify
(pipeline/nodes/classify.py) decides it -- one of the COs its topic is mapped
to -- and the Bloom level is that CO's level, as the college's papers print it.

STEP 1 -- writing one
    python3 -m pipeline.paper new         asks question by question and writes
                                          papers/<subject>_<exam>.yaml
    The starting values come from the college's own papers (the PYQ headers):
    T1 / T2 = 20 marks, 1 hour; T3 = 35 marks, 2 hours, and the usual number
    of questions. Each topic says if it was asked in past papers of that kind.

STEP 2 -- checking one, before any token is spent
    python3 -m pipeline.paper check papers/ds_t1.yaml
    errors   (the paper cannot be made): an unknown topic, marks that do not
             add up to total_marks, a difficulty that is not easy / medium /
             hard, no questions
    warnings (made anyway, but look): a CO of the subject that none of the
             chosen topics is mapped to (no question can test it), a "co:"
             line (ignored: classify decides)
    For each question the preview shows the COs its topic can test; the
    actual CO and Bloom level are known once the question is written.
"""
import json
import re
from collections import Counter
from pathlib import Path

import yaml

from pipeline.curriculum import ROOT, load_subjects

EXAMS = {"T1": "test1", "T2": "test2", "T3": "endsem"}      # blueprint name -> PYQ "exam" field
DIFFICULTIES = ("easy", "medium", "hard")
PAPERS_DIR = ROOT / "papers"
PYQ_DIR = ROOT / "Extracting" / "Processed" / "pyq_structured"
FALLBACK = {"T1": {"total_marks": 20, "time": "1 hour", "questions": 5},
            "T2": {"total_marks": 20, "time": "1 hour", "questions": 5},
            "T3": {"total_marks": 35, "time": "2 hours", "questions": 7}}

MARKS_RE = re.compile(r"Maximum\s*Marks\s*[:\-]?\s*\**\s*(\d+)", re.IGNORECASE)
TIME_RE = re.compile(r"Time\s*[:\-]?\s*\**\s*0?(\d+(?:\.\d+)?)\s*(?:hr|hour)", re.IGNORECASE)


# ---------------------------------------------------------------- what the past papers say
def _papers(folder=PYQ_DIR):
    """[{"subject", "exam", "total_marks", "hours", "marks": [...], "topics": {...}}] per PYQ paper."""
    out = []
    for path in sorted(Path(folder).glob("*_pyq.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        header = next((r for r in records if (r.get("metadata") or {}).get("type") == "header"), None)
        questions = [r["metadata"] for r in records if (r.get("metadata") or {}).get("type") == "question"]
        if not header or not questions:
            continue
        meta, text = header["metadata"], header.get("document") or ""
        total, hours = MARKS_RE.search(text), TIME_RE.search(text)
        out.append({"subject": meta.get("subject"), "exam": meta.get("exam"),
                    "total_marks": int(total.group(1)) if total else sum(q.get("marks") or 0 for q in questions),
                    "hours": float(hours.group(1)) if hours else None,
                    "marks": [q.get("marks") for q in questions if isinstance(q.get("marks"), int)],
                    "topics": {q.get("topic_id") for q in questions if q.get("topic_id")}})
    return out


def _hours_text(hours):
    if hours is None:
        return None
    h = int(hours) if float(hours).is_integer() else hours
    return f"{h} hour" if h == 1 else f"{h} hours"


def exam_defaults(subject, exam, folder=PYQ_DIR):
    """Starting values for a new blueprint, from the college's past papers of
    this kind (this subject's when it has one, else every subject's):
    {"total_marks", "time", "questions", "marks": [a typical paper's marks]}."""
    papers = [p for p in _papers(folder) if p["exam"] == EXAMS[exam]]
    mine = [p for p in papers if p["subject"] == subject] or papers
    if not mine:
        return dict(FALLBACK[exam], marks=[])
    total = Counter(p["total_marks"] for p in mine).most_common(1)[0][0]
    hours = Counter(p["hours"] for p in mine if p["hours"]).most_common(1)
    counts = sorted(len(p["marks"]) for p in mine)
    typical = next((p["marks"] for p in mine if p["total_marks"] == total and sum(p["marks"]) == total), [])
    return {"total_marks": total, "time": _hours_text(hours[0][0]) if hours else FALLBACK[exam]["time"],
            "questions": counts[len(counts) // 2], "marks": typical}


def topics_in_past_papers(subject, exam, folder=PYQ_DIR):
    """Topic ids the college's past papers of this kind asked about."""
    return set().union(*[p["topics"] for p in _papers(folder)
                         if p["subject"] == subject and p["exam"] == EXAMS[exam]] or [set()])


# ---------------------------------------------------------------- the file
def load_blueprint(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a blueprint is a YAML mapping (subject:, exam:, questions: ...)")
    return data


def save_blueprint(blueprint, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    head = (f"# Question paper blueprint -- check it:  python3 -m pipeline.paper check {path}\n"
            f"# then make the paper:              python3 -m pipeline.paper make {path}\n")
    Path(path).write_text(head + yaml.safe_dump(blueprint, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------- step 2: checking it
def validate(blueprint, subjects=None):
    """(errors, warnings, rows). rows: one per question, with the topic name,
    the CO's wording and the Bloom level (the CO's level). Errors mean the
    paper cannot be made."""
    subjects = load_subjects() if subjects is None else subjects
    errors, warnings, rows = [], [], []
    subject = subjects.get(str(blueprint.get("subject", "")).upper())
    if subject is None:
        return [f"subject must be one of {', '.join(subjects)} (got {blueprint.get('subject')!r})"], [], []
    exam = str(blueprint.get("exam", "")).upper()
    if exam not in EXAMS:
        errors.append(f"exam must be one of {', '.join(EXAMS)} (got {blueprint.get('exam')!r})")
    total = blueprint.get("total_marks")
    if not isinstance(total, int) or total <= 0:
        errors.append(f"total_marks must be a whole number above 0 (got {total!r})")
    questions = blueprint.get("questions") or []
    if not isinstance(questions, list) or not questions:
        errors.append("questions: the paper has none")
        questions = []

    for n, q in enumerate(questions, start=1):
        q = q if isinstance(q, dict) else {}
        topic_id = str(q.get("topic", ""))
        marks, difficulty = q.get("marks"), str(q.get("difficulty", "")).lower()
        topic = subject["topics"].get(topic_id)
        if topic is None:
            errors.append(f"Q{n}: topic {topic_id!r} is not a {subject['id']} topic "
                          f"({', '.join(subject['topics'])})")
        if not isinstance(marks, int) or marks <= 0:
            errors.append(f"Q{n}: marks must be a whole number above 0 (got {marks!r})")
        if difficulty not in DIFFICULTIES:
            errors.append(f"Q{n}: difficulty must be easy, medium or hard (got {q.get('difficulty')!r})")
        if "co" in q:
            warnings.append(f"Q{n}: the co line is ignored -- classify decides the CO once the question is written")
        cos = list(topic["cos"]) if topic else []
        rows.append({"n": n, "topic": topic_id, "topic_name": topic["name"] if topic else "?",
                     "marks": marks, "difficulty": difficulty, "cos": cos})

    given = sum(r["marks"] for r in rows if isinstance(r["marks"], int))
    if isinstance(total, int) and rows and given != total:
        errors.append(f"the questions add up to {given} marks, not total_marks {total}")
    reachable = {c for r in rows for c in r["cos"]}
    untested = [co for co in subject["cos"] if co not in reachable]
    if rows and untested:
        warnings.append(f"none of the chosen topics is mapped to {', '.join(untested)}: no question can test it")
    return errors, warnings, rows


def print_preview(blueprint, errors, warnings, rows, reusable=None, subjects=None):
    """The plan, as the teacher checks it. reusable: {question number: bank id}."""
    subjects = load_subjects() if subjects is None else subjects
    subject = subjects.get(str(blueprint.get("subject", "")).upper())
    name = subject["name"] if subject else blueprint.get("subject")
    print(f"\n{name} -- {str(blueprint.get('exam', '?')).upper()}   "
          f"{blueprint.get('total_marks')} marks, {blueprint.get('time', '?')}, {len(rows)} question(s)\n")
    print(f"  {'Q':3} {'topic':8} {'marks':>5}  {'difficulty':10} {'CO it can test':15} topic name")
    for r in rows:
        reuse = f"   <- reuses {reusable[r['n']]}" if reusable and r["n"] in reusable else ""
        print(f"  Q{r['n']:<2} {r['topic']:8} {str(r['marks']):>5}  {r['difficulty']:10} "
              f"{' or '.join(r['cos']) or '?':15} {r['topic_name'][:45]}{reuse}")
    if rows:
        print("\n  Difficulty: " + ", ".join(f"{d} {c}" for d, c in Counter(r["difficulty"] for r in rows).items())
              + "   (the CO and Bloom level of each question: decided by classify once it is written)")
    if reusable is not None:
        print(f"  {len(reusable)} question(s) can reuse an unused question from the bank (no tokens); "
              f"{len(rows) - len(reusable)} to generate")
    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    print("\n" + ("Fix the errors, then check again." if errors else "The plan is fine: make the paper with "
                  "python3 -m pipeline.paper make <blueprint>"))


# ---------------------------------------------------------------- step 1: the form
def _ask(prompt, default, read, show):
    answer = read(f"{prompt}" + (f" [{default}]" if default not in (None, "") else "") + ": ").strip()
    return answer or ("" if default is None else str(default))


def _choose(prompt, options, default, read, show):
    """options: [(value, label)]. The answer is a number from the list or the value itself."""
    for i, (value, label) in enumerate(options, start=1):
        show(f"  {i:2}. {label}")
    values = [v for v, _ in options]
    while True:
        answer = _ask(prompt, default, read, show)
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return values[int(answer) - 1]
        match = [v for v in values if v.lower() == answer.lower()]
        if match:
            return match[0]
        show(f"  please choose 1-{len(options)}")


def _number(prompt, default, read, show):
    while True:
        answer = _ask(prompt, default, read, show)
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        show("  please give a whole number above 0")


def new_blueprint(read=input, show=print, subjects=None, pyq_folder=PYQ_DIR):
    """Ask the teacher for the paper, question by question. Returns the blueprint."""
    subjects = load_subjects() if subjects is None else subjects
    show("\nA new question paper. Press Enter to take the value in [brackets].\n")
    sid = _choose("Subject", [(s["id"], f"{s['id']:5} {s['name']} (semester {s['semester']})")
                              for s in subjects.values()], None, read, show)
    subject = subjects[sid]
    exam = _choose("Exam", [("T1", "T1"), ("T2", "T2"), ("T3", "T3 (end semester)")], "T1", read, show)
    d = exam_defaults(sid, exam, pyq_folder)
    show(f"  past {exam} papers: {d['total_marks']} marks, {d['time']}, about {d['questions']} questions"
         + (f" ({', '.join(map(str, d['marks']))} marks)" if d["marks"] else ""))
    count = _number("Number of questions", d["questions"], read, show)
    total = _number("Total marks", d["total_marks"], read, show)
    time = _ask("Time", d["time"], read, show)
    seen = topics_in_past_papers(sid, exam, pyq_folder)

    questions, left = [], total
    topic_options = [(tid, f"{tid:8} {t['name']}" + (f"   (asked in past {exam} papers)" if tid in seen else ""))
                     for tid, t in subject["topics"].items()]
    for n in range(1, count + 1):
        show(f"\nQuestion {n} of {count} ({left} marks left)")
        topic = _choose("Topic", topic_options, None, read, show)
        suggest = left if n == count else max(1, round(left / (count - n + 1)))
        marks = _number("Marks", suggest, read, show)
        difficulty = _choose("Difficulty", [(x, x) for x in DIFFICULTIES], "medium", read, show)
        questions.append({"topic": topic, "marks": marks, "difficulty": difficulty})
        left -= marks
    return {"subject": sid, "exam": exam, "total_marks": total, "time": time, "questions": questions}


def blueprint_path(blueprint, folder=PAPERS_DIR):
    """papers/ds_t1.yaml, or ds_t1_2.yaml ... when that exists."""
    stem = f"{blueprint['subject'].lower()}_{blueprint['exam'].lower()}"
    path, n = Path(folder) / f"{stem}.yaml", 2
    while path.exists():
        path, n = Path(folder) / f"{stem}_{n}.yaml", n + 1
    return path
