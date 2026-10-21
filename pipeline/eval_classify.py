"""
pipeline/eval_classify.py -- how good is the classifier? It labels the 53
questions of the real papers (the PYQs) and is scored against the CO and
Bloom labels printed on those papers.

    python3 -m pipeline.eval_classify                 # the model: LLM_CHECK_MODEL, else LLM_MODEL
    python3 -m pipeline.eval_classify --baseline      # keyword rules: no model, no key, instant
    python3 -m pipeline.eval_classify --subject DS --limit 5
    python3 -m pipeline.eval_classify --fresh         # forget the saved answers, ask again

No Docker: the questions come from Extracting/Processed/pyq_structured/ and
the COs from prerequisite_graph/subjects/.

Each answer is saved the moment it arrives, in
eval_results/classify_<model>.jsonl. A run that stops (a free model's daily
limit, Ctrl+C) goes on from where it stopped when started again, and the
scores are always printed from everything saved so far. Answers from an older
version of the prompt (classify.PROMPT_VERSION) are not reused.

WHAT IS MEASURED
  CO              right / total, and Cohen's kappa: agreement beyond what
                  guessing with the same label frequencies would give
                  (0 = chance, 1 = perfect)
  Bloom label     the label classify prints: the level the college gives the
                  chosen CO (the papers print a question's CO level as its
                  Bloom level), so it is right when the CO is, mostly
  Bloom read      the model's own reading of the question: right / total,
                  within one level (Apply for Analyze), kappa
The classifier sees, as examples, one real question per CO of the subject --
never the question it is labelling (pick_examples leaves it out). The
question's topic is NOT given: the PYQ topics were partly chosen from their CO
(resolve_pyq_labels.py), so they would give the answer away.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from pipeline.curriculum import BLOOM_LEVELS, ROOT, bloom_rank, load_subjects
from pipeline.llm import LLMError, LLMFatalError
from pipeline.nodes.classify import PROMPT_VERSION, PYQ_DIR, classify, keyword_classify, load_pyqs, pick_examples

RESULTS_DIR = ROOT / "eval_results"
SHORT = {"Remember": "Rem", "Understand": "Und", "Apply": "App", "Analyze": "Ana",
         "Evaluate": "Eva", "Create": "Cre"}


# ---------------------------------------------------------------- the saved answers
def results_path(name, folder=RESULTS_DIR):
    return Path(folder) / f"classify_{re.sub(r'[^A-Za-z0-9.]+', '-', name).strip('-')}.jsonl"


def load_saved(path):
    """{question id: row} for the usable rows (the last one per question)."""
    saved = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("prompt_version") == PROMPT_VERSION and "error" not in row:
                saved[row["id"]] = row
    return saved


def label_all(questions, label, path, saved, log=print):
    """label(question) -> classification for every question not saved yet;
    each row is appended to `path` as soon as it is made. Returns
    (rows for all questions answered so far, why it stopped early or None)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = dict(saved)
    todo = [q for q in questions if q["id"] not in saved]
    stopped = None
    with open(path, "a", encoding="utf-8") as out:
        for n, q in enumerate(todo, start=1):
            base = {"id": q["id"], "subject": q["subject"], "marks": q["marks"], "label_co": q["label_co"],
                    "label_bloom": q["label_bloom"], "text": q["text"], "prompt_version": PROMPT_VERSION}
            try:
                row = dict(base, **label(q))
            except LLMFatalError as exc:
                stopped = str(exc)
                break
            except LLMError as exc:
                row = dict(base, error=str(exc))
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if "error" in row:
                log(f"  [{n}/{len(todo)}] {q['id']:24} no usable answer: {row['error'][:80]}")
            else:
                rows[q["id"]] = row
                log(f"  [{n}/{len(todo)}] {q['id']:24} {row['co']:4} {row['bloom']:10} "
                    f"read as {row['bloom_read']:10}"
                    f"   (paper: {q['label_co']} {q['label_bloom']})")
    wanted = {q["id"] for q in questions}
    return [rows[i] for i in sorted(rows) if i in wanted], stopped


# ---------------------------------------------------------------- the scores
def kappa(pairs):
    """Cohen's kappa for [(label, predicted)]; None when it is undefined
    (every label and prediction the same)."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    labels, predicted = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(labels[k] * predicted[k] for k in labels) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def score(rows, subjects):
    """The numbers the report prints, from the saved rows."""
    rows = [r for r in rows if "error" not in r]
    levels = {s: subjects[s]["co_levels"] for s in subjects}
    labelled_by_co = [r for r in rows if levels.get(r["subject"], {}).get(r["label_co"])]
    read_pairs = [(r["label_bloom"], r["bloom_read"]) for r in rows]
    out = {
        "n": len(rows),
        "co_right": sum(r["co"] == r["label_co"] for r in rows),
        "co_kappa": kappa([(f"{r['subject']}:{r['label_co']}", f"{r['subject']}:{r['co']}") for r in rows]),
        "bloom_right": sum(r["bloom"] == r["label_bloom"] for r in rows),
        "bloom_kappa": kappa([(r["label_bloom"], r["bloom"]) for r in rows]),
        "read_right": sum(a == b for a, b in read_pairs),
        "read_near": sum(abs(bloom_rank(a) - bloom_rank(b)) <= 1 for a, b in read_pairs),
        "read_kappa": kappa(read_pairs),
        # a property of the papers, not of the classifier
        "labels_are_co_level": sum(levels[r["subject"]][r["label_co"]] == r["label_bloom"] for r in labelled_by_co),
        "labels_with_co_level": len(labelled_by_co),
        "by_subject": {},
        "confusion": Counter(read_pairs),
        "wrong_co": [r for r in rows if r["co"] != r["label_co"]],
    }
    for s in sorted({r["subject"] for r in rows}):
        mine = [r for r in rows if r["subject"] == s]
        out["by_subject"][s] = {"n": len(mine), "co_right": sum(r["co"] == r["label_co"] for r in mine),
                                "bloom_right": sum(r["bloom"] == r["label_bloom"] for r in mine)}
    return out


def _pct(right, n):
    return f"{right}/{n}  {round(100 * right / n) if n else 0:3d}%"


def _k(value):
    return "n/a" if value is None else f"{value:.2f}"


def print_report(name, s, total, stopped=None):
    print(f"\nClassifier: {name}   ({s['n']} of {total} questions labelled)")
    if stopped:
        print(f"Stopped early: {stopped}\nRun the same command again later: it goes on from question {s['n'] + 1}.")
    if not s["n"]:
        return
    print(f"\n  {'':18}{'right':>14}   kappa")
    print(f"  {'CO':18}{_pct(s['co_right'], s['n']):>14}   {_k(s['co_kappa'])}")
    print(f"  {'Bloom label':18}{_pct(s['bloom_right'], s['n']):>14}   {_k(s['bloom_kappa'])}"
          f"   (the chosen CO's level: what classify prints)")
    print(f"  {'Bloom read':18}{_pct(s['read_right'], s['n']):>14}   {_k(s['read_kappa'])}"
          f"   (the model's own reading of the question)")
    print(f"  {'Bloom read +-1':18}{_pct(s['read_near'], s['n']):>14}")
    if s["labels_with_co_level"]:
        print(f"\n  On these papers, {s['labels_are_co_level']} of {s['labels_with_co_level']} Bloom labels "
              f"are exactly their CO's level.")

    print(f"\n  {'by subject':14}{'CO':>12}{'Bloom':>14}")
    for subject, b in s["by_subject"].items():
        print(f"  {subject + ' (' + str(b['n']) + ')':14}{_pct(b['co_right'], b['n']):>12}"
              f"{_pct(b['bloom_right'], b['n']):>14}")

    print("\n  Bloom read: rows = the paper's label, columns = the model's reading")
    print("  " + " " * 12 + "".join(f"{SHORT[level]:>5}" for level in BLOOM_LEVELS))
    for label in BLOOM_LEVELS:
        counts = [s["confusion"].get((label, got), 0) for got in BLOOM_LEVELS]
        if any(counts):
            print(f"  {label:12}" + "".join(f"{c or '.':>5}" for c in counts))

    if s["wrong_co"]:
        print(f"\n  Wrong CO ({len(s['wrong_co'])}):")
        for r in s["wrong_co"]:
            text = " ".join(r.get("text", "").split())
            print(f"    {r['id']:24} paper {r['label_co']}, got {r['co']}   \"{text[:70]}\"")


# ---------------------------------------------------------------- run it
def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the classifier against the labelled PYQs")
    ap.add_argument("--baseline", action="store_true", help="keyword rules instead of a model")
    ap.add_argument("--model", help="the model to score (default: LLM_CHECK_MODEL, else LLM_MODEL)")
    ap.add_argument("--subject", help="only this subject (SDF1, SDF2, DS, ALGO)")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved answers and ask again")
    args = ap.parse_args(argv)

    subjects = load_subjects()
    questions = load_pyqs()
    if args.subject:
        questions = [q for q in questions if q["subject"].lower() == args.subject.lower()]
    questions = questions[:args.limit] if args.limit else questions
    if not questions:
        raise SystemExit("No labelled questions found" + (f" for {args.subject}" if args.subject else "")
                         + f" in {PYQ_DIR}")

    if args.baseline:
        args.fresh = True                   # instant, and the rules may have changed: never reuse
        name = "keywords"
        def label(q):  # noqa: E306
            return keyword_classify(q["text"], subjects[q["subject"]])
    else:
        from pipeline.llm import make_checker, make_llm
        llm = make_llm(model=args.model) if args.model else (make_checker() or make_llm())
        name = llm.model
        pyqs = load_pyqs()

        def label(q):  # noqa: E306
            return classify(llm, q["text"], q["marks"], subjects[q["subject"]],
                            pick_examples(q["text"], q["subject"], pyqs, exclude_id=q["id"]))

    path = results_path(name, RESULTS_DIR)
    saved = {} if args.fresh else load_saved(path)
    if args.fresh and path.exists():
        path.unlink()
    print(f"{len(questions)} labelled questions; {sum(q['id'] in saved for q in questions)} already "
          f"answered in {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    try:
        rows, stopped = label_all(questions, label, path, saved)
    except KeyboardInterrupt:
        rows, stopped = list(load_saved(path).values()), "stopped by Ctrl+C"
        rows = [r for r in rows if r["id"] in {q["id"] for q in questions}]
    print_report(name, score(rows, subjects), len(questions), stopped)
    if not args.baseline:
        from pipeline.llm import usage_line
        print(f"\n({usage_line(llm)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
