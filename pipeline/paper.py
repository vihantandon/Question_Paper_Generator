"""
pipeline/paper.py -- a whole question paper from a teacher's blueprint (Phase 2).

    python3 -m pipeline.paper new                      # 1. the form -> papers/ds_t1.yaml
    python3 -m pipeline.paper check papers/ds_t1.yaml  # 2. the plan, no AI, no tokens
    python3 -m pipeline.paper make papers/ds_t1.yaml   # 3. the paper

The blueprint format and the checks are in pipeline/blueprint.py.

STEP 3 -- making it, question by question
  * a question already in the question bank that fits the slot (same topic,
    marks and difficulty, with a CO label) and that no paper has used yet is
    reused: no tokens. (Questions made while testing single topics land there.)
  * otherwise the pipeline writes one: draft -> quality check -> duplicate
    check -> classify, at most 3 drafts (pipeline/graph.py). The teacher does
    not give the CO: classify decides it (one of the COs the topic is mapped
    to), the Bloom level is that CO's level, the difficulty the one asked for.
    If classify gives no usable answer, the topic's first CO is used, with a
    note to check it.
  * when the paper is done, the marks per CO are shown, and every CO of the
    subject that no question tests.
  * every question the paper takes is saved in the bank as used by this paper,
    so later papers do not repeat it, and later questions in this paper are
    checked against it.
  * after each question the progress is written to papers/output/<name>.json.
    A run stopped by a free model's daily limit (or Ctrl+C) goes on from the
    next question when the same command runs again. A question that failed (3
    rejected drafts, no study material) is tried again then too.
  * when every question is done: papers/output/<name>.md (the paper) and
    <name>_key.md (the answer key, a draft to check before use).
  --fresh starts the paper again from nothing (the bank keeps its questions).
"""
import argparse
import json
import sys
from pathlib import Path

from pipeline.blueprint import (PAPERS_DIR, blueprint_path, load_blueprint, new_blueprint, print_preview,
                                save_blueprint, validate)
from pipeline.curriculum import load_subjects
from pipeline.dedup import find_unused, mark_used, save_question
from pipeline.llm import LLMFatalError

OUTPUT_DIR = PAPERS_DIR / "output"


# ---------------------------------------------------------------- progress
def output_paths(blueprint_file, out_dir=OUTPUT_DIR):
    stem = Path(blueprint_file).stem
    return {"progress": Path(out_dir) / f"{stem}.json", "paper": Path(out_dir) / f"{stem}.md",
            "key": Path(out_dir) / f"{stem}_key.md"}


def slot_spec(row):
    return {k: row[k] for k in ("topic", "marks", "difficulty")}


def load_progress(path, rows):
    """Questions already done for this blueprint: {question number: slot}. A
    slot whose spec changed in the blueprint since is made again."""
    if not Path(path).exists():
        return {}
    done = {}
    for slot in json.loads(Path(path).read_text(encoding="utf-8")).get("slots", []):
        n = slot.get("n")
        if (slot.get("status") == "done" and isinstance(n, int) and n <= len(rows)
                and slot.get("spec") == slot_spec(rows[n - 1])):
            done[n] = slot
    return done


def save_progress(path, paper_id, blueprint, slots):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"paper_id": paper_id, "blueprint": blueprint,
                                      "slots": [slots[n] for n in sorted(slots)]},
                                     indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- one question
def reused_slot(row, bank_id, meta):
    return {"n": row["n"], "spec": slot_spec(row), "status": "done", "source": "bank", "bank_id": bank_id,
            "question": meta.get("stem") or "", "parts": json.loads(meta.get("parts") or "[]"),
            "answer_outline": json.loads(meta.get("answer_key") or "[]"), "marks": row["marks"],
            "co": meta.get("co"), "bloom": meta.get("bloom") or "", "difficulty": row["difficulty"],
            "topic": row["topic"], "topic_name": row["topic_name"], "notes": []}


def generate(graph, row):
    """Run the question pipeline for one slot. Returns the final state."""
    return graph.invoke({"current_topic_id": row["topic"], "target_difficulty": row["difficulty"],
                         "target_marks": row["marks"]})


def labels(result, row, subject):
    """(co, bloom, notes) from classify -- or, when it gave no usable answer,
    the topic's first CO and its level, with a note to check it."""
    c = result.get("classification") or {}
    if c.get("co") and "error" not in c:
        return c["co"], c.get("bloom") or "", list(c.get("notes") or [])
    co = (row["cos"] or list(subject["cos"]))[0]
    why = c.get("error") or "classify did not run"
    return co, subject.get("co_levels", {}).get(co, ""), [
        f"no CO from classify ({why[:100]}): {co}, the topic's first CO, is used -- check it"]


def why_not(result):
    """One line: why the pipeline gave no question for this slot."""
    if result.get("grounding_warning") and not result.get("draft"):
        return result["grounding_warning"]
    if result.get("draft_error") and not result.get("draft"):
        return f"no draft: {result['draft_error']}"
    dedup = result.get("dedup") or {}
    if dedup and not dedup.get("passed"):
        return "every draft was too close to an existing question"
    problems = (result.get("quality") or {}).get("problems") or []
    if "error" in (result.get("quality") or {}):
        return f"the checker failed: {result['quality']['error']}"
    return "every draft was rejected" + (f": {problems[0]}" if problems else "")


def make_paper(blueprint, rows, res, graph, paper_id, progress_path, *, subjects=None, log=print,
               now=None, fresh=False):
    """Fill every question of the paper. Returns (slots, stopped): slots
    {n: slot}, stopped = why the run stopped early, or None."""
    from pipeline.graph import accepted
    subjects = load_subjects() if subjects is None else subjects
    subject = subjects[blueprint["subject"].upper()]
    slots = {} if fresh else load_progress(progress_path, rows)
    taken = {s.get("bank_id") for s in slots.values()}
    for row in rows:
        n = row["n"]
        if n in slots:
            log(f"Q{n}: done before ({slots[n]['source']})")
            continue
        found = find_unused(res.bank_collection, row["topic"], row["marks"], row["difficulty"], taken=taken)
        if found:
            bank_id, meta = found
            mark_used(res.bank_collection, bank_id, meta, paper_id)
            slots[n] = reused_slot(row, bank_id, meta)
            taken.add(bank_id)
            log(f"Q{n}: reused {bank_id} from the question bank (no tokens)")
            save_progress(progress_path, paper_id, blueprint, slots)
            continue

        log(f"Q{n}: writing a {row['marks']}-mark {row['difficulty']} question on {row['topic']} "
            f"{row['topic_name']} ...")
        try:
            result = generate(graph, row)
        except LLMFatalError as exc:
            save_progress(progress_path, paper_id, blueprint, slots)
            return slots, str(exc)
        if not accepted(result):
            slots[n] = {"n": n, "spec": slot_spec(row), "status": "failed", "reason": why_not(result),
                        "drafts": result.get("draft_attempts", 0)}
            log(f"Q{n}: FAILED -- {slots[n]['reason']}")
            save_progress(progress_path, paper_id, blueprint, slots)
            continue
        co, bloom, notes = labels(result, row, subject)
        if result["draft"].get("blend_dropped"):
            notes.append(result["draft"]["blend_dropped"])
        result["classification"] = dict(result.get("classification") or {}, co=co, bloom=bloom)
        bank_id = None
        if res.bank_collection is not None:
            bank_id = save_question(res.bank_collection, res.embed_fn, result, now=now, used_in=paper_id)
            taken.add(bank_id)
        d = result["draft"]
        slots[n] = {"n": n, "spec": slot_spec(row), "status": "done", "source": "generated", "bank_id": bank_id,
                    "question": d["question"], "parts": d["parts"], "answer_outline": d["answer_outline"],
                    "marks": row["marks"], "co": co, "bloom": bloom, "difficulty": row["difficulty"],
                    "co_free": (result.get("classification") or {}).get("co_free"), "notes": notes,
                    "topic": row["topic"], "topic_name": row["topic_name"],
                    "blend": (result.get("blend_topic") or {}).get("name"), "drafts": result.get("draft_attempts"),
                    "model_knowledge": [c["claim"] for c in (result.get("quality") or {}).get("claims", [])
                                        if c.get("status") == "model knowledge"]}
        log(f"Q{n}: done after {result.get('draft_attempts')} draft(s) -- {co}, {bloom}")
        save_progress(progress_path, paper_id, blueprint, slots)
    return slots, None


def co_coverage(slots, subject):
    """({CO: marks}, [COs of the subject no question tests])."""
    marks = {}
    for s in slots.values():
        if s.get("status") == "done" and s.get("co"):
            marks[s["co"]] = marks.get(s["co"], 0) + s["marks"]
    return marks, [co for co in subject["cos"] if co not in marks]


# ---------------------------------------------------------------- the paper as text
def render_paper(blueprint, slots, subjects=None):
    subjects = load_subjects() if subjects is None else subjects
    subject = subjects[blueprint["subject"].upper()]
    exam = str(blueprint["exam"]).upper()
    lines = [f"# {subject['name']} -- {exam} Examination", "",
             f"**Maximum Marks:** {blueprint['total_marks']}    **Maximum Time:** {blueprint.get('time', '')}", "",
             "| CO | Course outcome |", "|---|---|"]
    used = sorted({s["co"] for s in slots.values()}, key=lambda c: list(subject["cos"]).index(c)
                  if c in subject["cos"] else 99)
    lines += [f"| {co} | {subject['cos'].get(co, '')} |" for co in used]
    lines += ["", "**Note:** Attempt all questions.", ""]
    for n in sorted(slots):
        s = slots[n]
        lines += [f"**Q{n}.** {s['question'].strip()}  **[{s['marks']} marks] [{s['co']}, {s['bloom']}]**", ""]
        lines += [f"({p['label']}) {p['text']}  [{p['marks']}]" for p in s["parts"]]
        lines += [""]
    return "\n".join(lines)


def render_key(blueprint, slots, subjects=None):
    subjects = load_subjects() if subjects is None else subjects
    subject = subjects[blueprint["subject"].upper()]
    lines = [f"# Answer key (draft) -- {subject['name']} {str(blueprint['exam']).upper()}", "",
             "Written by the model and checked automatically: check it before use.", ""]
    marks, untested = co_coverage(slots, subject)
    lines += ["Marks per CO: " + ", ".join(f"{co} {m}" for co, m in sorted(marks.items()))
              + (f"; not tested: {', '.join(untested)}" if untested else ""), ""]
    for n in sorted(slots):
        s = slots[n]
        lines += [f"## Q{n} ({s['marks']} marks, {s['topic']} {s['topic_name']}, {s['co']} {s['bloom']}, "
                  f"{s['difficulty']}; "
                  f"{'reused from the question bank' if s['source'] == 'bank' else 'generated'})", ""]
        lines += [f"- {step}" for step in s["answer_outline"]]
        for note in s.get("notes") or []:
            lines += [f"- note: {note}"]
        if s.get("model_knowledge"):
            lines += ["", "Facts from the model's own knowledge (check these):"]
            lines += [f"- {fact}" for fact in s["model_knowledge"]]
        lines += [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- the command line
def check_command(path, show_reuse=True):
    blueprint = load_blueprint(path)
    errors, warnings, rows = validate(blueprint)
    reusable = None
    if show_reuse and not errors:
        bank = _bank_if_running()
        if bank is not None:
            reusable, taken = {}, set()
            for r in rows:
                found = find_unused(bank, r["topic"], r["marks"], r["difficulty"], taken=taken)
                if found:
                    reusable[r["n"]] = found[0]
                    taken.add(found[0])
        else:
            print("(Chroma is not running: reuse from the question bank not checked)")
    print_preview(blueprint, errors, warnings, rows, reusable)
    return 1 if errors else 0


def _bank_if_running():
    try:
        import chromadb

        from pipeline.dedup import get_bank
        from settings import CHROMA_HOST, CHROMA_PORT
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        return get_bank(client)
    except Exception:        # noqa: BLE001 -- the preview works without Chroma
        return None


def make_command(path, fresh=False, max_drafts=None):
    from pipeline.graph import MAX_DRAFTS, build_graph, grounding_from_settings
    from pipeline.llm import make_checker, make_llm, usage_line
    from pipeline.resources import connect
    blueprint = load_blueprint(path)
    errors, warnings, rows = validate(blueprint)
    if errors:
        print_preview(blueprint, errors, warnings, rows)
        return 1
    for w in warnings:
        print(f"warning: {w}")
    paths = output_paths(path)
    paper_id = f"{Path(path).stem}"
    llm = make_llm()
    checker = make_checker()
    res = connect(llm=llm, checker=checker)
    try:
        graph = build_graph(res, classify=True, dedup=True, grounding=grounding_from_settings(),
                            max_drafts=max_drafts or MAX_DRAFTS)
        slots, stopped = make_paper(blueprint, rows, res, graph, paper_id, paths["progress"], fresh=fresh)
    finally:
        res.close()
    done = {n: s for n, s in slots.items() if s["status"] == "done"}
    failed = [s for s in slots.values() if s["status"] == "failed"]
    print(f"\n{len(done)} of {len(rows)} question(s) done; progress in {paths['progress']}")
    subject = load_subjects()[blueprint["subject"].upper()]
    marks, untested = co_coverage(done, subject)
    if done:
        print("Marks per CO: " + ", ".join(f"{co} {m}" for co, m in sorted(marks.items())))
    if done and untested:
        print(f"warning: no question tests {', '.join(untested)}")
    for s in done.values():
        for note in s.get("notes") or []:
            print(f"Q{s['n']} note: {note}")
    if stopped:
        print(f"Stopped: {stopped}\nRun the same command again later: it goes on from the next question.")
    for s in failed:
        print(f"Q{s['n']} failed: {s['reason']}  -- run again to retry, or change it in the blueprint")
    if len(done) == len(rows):
        paths["paper"].write_text(render_paper(blueprint, done), encoding="utf-8")
        paths["key"].write_text(render_key(blueprint, done), encoding="utf-8")
        print(f"The paper: {paths['paper']}\nThe answer key (check before use): {paths['key']}")
    print(f"({usage_line(llm, checker)})")
    return 0 if len(done) == len(rows) else 2


def new_command(out=None):
    blueprint = new_blueprint()
    errors, warnings, rows = validate(blueprint)
    print_preview(blueprint, errors, warnings, rows)
    path = Path(out) if out else blueprint_path(blueprint)
    save_blueprint(blueprint, path)
    print(f"\nSaved {path}. Change it in any text editor, then:\n"
          f"  python3 -m pipeline.paper check {path}\n  python3 -m pipeline.paper make {path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make a question paper from a blueprint")
    sub = ap.add_subparsers(dest="command", required=True)
    p_new = sub.add_parser("new", help="write a blueprint, question by question")
    p_new.add_argument("--out", help="where to save it (default papers/<subject>_<exam>.yaml)")
    p_check = sub.add_parser("check", help="the plan and its problems: no AI, no tokens")
    p_check.add_argument("blueprint")
    p_make = sub.add_parser("make", help="make the paper (goes on where a stopped run left off)")
    p_make.add_argument("blueprint")
    p_make.add_argument("--fresh", action="store_true", help="start the paper again from nothing")
    p_make.add_argument("--max-drafts", type=int, help="drafts per question before it fails (default 3)")
    args = ap.parse_args(argv)
    if args.command == "new":
        return new_command(args.out)
    if args.command == "check":
        return check_command(args.blueprint)
    return make_command(args.blueprint, fresh=args.fresh, max_drafts=args.max_drafts)


if __name__ == "__main__":
    sys.exit(main())
