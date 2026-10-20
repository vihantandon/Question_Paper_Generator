"""
The pipeline graph:

    START -> graph_lookup -> retrieve -> draft -> quality_check -> END
                                           ^            |
                                           +-- fail ----+   (at most MAX_DRAFTS drafts)

draft and quality_check (the language model) are added when the Resources
carry an llm; without one (--no-llm, or the tests that only check
retrieval) the graph ends after retrieve. --no-check stops after draft.

quality_check sends a failed draft back to draft with revision_feedback --
this loop is the reason the project uses LangGraph instead of a plain chain.
The question is ACCEPTED when a draft passes, and REJECTED when MAX_DRAFTS
drafts have failed, or when nothing could be drafted or checked (a later
paper planner then picks another topic for that slot).

Later phases add steps after quality_check (classify -> dedup -> constraint check).

Try it (from the repo root; Neo4j + Chroma must be running):
    python3 -m pipeline.graph ALGO_06 hard 10            # writes one question
    python3 -m pipeline.graph DS_03 medium 5 --show-prompt
    python3 -m pipeline.graph DS_03 easy 5 --no-llm --show-prompt   # no API call
    python3 -m pipeline.graph SDF1_01 easy 2             # no grounding text: not drafted
    python3 -m pipeline.graph DS_03 medium 5 --no-check  # draft only, one request
The model is set in .env (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY, and
LLM_CHECK_MODEL for a different checking model; see settings.py).
"""
import argparse

from langgraph.graph import END, START, StateGraph

from pipeline.nodes.draft import build_messages, make_draft_node
from pipeline.nodes.graph_lookup import make_graph_lookup_node
from pipeline.nodes.quality_check import GROUNDING_MODES, make_quality_check_node
from pipeline.nodes.retrieve import make_retrieve_node
from pipeline.resources import Resources, connect
from pipeline.state import GraphState


MAX_DRAFTS = 3      # first draft + 2 rewrites with the checker's feedback


def after_draft(state):
    """Check a draft; with no draft (no grounding text, the model gave up) stop."""
    return "quality_check" if state.get("draft") else END


def make_after_check(max_drafts):
    def after_check(state):
        """Passed -> done. Failed -> draft again with the feedback, unless
        max_drafts drafts were made, or the CHECKER failed (asking the
        writer again would not help)."""
        if state.get("quality_passed"):
            return END
        if "error" in (state.get("quality") or {}):
            return END
        if state.get("draft_attempts", 0) >= max_drafts:
            return END
        return "draft"
    return after_check


def build_graph(res: Resources, check=True, max_drafts=MAX_DRAFTS, grounding="balanced"):
    builder = StateGraph(GraphState)
    builder.add_node("graph_lookup", make_graph_lookup_node(res.driver))
    builder.add_node("retrieve", make_retrieve_node(
        res.grounding_collections, res.pyq_collection, res.embed_fn))
    builder.add_edge(START, "graph_lookup")
    builder.add_edge("graph_lookup", "retrieve")
    if res.llm is None:
        builder.add_edge("retrieve", END)
        return builder.compile()

    builder.add_node("draft", make_draft_node(res.llm, grounding=grounding))
    builder.add_edge("retrieve", "draft")
    if not check:
        builder.add_edge("draft", END)
        return builder.compile()

    builder.add_node("quality_check", make_quality_check_node(res.checker or res.llm, grounding))
    builder.add_conditional_edges("draft", after_draft, ["quality_check", END])
    builder.add_conditional_edges("quality_check", make_after_check(max_drafts), ["draft", END])
    return builder.compile()


def accepted(r):
    """The question came out of the loop with a draft that passed the check."""
    return bool(r.get("draft")) and bool(r.get("quality_passed"))


def print_draft(r):
    if "draft" not in r:
        return
    d = r.get("draft")
    if not d:
        print(f"\nNo draft: {r.get('draft_error')}")
        return
    print(f"\nDraft ({d.get('model')}, attempt {r.get('draft_attempts')}), {d['marks']} marks:")
    print("  " + d["question"].replace("\n", "\n  "))
    for p in d["parts"]:
        print(f"  ({p['label']}) {p['text']}   [{p['marks']}]")
    print("Answer outline:")
    for step in d["answer_outline"]:
        print(f"  - {step}")
    print("Sources used: " + ", ".join(f"{label} ({cid})" for label, cid in zip(d["sources_used"], d["source_ids"])))
    if d.get("blend_used"):
        print(f"Blend: {d['blend_used']}")


def print_question(d, indent="  "):
    print(indent + d["question"].replace("\n", "\n" + indent))
    for p in d["parts"]:
        print(f"{indent}({p['label']}) {p['text']}   [{p['marks']}]")


def print_check(n, q):
    if "error" in q:
        print(f"\nCheck {n}: the checker gave no usable report ({q['error']})")
        return
    print(f"\nCheck {n} ({q.get('model')}): {'PASS' if q['passed'] else 'FAIL'}")
    for c in q["claims"]:
        where = "question" if c["line"] is None else f"line {c['line']}"
        status = c.get("status", "")
        if c.get("kind") == "working":
            basis = (f"[working, builds on {', '.join(c['builds_on']) or 'no source'}"
                     f"{', does NOT follow' if c['follows'] is False else ''}]")
        elif status == "grounded":
            basis = f"[{c['source']}] \"{c['quote'][:70]}\""
        elif status == "model knowledge":
            basis = "[model knowledge: not in the sources -- check when reviewing]"
        else:
            basis = f"[{status}]" + (f" [{c['source']}] \"{c['quote'][:60]}\"" if c.get("source") else "")
        print(f"  {'ok ' if c['supported'] else 'NO '} ({where}) {c['claim'][:80]}")
        print(f"        {basis}")
    counts = q.get("counts") or {}
    if counts:
        print(f"  facts: {counts.get('grounded', 0)} grounded, {counts.get('model knowledge', 0)} model "
              f"knowledge, {counts.get('working', 0)} working steps  ({q.get('mode', '?')} grounding)")
    for problem in q["problems"]:
        print(f"  ! {problem}")


def print_checks(r):
    """Every rejected draft with the check that rejected it, then the last check."""
    reports = r.get("quality_reports") or []
    drafts = r.get("drafts") or []
    for n, q in enumerate(reports, start=1):
        if n < len(reports) and n <= len(drafts):
            print(f"\n----- draft {n} (rejected) -----")
            print_question(drafts[n - 1])
        print_check(n, q)
    if reports:
        verdict = "ACCEPTED" if accepted(r) else "REJECTED"
        print(f"\n{verdict} after {r.get('draft_attempts', 0)} draft(s)")
        last = reports[-1]
        if accepted(r) and last.get("mode") == "balanced":
            review = [c["claim"] for c in last["claims"] if c["status"] == "model knowledge"]
            print("The answer outline is a draft answer key: check it before use."
                  + (" Facts from the model's own knowledge:" if review else ""))
            for claim in review:
                print(f"  - {claim}")


def print_result(r):
    t = r["current_topic"]
    print(f"\nTopic : {t['id']} {t['name']} (semester {t['semester']})")
    b = r.get("blend_topic")
    print("Blend : " + (f"{b['id']} {b['name']} (semester {b['semester']}, {b['hops']} hop(s) back)"
                        if b else "none"))
    print("COs   : " + ", ".join(c["co_id"] for c in r["topic_cos"]))
    print(f"\nPrompt instruction:\n  {r['prompt_instruction']}")

    print(f"\nGrounding chunks ({len(r['relevant_chunks'])}):")
    for i, c in enumerate(r["relevant_chunks"], start=1):
        # d = distance to what found it: the topic, or (found_by "explains ...") the tutorial question
        print(f"  S{i:<2} [{c['role']:5}] {c['type']:8} {c['topic_id']:8} d={c['distance']:.3f}  "
              f"{c['section'][:34]:34}  found by: {c.get('found_by', 'topic')}")
        print(f"          {c['text'][:110].replace(chr(10), ' ')}")

    print(f"\nStyle examples ({len(r['style_examples'])}):")
    for q in r["style_examples"]:
        print(f"  [{q['match']:7}] {q['id']:22} {q['marks']} marks, {q['bloom_level']}, d={q['distance']:.3f}")
        print(f"          {q['text'][:110].replace(chr(10), ' ')}")

    if r.get("grounding_warning"):
        print(f"\nWARNING: {r['grounding_warning']}")
    print_draft(r)
    print_checks(r)


def grounding_from_settings():
    from settings import GROUNDING
    if GROUNDING not in GROUNDING_MODES:
        raise SystemExit(f"GROUNDING in .env must be one of {', '.join(GROUNDING_MODES)}, not {GROUNDING!r}")
    return GROUNDING


def main():
    ap = argparse.ArgumentParser(description="Write and check one question for one topic")
    ap.add_argument("topic_id", nargs="?", default="ALGO_06")
    ap.add_argument("difficulty", nargs="?", default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("marks", nargs="?", type=int, default=10)
    ap.add_argument("--no-llm", action="store_true",
                    help="stop after retrieve: no API key needed, nothing is sent to a model")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the prompt the model gets (works with --no-llm too)")
    ap.add_argument("--no-check", action="store_true",
                    help="stop after draft: no quality check, no rewrites")
    ap.add_argument("--max-drafts", type=int, default=MAX_DRAFTS,
                    help="drafts to try before the question is rejected")
    ap.add_argument("--grounding", choices=GROUNDING_MODES, default=None,
                    help="override GROUNDING from .env (balanced | strict)")
    args = ap.parse_args()
    grounding = args.grounding or grounding_from_settings()

    llm = checker = None
    if not args.no_llm:
        from pipeline.llm import make_llm
        from settings import LLM_CHECK_MODEL
        llm = make_llm()          # before connect(): a missing key stops the run at once
        if LLM_CHECK_MODEL and not args.no_check:
            checker = make_llm(model=LLM_CHECK_MODEL)

    res = connect(llm=llm, checker=checker)
    try:
        graph = build_graph(res, check=not args.no_check, max_drafts=args.max_drafts, grounding=grounding)
        result = graph.invoke({
            "current_topic_id": args.topic_id,
            "target_difficulty": args.difficulty,
            "target_marks": args.marks,
        })
        if args.show_prompt:
            if any(c["role"] == "main" for c in result["relevant_chunks"]):
                for m in build_messages(result):
                    print(f"\n===== prompt: {m['role']} =====\n{m['content']}")
            else:
                print("\n(no prompt: this topic has no grounding text, so nothing is sent to the model)")
        print_result(result)
        if llm is not None:
            sent = f"{llm.requests} request(s) sent to {llm.model}"
            if checker is not None:
                sent += f", {checker.requests} to {checker.model}"
            print(f"\n({sent})")
    finally:
        res.close()


if __name__ == "__main__":
    main()
