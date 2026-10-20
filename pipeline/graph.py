"""
The pipeline graph:   START -> graph_lookup -> retrieve -> draft -> END

draft (the language model) is added when the Resources carry an llm; without
one (--no-llm, or the tests that only check retrieval) the graph ends after
retrieve, as before.

Later phases add steps after draft (quality check -> classify -> dedup ->
constraint check), including the loops back that are the reason this project
uses LangGraph instead of a plain chain.

Try it (from the repo root; Neo4j + Chroma must be running):
    python3 -m pipeline.graph ALGO_06 hard 10            # writes one question
    python3 -m pipeline.graph DS_03 medium 5 --show-prompt
    python3 -m pipeline.graph DS_03 easy 5 --no-llm --show-prompt   # no API call
    python3 -m pipeline.graph SDF1_01 easy 2             # no grounding text: not drafted
The model is set in .env (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY, see settings.py).
"""
import argparse

from langgraph.graph import END, START, StateGraph

from pipeline.nodes.draft import build_messages, make_draft_node
from pipeline.nodes.graph_lookup import make_graph_lookup_node
from pipeline.nodes.retrieve import make_retrieve_node
from pipeline.resources import Resources, connect
from pipeline.state import GraphState


def build_graph(res: Resources):
    builder = StateGraph(GraphState)
    builder.add_node("graph_lookup", make_graph_lookup_node(res.driver))
    builder.add_node("retrieve", make_retrieve_node(
        res.grounding_collections, res.pyq_collection, res.embed_fn))
    builder.add_edge(START, "graph_lookup")
    builder.add_edge("graph_lookup", "retrieve")
    if res.llm is not None:
        builder.add_node("draft", make_draft_node(res.llm))
        builder.add_edge("retrieve", "draft")
        builder.add_edge("draft", END)
    else:
        builder.add_edge("retrieve", END)
    return builder.compile()


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


def main():
    ap = argparse.ArgumentParser(description="Run graph_lookup -> retrieve -> draft for one topic")
    ap.add_argument("topic_id", nargs="?", default="ALGO_06")
    ap.add_argument("difficulty", nargs="?", default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("marks", nargs="?", type=int, default=10)
    ap.add_argument("--no-llm", action="store_true",
                    help="stop after retrieve: no API key needed, nothing is sent to a model")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the prompt the model gets (works with --no-llm too)")
    args = ap.parse_args()

    llm = None
    if not args.no_llm:
        from pipeline.llm import make_llm
        llm = make_llm()          # before connect(): a missing key stops the run at once

    res = connect(llm=llm)
    try:
        result = build_graph(res).invoke({
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
            print(f"\n({llm.requests} request(s) sent to {llm.model})")
    finally:
        res.close()


if __name__ == "__main__":
    main()
