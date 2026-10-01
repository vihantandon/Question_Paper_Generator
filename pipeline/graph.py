"""
The pipeline graph:   START -> graph_lookup -> retrieve -> END

Later phases add steps after retrieve (draft -> quality check -> classify
-> dedup -> constraint check), including the loops back that are the
reason this project uses LangGraph instead of a plain chain.

Try it (from the repo root; Neo4j + Chroma must be running):
    python3 -m pipeline.graph ALGO_06 hard 10
    python3 -m pipeline.graph SDF1_01 easy 2       # shows the no-grounding warning
"""
import argparse

from langgraph.graph import END, START, StateGraph

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
    builder.add_edge("retrieve", END)
    return builder.compile()


def print_result(r):
    t = r["current_topic"]
    print(f"\nTopic : {t['id']} {t['name']} (semester {t['semester']})")
    b = r.get("blend_topic")
    print("Blend : " + (f"{b['id']} {b['name']} (semester {b['semester']}, {b['hops']} hop(s) back)"
                        if b else "none"))
    print("COs   : " + ", ".join(c["co_id"] for c in r["topic_cos"]))
    print(f"\nPrompt instruction:\n  {r['prompt_instruction']}")

    print(f"\nGrounding chunks ({len(r['relevant_chunks'])}):")
    for c in r["relevant_chunks"]:
        print(f"  [{c['role']:5}] {c['type']:8} {c['topic_id']:8} d={c['distance']:.3f}  "
              f"{c['section'][:40]}")
        print(f"          {c['text'][:110].replace(chr(10), ' ')}")

    print(f"\nStyle examples ({len(r['style_examples'])}):")
    for q in r["style_examples"]:
        print(f"  [{q['match']:7}] {q['id']:22} {q['marks']} marks, {q['bloom_level']}, d={q['distance']:.3f}")
        print(f"          {q['text'][:110].replace(chr(10), ' ')}")

    if r.get("grounding_warning"):
        print(f"\nWARNING: {r['grounding_warning']}")


def main():
    ap = argparse.ArgumentParser(description="Run graph_lookup -> retrieve for one topic")
    ap.add_argument("topic_id", nargs="?", default="ALGO_06")
    ap.add_argument("difficulty", nargs="?", default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("marks", nargs="?", type=int, default=10)
    args = ap.parse_args()

    res = connect()
    try:
        result = build_graph(res).invoke({
            "current_topic_id": args.topic_id,
            "target_difficulty": args.difficulty,
            "target_marks": args.marks,
        })
        print_result(result)
    finally:
        res.close()


if __name__ == "__main__":
    main()
