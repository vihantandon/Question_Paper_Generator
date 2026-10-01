"""
The question-generation pipeline, built with LangGraph.

    pipeline/
      state.py            what flows between the steps (GraphState)
      resources.py        Neo4j / Chroma / embedding model, opened once
      difficulty_blend.py graph queries + blend rules (easy / medium / hard)
      nodes/
        graph_lookup.py   step 1: topic, COs, which prerequisite to blend in
        retrieve.py       step 2: grounding text + similar past questions
      graph.py            wires the steps together; run it to try the pipeline

Run from the repo root:
    python3 -m pipeline.graph ALGO_06 hard 10
"""
