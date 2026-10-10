# Bug review: Question_Paper_Generator (read-only, nothing changed)

## Context

You asked for a bug review of the whole repo with no changes. I read every source file (pipeline, retrieval, embedding, extraction, loader, YAMLs, tests) and checked each suspect against the real data files. No repo file was edited, and nothing was written to Chroma or Neo4j.

**This is a report, not a change plan.** Approving it changes nothing. The fix order at the end is only a suggestion for when you want to act.

## What is healthy

- Unit tests pass: 23/23 (`python3 -m unittest tests.test_phase1`).
- Chroma on disk matches the repo exactly: `book_content` 5516, `tut_content` 24, `pyq_bank` 53 rows; 0 missing, 0 stale, metadata identical to the JSON/JSONL files.
- All 10 PYQ papers parsed correctly: 53 questions, mark totals 20/20/35, right subject, year and exam, no duplicate ids.
- All 15 tutorial sheets are covered by the 18 tutorial chunks; `topic_map.yaml` chapter numbers match the real chapter titles.
- Docker was not running, so the live pipeline (`python -m pipeline.graph`) and the Neo4j loader were not exercised end to end.

## A. Bugs affecting results today

**A1. The embedding model only reads the first part of each book chunk.**
`Extracting/extract.py:8` targets 350 words and `split_into_chunks` (`:74-96`) never splits a long paragraph, but `all-MiniLM-L6-v2` reads at most 256 tokens.
- 92–98% of usable book chunks are over the limit.
- Median share of a chunk that is actually embedded: DS 36%, APS 48%, SDF-1 59%, SDF-2 63%.
- Search (and the PYQ topic scoring in `resolve_pyq_labels.py`) therefore ignores the rest of each chunk.

**A2. Chapter title pages are stored as chunks and win the search.**
`extract.py:92-94` merges a short *last* chunk but not a short *first* one. Eleven usable SDF-2 chunks are under 20 words, e.g. `SDF-2_023_00` = "Chapter 16 / Inheritance / 419 / C++". In a simulated search this is the #1 grounding hit for SDF2_03; the same happens for SDF2_04 and SDF1_05.

**A3. Duplicate chunks, and retrieval does not top up after removing them.**
- `extract.py:35-39` ends each section on the next section's start page, inclusive (`:72`), so sections sharing a page produce identical text. In `DS_chunks.jsonl`, 1,374 of 2,734 chunks are exact copies of another chunk (APS 41, SDF-2 3). Sixteen texts carry two different `topic_id`s.
- `retrieval.py:85-100` asks each collection for only `n`, drops duplicate texts, and never fetches more. A fake collection asked for 4 returned 1. On real data, ALGO_01 gets 2 of 4 chunks, and ALGO_07 and DS_06 get 3.
- The de-duplication is per call, so a tutorial chunk tagged with both topics (e.g. `DS_tut_chunk_007`: DS_02 + DS_01) can come back as both "main" and "blend" (`retrieval.py:112-117`).

**A4. The blend-topic choice is order-dependent and ignores edge strength** (`pipeline/difficulty_blend.py`).
- The query sorts by hops only (`:28`), then the code takes `direct[0]` (`:53`, `:62`) or a sort with ties (`:69-70`). Ties fall to Neo4j's row order, which is not guaranteed. Simulated from the YAMLs, the medium pick is order-dependent for 17 of 30 topics and the hard pick for 21 of 30. The tests cannot catch this because the fake driver returns a fixed order.
- `strength` (hard/soft) is loaded into Neo4j but never read, so "medium" can blend a soft prerequisite.
- The hard sort puts semester first (`:69`), so every DS/ALGO topic blends a semester-1 topic, up to 4 hops away (e.g. NP-completeness → Arrays). The comment at `:66-68` says the sort avoids "an obscure 4-hop trivia pick".
- An unknown difficulty returns `None` silently when the topic has no prerequisites and raises only otherwise (`:48`). The CLI restricts the choices, so this only affects programmatic callers.

**A5. A student's name and enrollment number are committed.**
- `redact_pii` (`split_pyq_questions.py:193-197`) only drops lines containing "enrollment" or a LaTeX-underlined name. The headers in `pyq_structured/DS_2024_endsem_pyq.json` and `SDF1_2023_test2_pyq.json` still contain the name.
- All 10 `pyq_pages/*.jsonl` files are never redacted and contain the name plus enrollment number.
- If the GitHub repo is public, this is exposed, along with the four full textbook PDFs in `Books/`.

## B. Latent bugs (current data is fine; new input will trigger them)

**B1. Chroma settings in `.env` are ignored.**
`pipeline/graph.py:17` imports `retrieval` (which reads `CHROMA_HOST/PORT` at import, `retrieval.py:36-37`) before `:18` imports `pipeline.resources`, which loads `.env`. I confirmed the import order. Every other script hard-codes `localhost:8000`. This is harmless today because `.env` holds the defaults.
Also, `load_dotenv_file` (`resources.py:33-35`) keeps inline comments in the value (`KEY=abc # note` → `'abc # note'`), while Docker Compose strips them.

**B2. Stale vectors after any re-extraction.**
`embed_chunks.py` and `embed_tutorial_chunks.py` only upsert, and chunk ids are positional (`extract.py:115`, `tutorial_chunks.py:1202`). Re-chunking or re-tagging leaves old rows under old topic ids, and retrieval will serve them. `check_chroma.py:107-110` reports them only as a NOTE.

**B3. PYQ splitter** (`split_pyq_questions.py`), each confirmed with a small input:
- `:40` — the question regex is not anchored to a line start. "…two queues Q1 and Q2 …" turns 2 questions into 4 blocks with duplicate ids, and `embed_pyqs.py:95-98` then aborts.
- `:40` — "Q1(a)", "Q2)" and "Q3-" are not recognised (0 questions found, with a warning).
- `:155-162` — "Software Development Fundamentals-I" followed by a word starting with "I" is detected as SDF2, even when the filename says SDF 1.
- `:61-78` — `[CO2, Apply, 5 Marks]`, `[5 Marks, CO2]`, `(CO2, 5 Marks)`, `[CO-2, 5M]` and `[… 5 marks each]` give no CO or marks (a NOTE is printed).
- `:173` — the year is the first "20xx", so "2023-24 … May 2024" gives 2023.
- `:456` — two papers with the same subject/year/exam (or no year → "None") overwrite each other.

**B4. Tutorial extractor** (`Extracting/tutorial_chunks.py`):
- `:209-221` — any line repeated on at least half the pages is treated as a running header, anywhere on the page. On a multi-page sheet with code, lines like `{`, `}`, `else`, `return 0;` are deleted from every question. Meanwhile `:194-197` compares raw text with lower-cased keys, so a real header containing capitals is *not* removed. Both confirmed.
- `:255` — one stray line such as "12. …" at the margin makes every later question up to Q12 merge into one; restarted numbering (Tutorial Q1–2, Assignment Q1–3) merges too.
- `:474-485` — if every Gemini model is reported unavailable, `_pick_model` sleeps 90 s and recurses without using an attempt: a roughly 24-hour hang instead of an error.
- `:725-738` — `topic_map.yaml` has no `subjects:` block, so the hard-coded DS + APS catalog (`:693-722`) is always used. SDF tutorials are skipped, and a topic renamed in the YAML never reaches the tagger.
- `:1185` — the default `--prereq-depth 1` adds prerequisite topics' COs to `cos` (DS_04 → `[CO4, CO1]`); this lands in Chroma as `cos_all`.
- `:1177` — diagram transcription is outside the try/except, so one Gemini failure aborts the run before any file is written.
- `:346-347` — an image above Q1 (e.g. a logo) is attached to Q1; `:535` sends every non-PNG as JPEG.

**B5. `tag_topics.py`:**
- `:274` — `--push` still writes to Chroma with `--report-only` ("change nothing"), and deletes vectors if `--drop-suspect` is also set.
- `--drop-suspect` without `--push` removes chunks from the JSONL; a later `--push` can no longer delete them from Chroma.
- `:160` — division by zero if a book ends up with 0 chunks; the report runs before the write.

**B6. `resolve_pyq_labels.py` / `embed_pyqs.py`:**
- `sanitize_metadata` (`embed_pyqs.py:45-46`) stores `None` as the string "NONE". In `pyq_bank`, `topic_confidence` is a float on 26 rows and "NONE" on 19, so numeric filters will skip those rows. Retrieval would pass `-1` marks and "NONE" Bloom values to the future prompt (`retrieval.py:132-133`).
- `:170-174` — a record marked manual keeps its old `needs_review`, and `topic_id_from_co` records the manual topic, not the splitter's.
- Because of A3, a topic's "top 3 chunks" can be three copies of one page, which inflates DS-handbook topics in the content-override decision.

**B7. `pdf_extract_full.py`:**
- `:168` — the `subject` field is the filename ("End Sem SDF 2"), and `page_id` repeats it. Visible in the data; unused downstream.
- `:102` — `message.content` can be `None`. Classifier output such as "User Safety: safe" is saved as page text and never regenerated (`:172`); the splitter then drops that page silently.
- `:179-188` — a failure on the last page discards every page already transcribed for that PDF.

## C. Minor

- `Extracting/pyqs_processing/test_pyq_embed.py` is a script named like a test; `pytest` from the repo root would import it, loading the model and connecting to Chroma.
- `pipeline/difficulty_blend.py:111-147` is a dead older copy of the node, with a hard-coded password and no unknown-topic check.
- `pipeline/resources.py:98` — if Chroma is down a raw exception escapes and the Neo4j driver opened at `:85` is not closed.
- `docker-compose.yml:27` uses `chromadb/chroma:latest` unpinned. All four ports are published on every interface, with the default Neo4j password (your `.env` still uses it) and no Redis password.
- `build_prereq_graph.py:258-273` — `--reset` and the reload are separate transactions, so a mid-load failure leaves a partial graph; `validate()` has no cycle or self-loop check.
- `extract.py:118` will write an absolute local path into `source_file` on the next run; the PYQ `source` values contain Windows backslashes.
- `embed_chunks.py` lacks the `USE_TF=0` workaround the other scripts set.

## Your uncommitted change

`DS_2024_endsem_pyq.json` differs only in `topic_confidence` and `topic_scores` for one question (0.5714 → 0.561). That is a re-run of `resolve_pyq_labels.py --write` after the grounding vectors changed, not a bug. It shows that "same result every time" holds only while `book_content` and `tut_content` are unchanged.

## How this was checked

- Read every tracked source file in full.
- Ran the unit tests with bytecode writing disabled.
- Analysed the JSONL/JSON data files with read-only scripts.
- Opened `chroma_data/chroma.sqlite3` read-only and compared ids and metadata with the files.
- Exercised the real functions with small in-memory inputs (splitter, tutorial extractor with `pymupdf` stubbed, blend selection over a graph built from the YAMLs).
- Simulated grounding retrieval in memory with the locally cached embedding model.

## Suggested fix order (only if and when you want changes)

1. **A5** — remove the name and enrollment number from the committed files and widen `redact_pii`.
2. **A1 + A2 + A3** together — one re-chunking pass: cap chunk size to the model's window, merge tiny leading chunks, stop the page overlap, and over-fetch before de-duplicating in `retrieval.py`. This needs a re-embed, so do **B2** (delete stale ids) at the same time.
3. **A4** — make the blend choice deterministic and decide whether `strength` and hop distance should count.
4. **B1** — one shared config loaded before anything reads the environment.
5. **B3 / B4** — before adding new PYQ papers or tutorial sheets.
