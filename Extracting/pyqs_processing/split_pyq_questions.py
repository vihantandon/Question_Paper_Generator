# split_pyq_questions.py
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

PAGES_DIR = Path("Extracting/Processed")
# Output now lives in its own folder, separate from the raw page-level
# jsonl files in Extracting/Processed, so the two don't get mixed together.
OUT_DIR = Path("Extracting/PYQ_Structured")

# Question boundary. The trailing punctuation after the number is OPTIONAL
# ("Q.1 Add code..." has no ":"/"." right after the "1", just whitespace),
# so we only require a following whitespace character as the boundary.
# Still supports bold-markdown headers ("**Q1:**") via the optional "\*{0,2}".
QUESTION_START_RE = re.compile(r'\*{0,2}Q\.?\s*(\d+)\s*[:.]?\*{0,2}(?=\s)')

# Some pages carry a redundant markdown heading directly above the real
# question line, e.g.:
#   ### Q.1
#   `Q.1 Add code in line no. 4...`
# Both would match QUESTION_START_RE, creating a spurious duplicate split
# (an empty "Q.1" block followed by the real one). These heading-only lines
# repeat what's already in the body text below, so drop them before splitting.
REDUNDANT_QUESTION_HEADING_RE = re.compile(r'(?m)^#{1,6}\s*Q\.?\s*\d+[a-zA-Z]?\s*[:.]?\s*$\n?')

# Pages that failed extraction / got flagged rather than transcribed
# (e.g. "User Safety: safe") -- not real question content, skip them.
JUNK_PAGE_RE = re.compile(r'^\s*user\s+safety\s*:', re.IGNORECASE)
MIN_PAGE_TEXT_LEN = 30

# A CO/course-outcome code, either course-specific ("C109.4", "C210.3") or
# canonical ("CO2", "CO 2"). Reused by both the marks-tag patterns and the
# course-outcome table parser below.
_CODE = r'[A-Za-z]+\.?\s*\d+(?:\.\d+)?'

MARKS_PATTERNS = [
    # with Bloom's level: [CO2(Applying), 6 Marks] / [C109.4 (Applying), 5 Marks]
    #                     [CO1 (Analyse level); 5 Marks] / [CO2(Applying), 5M]
    # The marks group captures the WHOLE numeric expression before "Marks"
    # (e.g. "2+2", "3+1+2", "1 + 1.5 + 1.5 + 2", "2+2+2=6") so it can be
    # summed correctly afterwards instead of just grabbing the last number.
    re.compile(
        rf'\[\s*({_CODE})\s*\(([^)]+)\)\s*[,;:]?\s*([\d.+\-=\s]+?)\s*(?:Marks?|M)\b\.?\s*(?:,[^\]]*)?\s*\]',
        re.IGNORECASE,
    ),
    # without Bloom's level: [CO2, 6 Marks] / [CO2, 2+2+2=6 Marks] / [C109.4, 5M]
    # also handles the bloom level trailing AFTER marks with no parens:
    # [CO1, 4 Marks, Understand]
    re.compile(
        rf'\[\s*({_CODE})\s*[,;:]?\s*([\d.+\-=\s]+?)\s*(?:Marks?|M)\b\.?\s*(?:,\s*([^\]]+?))?\s*\]',
        re.IGNORECASE,
    ),
]

# Course-outcome table row in the paper's header, either layout:
#   LaTeX-ish:  "C109.1 & Explain the logic for solving problems... \\"
#   Markdown:   "| CO1 | Explain the logic for solving problems... |"
#               "| **C109.5** | Apply and implement arrays... |"  (bold cells)
CO_TABLE_ROW_RE_LATEX = re.compile(r'([A-Za-z]+\d+(?:\.\d+)?)\s*&\s*(.+?)\s*\\\\')
CO_TABLE_ROW_RE_MD = re.compile(
    r'\|\s*\*{0,2}([A-Za-z]+\.?\s*\d+(?:\.\d+)?)\*{0,2}\s*\|\s*\*{0,2}([^|]{15,}?)\*{0,2}\s*\|'
)

FILENAME_SUBJECT_HINTS = [
    ("sdf1", "SDF1"), ("sdf2", "SDF2"),
    ("dsa", "DS"),
    ("aps", "ALGO"), ("algo", "ALGO"),
]


def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def normalize_code(raw_code: str) -> str:
    """'C109.1' -> 'C109.1', 'CO 4' -> 'CO4', 'co4' -> 'CO4'."""
    return re.sub(r'\s+', '', raw_code).upper()


def load_subject_yamls(yaml_paths: list[str]) -> dict:
    out = {}
    for p in yaml_paths:
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        out[data["subject"]] = data
    return out


def build_co_to_topics(subject_yaml: dict) -> dict:
    m = defaultdict(list)
    for t in subject_yaml["topics"]:
        for co in t.get("co", []):
            m[co].append(t)
    return m


def build_code_to_co(header_text: str, co_descriptions: dict) -> dict:
    """
    Maps whatever CO code a paper actually prints (course-specific, e.g.
    "C109.4", or canonical, e.g. "CO4") to the canonical "CO<n>" key used in
    the subject yaml. Seeds an identity mapping for canonical codes first
    (so papers that print "CO4" directly resolve with no table needed), then
    parses the course-outcome table printed in the header and fuzzy-matches
    each row's description against the yaml's CO descriptions to resolve
    course-specific codes like "C109.4" -> "CO4".
    """
    code_to_co = {co_id: co_id for co_id in co_descriptions}  # identity fallback

    rows = CO_TABLE_ROW_RE_LATEX.findall(header_text) + CO_TABLE_ROW_RE_MD.findall(header_text)
    for raw_code, raw_desc in rows:
        raw_code_norm = normalize_code(raw_code)
        raw_desc_clean = re.sub(r'\s+', ' ', raw_desc).strip()
        if len(raw_desc_clean) < 15:
            continue
        best_co, best_score = None, 0.0
        for co_id, co_desc in co_descriptions.items():
            score = difflib.SequenceMatcher(None, raw_desc_clean.lower(), co_desc.lower()).ratio()
            if score > best_score:
                best_co, best_score = co_id, score
        if best_co and best_score >= 0.55:  # loose threshold; table text is OCR'd
            code_to_co[raw_code_norm] = best_co
    return code_to_co


def detect_subject(header_text: str, subject_yamls: dict, filename: str = "") -> str | None:
    norm_header = normalize(header_text)
    candidates = sorted(subject_yamls.items(),
                         key=lambda kv: len(normalize(kv[1]["subject_name"])),
                         reverse=True)
    for code, y in candidates:
        if normalize(y["subject_name"]) in norm_header:
            return code

    # header text didn't help — fall back to the filename's naming convention
    fname_norm = normalize(filename)
    for hint, code in sorted(FILENAME_SUBJECT_HINTS, key=lambda kv: -len(kv[0])):
        if hint in fname_norm and code in subject_yamls:
            return code
    return None


def detect_exam_and_year(header_text: str, filename: str = "") -> tuple[str, int | None]:
    year_match = re.search(r'(20\d{2})', header_text)
    year = int(year_match.group(1)) if year_match else None

    norm = normalize(header_text)
    if "endsem" in norm or "endterm" in norm:
        return "endsem", year
    if "midsem" in norm or "midterm" in norm:
        return "midsem", year

    # Underscore counts as a "word" character in regex, so "\bt1\b" never
    # matches inside "T1_APS_pages" (no boundary between "1" and "_").
    # Treat underscores as separators before checking.
    filename_spaced = re.sub(r'[_\W]+', ' ', filename)
    if re.search(r'\bt\s*1\b', filename_spaced, re.I):
        return "test1", year
    if re.search(r'\bt\s*2\b', filename_spaced, re.I):
        return "test2", year
    return "unknown", year


def redact_pii(header_text: str) -> str:
    lines = header_text.split("\n")
    kept = [l for l in lines if "enrollment" not in l.lower()
            and not re.search(r'name\s*\$?\\underline', l, re.I)]
    return "\n".join(kept)


def split_into_questions(full_text: str) -> tuple[str, list[str]]:
    matches = list(QUESTION_START_RE.finditer(full_text))
    if not matches:
        return full_text, []
    header = full_text[:matches[0].start()]
    blocks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        blocks.append(full_text[m.start():end].strip())
    return header, blocks


def _clean_marks(value: float) -> float | int:
    return int(value) if float(value).is_integer() else value


def _resolve_marks_expr(expr: str):
    """
    Turns the raw numeric text captured before 'Marks' into a single total.
    Two conventions show up in the papers:
      - an explicit total after '=', e.g. "2+2+2=6"  -> use the 6, ignore the rest
      - a plain addition with no '=', e.g. "2+2" or "3+1+2" or "1 + 1.5 + 1.5 + 2"
        -> sum every number in the expression (the ORIGINAL script only grabbed
        the last number here, which silently undercounted these questions)
    A plain single number, e.g. "5", is handled the same way (sum of one number).
    """
    if '=' in expr:
        expr = expr.rsplit('=', 1)[1]
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', expr)]
    if not nums:
        return None
    return _clean_marks(sum(nums))


def tidy_markdown_noise(text: str) -> str:
    """Strips leftover markdown artifacts from the vision transcription that
    become visible once the [CO..., Marks] tag is removed: stray inline-code
    backtick wrappers and standalone '---' horizontal-rule lines. Careful to
    leave real ``` code-fence markers alone (a fence line has 3+ backticks;
    a stray wrapper line has 1-2), then collapses the resulting blank lines."""
    lines = text.split('\n')
    kept = []
    for line in lines:
        compact = re.sub(r'\s+', '', line)
        if compact and set(compact) == {'`'} and len(compact) < 3:
            continue  # e.g. "`   `" -- leftover inline-code wrapper, not a ``` fence
        if re.fullmatch(r'-{3,}', line.strip()):
            continue  # markdown horizontal rule
        kept.append(line)
    text = '\n'.join(kept)
    # Remove leftover isolated single backticks (inline-code wrapping the vision
    # model added around whole lines), while leaving ``` code-fence markers
    # alone -- a backtick is only stripped if it has no backtick touching it.
    text = re.sub(r'(?<!`)`(?!`)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_marks_block(question_text: str, code_to_co: dict) -> dict:
    """
    Finds EVERY [CO..., Marks] tag in the block -- not just the first -- since
    a multi-part question (a)/(b)/(c) is often tagged with a separate CO+marks
    bracket per sub-part (e.g. 3M + 1.5M + 1.5M), and the original script only
    picked up the first one, silently dropping the rest of the question's marks.
    """
    candidates = []
    for idx, pattern in enumerate(MARKS_PATTERNS):
        for m in pattern.finditer(question_text):
            candidates.append((m.start(), m.end(), idx, m))
    # Both patterns can match the same tag; keep the earliest-starting,
    # non-overlapping matches (pattern 0 -- the bloom-parens form -- wins
    # ties since it's the more specific of the two).
    candidates.sort(key=lambda c: (c[0], c[2]))
    kept = []
    last_end = -1
    for start, end, idx, m in candidates:
        if start >= last_end:
            kept.append((start, end, idx, m))
            last_end = end

    if not kept:
        return {"raw_co_code": None, "co_short": None, "bloom_level": None, "marks": None,
                "marks_breakdown": None, "clean_text": question_text.strip()}

    tags = []
    total_marks = 0.0
    any_marks = False
    for start, end, idx, m in kept:
        raw_code = normalize_code(m.group(1))
        if idx == 0:
            bloom = m.group(2).strip()
            marks_expr = m.group(3)
        else:
            marks_expr = m.group(2)
            bloom = m.group(3).strip() if m.group(3) else None
        marks_val = _resolve_marks_expr(marks_expr)
        if marks_val is not None:
            total_marks += marks_val
            any_marks = True
        tags.append({"raw_co_code": raw_code, "co_short": code_to_co.get(raw_code),
                     "bloom_level": bloom, "marks": marks_val})

    # Strip every matched tag span out of the text (not just the first).
    pieces, prev = [], 0
    for start, end, idx, m in kept:
        pieces.append(question_text[prev:start])
        prev = end
    pieces.append(question_text[prev:])
    clean_text = tidy_markdown_noise("".join(pieces))

    primary = tags[0]
    return {
        "raw_co_code": primary["raw_co_code"],
        "co_short": primary["co_short"],
        "bloom_level": primary["bloom_level"],
        "marks": _clean_marks(total_marks) if any_marks else None,
        # None for a simple single-tag question; the full per-sub-part
        # breakdown (each with its own CO/bloom/marks) when a question had
        # more than one tag, so multi-CO sub-parts stay auditable.
        "marks_breakdown": tags if len(tags) > 1 else None,
        "clean_text": clean_text,
    }


def assign_topic(co_short: str, co_to_topics: dict) -> dict:
    candidates = co_to_topics.get(co_short, [])
    if len(candidates) == 1:
        t = candidates[0]
        return {"topic_id": t["id"], "topic_name": t["name"], "needs_review": False, "review_reason": None}
    if not candidates:
        return {"topic_id": None, "topic_name": None, "needs_review": True,
                "review_reason": f"CO '{co_short}' has no topics mapped to it"}
    cand_str = ", ".join(f"{t['id']} ({t['name']})" for t in candidates)
    return {"topic_id": None, "topic_name": None, "needs_review": True,
            "review_reason": f"CO '{co_short}' maps to {len(candidates)} topics — pick one: {cand_str}"}


def process_pages_file(pages_path: Path, subject_yamls: dict) -> list[dict]:
    rows = [json.loads(l) for l in pages_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["page_num"])

    # Drop pages that failed extraction entirely (e.g. "User Safety: safe")
    # rather than let junk content pollute the header/question text.
    good_rows = []
    skipped = 0
    for r in rows:
        text = r.get("text", "")
        if len(text.strip()) < MIN_PAGE_TEXT_LEN or JUNK_PAGE_RE.match(text.strip()):
            skipped += 1
            continue
        good_rows.append(r)
    if skipped:
        print(f"  skipped {skipped} junk/empty page(s) in {pages_path.name}")
    rows = good_rows
    if not rows:
        print(f"  WARNING: no usable pages left in {pages_path.name} after filtering junk pages.")
        return []

    full_text = "\n".join(r["text"] for r in rows)
    full_text = REDUNDANT_QUESTION_HEADING_RE.sub("", full_text)
    source_file = rows[0]["source_file"]

    header_block, question_blocks = split_into_questions(full_text)
    subject_code = detect_subject(header_block, subject_yamls, filename=pages_path.stem)
    exam, year = detect_exam_and_year(header_block, filename=pages_path.stem)

    if subject_code is None:
        snippet = header_block[:300].replace("\n", " ")
        print(f"  WARNING: could not detect subject for {pages_path.name} "
              f"(checked header text and filename).")
        print(f"  Header snippet: {snippet!r}")
        return []

    if not question_blocks:
        snippet = full_text[:400].replace("\n", " ")
        print(f"  WARNING: 0 questions found in {pages_path.name} — the 'Q<number>' "
              f"pattern found no matches. This paper likely numbers questions differently.")
        print(f"  First 400 chars: {snippet!r}")

    subject_yaml = subject_yamls[subject_code]
    co_to_topics = build_co_to_topics(subject_yaml)
    co_descriptions = subject_yaml["course_outcomes"]
    code_to_co = build_code_to_co(header_block, co_descriptions)

    records = [{
        "id": f"{subject_code}_{year}_{exam}_header",
        "document": redact_pii(header_block).strip(),
        "metadata": {"source": source_file, "subject": subject_code,
                    "subject_name": subject_yaml["subject_name"],
                    "year": year, "exam": exam, "type": "header"},
    }]

    total_marks = 0
    unmatched_marks = 0
    first_unmatched_snippet = None

    for i, block in enumerate(question_blocks, start=1):
        q_num_match = QUESTION_START_RE.match(block)
        q_num = int(q_num_match.group(1)) if q_num_match else i
        parsed = parse_marks_block(block, code_to_co)

        if parsed["marks"] is None:
            unmatched_marks += 1
            if first_unmatched_snippet is None:
                first_unmatched_snippet = block[-200:].replace("\n", " ")
        else:
            total_marks += parsed["marks"]

        topic_info = {"topic_id": None, "topic_name": None, "needs_review": True,
                      "review_reason": "no CO detected in question text"}
        if parsed["co_short"] and parsed["co_short"] in co_descriptions:
            topic_info = assign_topic(parsed["co_short"], co_to_topics)
        elif parsed["raw_co_code"]:
            topic_info["review_reason"] = (
                f"code '{parsed['raw_co_code']}' could not be resolved to a CO in "
                f"{subject_code}.yaml (course-outcome table missing or didn't match)"
            )

        records.append({
            "id": f"{subject_code}_{year}_{exam}_q{q_num}",
            "document": parsed["clean_text"],
            "metadata": {
                "source": source_file, "subject": subject_code, "year": year, "exam": exam,
                "type": "question", "question_number": q_num,
                "raw_co_code": parsed["raw_co_code"],
                "course_outcome": parsed["co_short"],
                "co_description": co_descriptions.get(parsed["co_short"]),
                "bloom_level": parsed["bloom_level"],
                "marks": parsed["marks"], "marks_breakdown": parsed["marks_breakdown"],
                **topic_info,
            },
        })

    print(f"  {pages_path.name}: {subject_code} {year} {exam} | "
          f"{len(question_blocks)} questions, {total_marks} marks, "
          f"{sum(1 for r in records[1:] if not r['metadata']['needs_review'])} auto-tagged")
    if unmatched_marks:
        print(f"  NOTE: {unmatched_marks}/{len(question_blocks)} questions had no "
              f"recognizable marks tag. Example (last 200 chars): {first_unmatched_snippet!r}")
    return records


def process_all(pages_dir=PAGES_DIR, out_dir=OUT_DIR, yaml_paths=None):
    yaml_paths = yaml_paths or ["sdf1.yaml", "sdf2.yaml", "ds.yaml", "algo.yaml"]
    subject_yamls = load_subject_yamls(yaml_paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pages_file in sorted(pages_dir.glob("*_pages.jsonl")):
        print(f"Processing {pages_file.name}...")
        records = process_pages_file(pages_file, subject_yamls)
        if not records:
            continue
        meta = records[0]["metadata"]
        out_path = out_dir / f"{meta['subject']}_{meta['year']}_{meta['exam']}_pyq.json"
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {len(records)} records -> {out_path}\n")


if __name__ == "__main__":
    process_all()