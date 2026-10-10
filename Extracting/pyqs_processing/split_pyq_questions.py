# split_pyq_questions.py
"""
Step 2 of the PYQ pipeline: page texts -> one atomic record per question.

    Extracting/Processed/pyq_pages/*_pages.jsonl
        -> Extracting/Processed/pyq_structured/<SUBJ>_<year>_<exam>_pyq.json

COs and topics come from prerequisite_graph/subjects/*.yaml -- the SAME
files the Neo4j loader reads (there is no second copy any more), so a CO you
edit there is what PYQ tagging uses too.

All paths are found relative to this file, so it runs from any folder:
    python Extracting/pyqs_processing/split_pyq_questions.py          # new papers only
    python Extracting/pyqs_processing/split_pyq_questions.py --all    # every paper again

A paper that already has a *_pyq.json file is SKIPPED, so the files you have
checked and fixed by hand are not touched when you add a new paper.
--all splits every paper again and rewrites its file; even then a record
marked "tag_method": "manual" is kept exactly as it is. Other hand edits in
a re-split file are lost, so use --out-dir to write somewhere else and
compare first.
Next steps: Embedding/embed_pyqs.py, then resolve_pyq_labels.py
"""
import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent            # Extracting/pyqs_processing
ROOT = HERE.parent.parent                          # repo root
PROCESSED = ROOT / "Extracting" / "Processed"
PAGES_DIR = PROCESSED / "pyq_pages"                # input  (from pdf_extract_full.py)
OUT_DIR = PROCESSED / "pyq_structured"             # output (read by embed_pyqs.py)
SUBJECTS_DIR = ROOT / "prerequisite_graph" / "subjects"   # single source of truth

# Question boundary: "Q1.", "Q1:", "Q.1", "Q. 5", "**Q1:**", "Q2)", "Q3 -",
# "Q1(a)". The punctuation after the number is OPTIONAL ("Q.1 Add code..."
# has only a space there).
#
# It must be at the START of a line; markdown in front of it is fine
# ("### Q2.", "`Q.1 Add code...`", "> Q3."). It used to match anywhere, so a
# sentence like "use two queues Q1 and Q2" or a note "Under Q2: operator
# overloading" was cut into extra questions with the same ids.
_LINE_DECORATION = r'[ \t]*(?:>[ \t]*)?(?:#{1,6}[ \t]*)?`?'
QUESTION_START_RE = re.compile(
    r'(?m)^' + _LINE_DECORATION +
    r'\*{0,2}Q\.?[ \t]*(\d+)[ \t]*\*{0,2}(?:[ \t]*[:.)\-–—]\*{0,2})?(?=\s|$|\()')

# Some pages carry a markdown heading directly above the real question line:
#   ### Q.1
#   `Q.1 Add code in line no. 4...`
# Both look like the start of question 1, so that heading-only line is dropped
# before splitting (see drop_redundant_headings).
QUESTION_HEADING_ONLY_RE = re.compile(r'(?m)^#{1,6}[ \t]*Q\.?[ \t]*(\d+)[a-zA-Z]?[ \t]*[:.]?[ \t]*$\n?')

# Pages that failed extraction / got flagged or refused rather than
# transcribed (e.g. "User Safety: safe", "I'm sorry, I can't ...") -- not real
# question content. pdf_extract_full.py retries such a page instead of
# storing it; a page stored before that fix is skipped here.
JUNK_PAGE_RE = re.compile(
    r"^\s*(?:user\s+safety\s*:|i\s*['’]?\s*a?m\s+sorry\b|i\s+can(?:not|\s*['’]\s*t)\s+"
    r"(?:assist|help|transcribe|comply|provide)\b)", re.IGNORECASE)
MIN_PAGE_TEXT_LEN = 30

# What a vision model sometimes adds to a page it was asked only to transcribe:
#   Here is the complete transcription of the page, preserving ...
#   ### Handwritten Notes on the Page:
#   *   On the left of Q3: **DMA**
#   *   Under Q2: operator overloading
# Neither is part of the paper, and both used to end up inside the question
# printed above them (see strip_model_chatter).
CHATTER_LINE_RE = re.compile(
    r"^[\s>*_#]*(?:(?:sure|certainly|okay|ok)\b[\s,.!:]*)?(?:here\s+is|here['’]s|below\s+is|this\s+is)\b"
    r".*\btranscri(?:ption|bed|pt)\b.*$", re.IGNORECASE)
HANDWRITTEN_HEADING_RE = re.compile(          # a short heading, not a sentence about handwriting
    r"^[\s>*_#]*hand[\s\-]?written\s+(?:notes?|annotations?|marks?|markings?|comments?)\b.{0,60}$",
    re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[*\-+•]|\d+[.)])\s+")
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")

# The [CO, Bloom's level, marks] tag printed after a question. The three parts
# come in any order and the brackets may be round:
#   [CO2(Applying), 6 Marks]     [C109.4 (Applying), 5 Marks]    [CO2(Applying), 5M]
#   [CO1 (Analyse level); 5 Marks]   [CO2, 2+2+2=6 Marks]        [CO1, 4 Marks, Understand]
#   [CO2, Apply, 5 Marks]        [5 Marks, CO2]      (CO2, 5 Marks)      [CO-2, 5M]
#   [CO2, 5 marks each]          [5 Marks]
# A tag is found by looking inside every bracket pair (parse_tag), not by one
# fixed pattern per layout, which missed every layout it did not list.
TAG_GROUP_RE = re.compile(r'\[([^\[\]]{1,160})\]|\(((?:[^()]|\([^()]*\)){1,160})\)')

# A CO code: canonical ("CO2", "CO 2", "CO-2"), OCR'd ("C03") or
# course-specific ("C109.4", "C210.3").
TAG_CODE_RE = re.compile(r'\bC[A-Za-z]{0,2}[ \t]*[.\-]?[ \t]*\d+(?:\.\d+)?', re.IGNORECASE)

# The marks: the WHOLE numeric expression before "Marks" / "M" ("2+2",
# "3+1+2", "1 + 1.5 + 1.5 + 2", "2+2+2=6"), so it can be summed afterwards
# instead of only its last number. "each" = per sub-part, total not printed.
TAG_MARKS_RE = re.compile(r'(?<![A-Za-z\d.])(\d[\d.+\-=\s]*?)\s*(?:Marks?|M)\b\.?(\s+each\b)?',
                          re.IGNORECASE)

BLOOM_WORD_RE = re.compile(r'remember|recall|knowledge|understand|comprehen|appl|analy|evaluat|'
                           r'creat|synthes|level|\bL\s*[1-6]\b', re.IGNORECASE)

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


OCR_ZERO_FOR_O_RE = re.compile(r'C0(\d)')


def normalize_code(raw_code: str) -> str:
    """'C109.1' -> 'C109.1', 'CO 4' -> 'CO4', 'co4' -> 'CO4', 'CO-4' -> 'CO4',
    'C03' -> 'CO3' (OCR read the letter O as the digit 0)."""
    code = re.sub(r'[\s\-]+', '', raw_code).upper()
    m = OCR_ZERO_FOR_O_RE.fullmatch(code)
    return f"CO{m.group(1)}" if m else code


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


def _words(s: str) -> str:
    """The words and numbers of s in lower case, separated by one space."""
    return " ".join(re.findall(r'[a-z0-9]+', s.lower()))


def detect_subject(header_text: str, subject_yamls: dict, filename: str = "") -> str | None:
    # Whole words, not one squashed string: "Fundamentals-I" followed by any
    # word starting with "i" used to read as "...fundamentalsii" = SDF2.
    header_words = f" {_words(header_text)} "
    candidates = sorted(subject_yamls.items(),
                         key=lambda kv: len(normalize(kv[1]["subject_name"])),
                         reverse=True)
    for code, y in candidates:
        if f" {_words(y['subject_name'])} " in header_words:
            return code

    # header text didn't help — fall back to the filename's naming convention
    fname_norm = normalize(filename)
    for hint, code in sorted(FILENAME_SUBJECT_HINTS, key=lambda kv: -len(kv[0])):
        if hint in fname_norm and code in subject_yamls:
            return code
    return None


SESSION_RE = re.compile(r'(?<!\d)(20\d{2})\s*[-–/]\s*(?:20)?(\d{2})(?!\d)')    # "2023-24", "2023-2024"


def detect_year(header_text: str) -> int | None:
    """The year the exam was held.

    A year printed on its own ("ODD 2023", "May 2024") is the exam's year. The
    first "20xx" used to be taken, so "Session 2023-24 ... May 2024" gave 2023.
    If only a session is printed, an odd semester is held in its first year
    and an even one in its second."""
    sessions = [m for m in SESSION_RE.finditer(header_text)         # not a date like 2024-05-17
                if int(m.group(2)) == (int(m.group(1)) + 1) % 100]
    for m in re.finditer(r'(?<!\d)20\d{2}(?!\d)', header_text):
        if not any(s.start() <= m.start() < s.end() for s in sessions):
            return int(m.group(0))
    if sessions:
        first = int(sessions[0].group(1))
        return first if re.search(r'\bodd\b', header_text, re.I) else first + 1
    return None


def detect_exam_and_year(header_text: str, filename: str = "") -> tuple[str, int | None]:
    year = detect_year(header_text)

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


# The scanned papers carry the candidate's own handwritten details, and the
# vision model transcribes them along with the printed text, e.g.:
#   "Name Asha Rao Enrollment No. 21103045"
#   "Name: Asha (handwritten)"   /   "Enrollment No.: 2110 3045 (handwritten)"
#   "Name $\underline{\text{ Asha Rao }}$ Enrollment No. $\underline{...}$"
# A line only counts when it STARTS with the form label and a filled-in value
# follows, so question text ("Name the data structure ...", "... has name
# (string) and rollno (int)", "roll_no = 5;") is left alone.
_FIELD_START = r'^[\s>*_#|`•\-]*'                  # markdown / table decoration
_NAME_FIELD_RE = re.compile(
    _FIELD_START + r"(?:(?:student|candidate)(?:'?s)?\s+)?name"
    r"(?:\s+of\s+(?:the\s+)?(?:student|candidate))?\b(?P<sep>\s*[:\-–])?", re.I)
_ID_LABEL = (r"(?:(?:enrol+ment|roll|registration|regn?\.?)\s+(?:no|number|num|id)\b"
             r"|enrol+ment(?=\s*:))[\s.:#\-–]*")
_ID_FIELD_RE = re.compile(_FIELD_START + _ID_LABEL, re.I)
_ID_LABEL_RE = re.compile(r'\b' + _ID_LABEL, re.I)


def _filled_in(value: str, id_like: bool) -> bool:
    """True if a form field holds a real value: not blank and not prose."""
    latex = re.search(r'\\underline\s*\{(.*)\}', value)
    if latex:
        return bool(re.search(r'[A-Za-z0-9]', re.sub(r'\\[A-Za-z]+', '', latex.group(1))))
    words = re.sub(r'\([^)]*\)', ' ', value)             # "(handwritten)"
    words = re.sub(r'[*_`$"|]', ' ', words).split()
    if not 0 < len(words) <= 4:
        return False
    if id_like:                                          # "21103045", "2110 3045", "NJG132512"
        first = words[0]
        return bool(re.fullmatch(r'[A-Za-z0-9/\-]+', first)) and (
            any(ch.isdigit() for ch in first) or (first.isupper() and len(first) >= 5))
    return all(re.fullmatch(r"[A-Z][A-Za-z.'\-]*", w) for w in words)


def _is_candidate_field(line: str, in_header: bool) -> bool:
    id_first = _ID_FIELD_RE.match(line)                  # "Enrollment No. 21103045"
    if id_first:
        return _filled_in(line[id_first.end():], id_like=True)
    name = _NAME_FIELD_RE.match(line)
    if not name:
        return False
    rest = line[name.end():]
    if not (name.group("sep") or rest[:1].isspace()):    # "Name(String)"
        return False
    id_later = _ID_LABEL_RE.search(rest)                 # "Name ... Enrollment No. ..."
    if id_later and _filled_in(rest[id_later.end():], id_like=True):
        return True
    # A bare "Name Asha Rao" is only trusted above the first question;
    # further down the page it could be question text.
    value = rest[:id_later.start()] if id_later else rest
    return in_header and _filled_in(value, id_like=False)


def redact_pii(text: str) -> str:
    """Drop the lines holding the candidate's own name / enrollment number.
    Works on a header block and on a whole page of transcribed text."""
    first_question = QUESTION_START_RE.search(text)
    header_end = first_question.start() if first_question else len(text)
    kept, pos = [], 0
    for line in text.split("\n"):
        start, pos = pos, pos + len(line) + 1
        if not _is_candidate_field(line, in_header=start < header_end):
            kept.append(line)
    return "\n".join(kept)


def is_junk_page(text: str) -> bool:
    """True if a page's "transcription" is a classifier verdict or a refusal."""
    return bool(JUNK_PAGE_RE.match(text or ""))


def strip_model_chatter(page_text: str) -> str:
    """One page of transcribed text without what the model added of its own:
    an opening sentence about the transcription, and a list of the notes
    someone wrote on the paper by hand (the heading and the list under it).
    Everything printed on the paper is kept."""
    lines = (page_text or "").split("\n")
    kept, i = [], 0
    while i < len(lines):
        line = lines[i]
        opening = not any(l.strip() for l in kept) and CHATTER_LINE_RE.match(line)
        notes = HANDWRITTEN_HEADING_RE.match(line)
        if not (opening or notes):
            kept.append(line)
            i += 1
            continue
        i += 1
        if notes:                                # ... and the list that belongs to the heading
            while i < len(lines) and (not lines[i].strip() or _LIST_ITEM_RE.match(lines[i])):
                i += 1
        while i < len(lines) and (not lines[i].strip() or _RULE_RE.match(lines[i])):
            i += 1                               # the blank lines / "---" that set it apart
    return "\n".join(kept)


def drop_redundant_headings(text: str) -> str:
    """Remove a heading-only line ("### Q.1") when the next line starts the
    same question again. A heading that is NOT repeated ("### Q2." with the
    question text under it) is that question's only marker and stays: every
    such heading used to be removed, which lost the question."""
    def keep_or_drop(heading):
        repeated = QUESTION_START_RE.match(text[heading.end():].lstrip())
        return "" if repeated and repeated.group(1) == heading.group(1) else heading.group(0)
    return QUESTION_HEADING_ONLY_RE.sub(keep_or_drop, text)


def split_into_questions(full_text: str) -> tuple[str, list[str]]:
    """(header, [one block of text per question]).

    A marker whose number already started a question does not start another
    one: "Q1(a)" ... "Q1(b)" are parts of question 1, and a line of question
    text that happens to begin with "Q1 and Q2 are ..." stays where it is.
    So no two blocks ever get the same question number."""
    starts, seen = [], set()
    for m in QUESTION_START_RE.finditer(full_text):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            starts.append(m.start())
    if not starts:
        return full_text, []
    header = full_text[:starts[0]]
    blocks = []
    for start, end in zip(starts, starts[1:] + [len(full_text)]):
        block = full_text[start:end].strip()
        blocks.append(re.sub(r'^[ \t>#`]+', '', block))      # markdown in front of the marker
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


def parse_tag(inside: str, square: bool) -> dict | None:
    """Read what is between one pair of brackets as a [CO, Bloom, marks] tag.
    Returns {"raw_co_code", "bloom_level", "marks", "each"}, or None if this
    bracket is not a tag (a subscript, a function call, a remark ...)."""
    code = TAG_CODE_RE.search(inside)
    rest = inside[:code.start()] + " " + inside[code.end():] if code else inside
    marks = TAG_MARKS_RE.search(rest)       # looked for AFTER the code is out: "CO2 5 Marks" is 5, not 2 + 5
    if not marks:
        return None
    rest = rest[:marks.start()] + " " + rest[marks.end():]
    bloom = " ".join(re.sub(r'[,;:()\[\]]', ' ', rest).split())
    if code is None or not square:
        # Without a CO code, or in round brackets, it has to be nothing but a
        # tag: "(CO2 carries 5 marks in total)", "[Hint: worth 5 Marks]" and
        # "a[10 M]" are not tags.
        only_a_tag = (not bloom or BLOOM_WORD_RE.search(bloom)) and len(bloom.split()) <= 3
        spelled_out = re.search(r'marks?', marks.group(0), re.IGNORECASE)
        if not only_a_tag or (code is None and not (square and spelled_out)):
            return None
    each = bool(marks.group(2))
    return {"raw_co_code": normalize_code(code.group(0)) if code else None,
            "bloom_level": bloom or None,
            "marks": None if each else _resolve_marks_expr(marks.group(1)),
            "each": each}


def parse_marks_block(question_text: str, code_to_co: dict) -> dict:
    """
    Finds EVERY [CO..., Marks] tag in the block -- not just the first -- since
    a multi-part question (a)/(b)/(c) is often tagged with a separate CO+marks
    bracket per sub-part (e.g. 3M + 1.5M + 1.5M), and the original script only
    picked up the first one, silently dropping the rest of the question's marks.
    """
    kept = []                                   # (start, end, tag)
    for group in TAG_GROUP_RE.finditer(question_text):
        square = group.group(1) is not None
        tag = parse_tag(group.group(1) if square else group.group(2), square)
        if tag:
            kept.append((group.start(), group.end(), tag))

    if not kept:
        return {"raw_co_code": None, "co_short": None, "bloom_level": None, "marks": None,
                "marks_breakdown": None, "marks_each": False, "clean_text": question_text.strip()}

    tags = []
    total_marks = 0.0
    any_marks = False
    for start, end, tag in kept:
        if tag["marks"] is not None:
            total_marks += tag["marks"]
            any_marks = True
        tags.append({"raw_co_code": tag["raw_co_code"], "co_short": code_to_co.get(tag["raw_co_code"]),
                     "bloom_level": tag["bloom_level"], "marks": tag["marks"]})

    # Strip every matched tag span out of the text (not just the first).
    pieces, prev = [], 0
    for start, end, tag in kept:
        pieces.append(question_text[prev:start])
        prev = end
    pieces.append(question_text[prev:])
    clean_text = tidy_markdown_noise("".join(pieces))

    # "5 marks each" gives the marks of ONE sub-part; how many sub-parts there
    # are is not something to guess, so the total is left empty for a human.
    marks_each = any(tag["each"] for _, _, tag in kept)
    primary = next((t for t in tags if t["raw_co_code"]), tags[0])
    return {
        "raw_co_code": primary["raw_co_code"],
        "co_short": primary["co_short"],
        "bloom_level": primary["bloom_level"],
        "marks": _clean_marks(total_marks) if any_marks and not marks_each else None,
        # None for a simple single-tag question; the full per-sub-part
        # breakdown (each with its own CO/bloom/marks) when a question had
        # more than one tag, so multi-CO sub-parts stay auditable.
        "marks_breakdown": tags if len(tags) > 1 else None,
        "marks_each": marks_each,
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
        text = strip_model_chatter(r.get("text", ""))
        if len(text.strip()) < MIN_PAGE_TEXT_LEN or is_junk_page(text):
            skipped += 1
            continue
        good_rows.append({**r, "text": text})
    if skipped:
        print(f"  skipped {skipped} junk/empty page(s) in {pages_path.name}")
    rows = good_rows
    if not rows:
        print(f"  WARNING: no usable pages left in {pages_path.name} after filtering junk pages.")
        return []

    full_text = "\n".join(r["text"] for r in rows)
    full_text = drop_redundant_headings(full_text)
    # pages extracted on Windows hold "PYQs\T1 APS.pdf"; store one spelling
    source_file = rows[0]["source_file"].replace("\\", "/")

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
    marks_each = []

    for i, block in enumerate(question_blocks, start=1):
        q_num_match = QUESTION_START_RE.match(block)
        q_num = int(q_num_match.group(1)) if q_num_match else i
        parsed = parse_marks_block(block, code_to_co)

        if parsed["marks_each"]:
            marks_each.append(q_num)
        elif parsed["marks"] is None:
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
        if parsed["marks_each"]:
            note = 'marks are given per part ("... marks each"): fill in the total by hand'
            topic_info = {**topic_info, "needs_review": True,
                          "review_reason": "; ".join(filter(None, [topic_info["review_reason"], note]))}

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
    if marks_each:
        print(f"  NOTE: Q{', Q'.join(map(str, marks_each))}: the marks are given per part "
              f"(\"... marks each\"), so the total is left empty. Fill in \"marks\" by hand.")
    return records


# ------------------------------------------------------------- output files
def paper_source(pages_path: Path) -> str | None:
    """The PDF a pages file was extracted from, e.g. "PYQs/T1 Dsa.pdf"."""
    for line in pages_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)["source_file"].replace("\\", "/")
    return None


def is_manual(record: dict) -> bool:
    """A record a human has fixed (same rule as resolve_pyq_labels.py)."""
    meta = record["metadata"]
    return meta.get("tag_method") == "manual" or "manual" in (meta.get("review_reason") or "").lower()


def papers_already_split(out_dir: Path) -> dict:
    """{source PDF: its *_pyq.json}, read from the header record of each file."""
    done = {}
    for f in sorted(out_dir.glob("*_pyq.json")):
        try:
            source = json.loads(f.read_text(encoding="utf-8"))[0]["metadata"]["source"]
        except (ValueError, LookupError, TypeError):
            print(f"  WARNING: {f.name} is not a PYQ file this script wrote; left alone.")
            continue
        done[source.replace("\\", "/")] = f
    return done


def keep_manual_records(new_records: list[dict], old_records: list[dict]) -> list[dict]:
    """The re-split records, with every record a human fixed put back exactly
    as it was (text, labels, notes). A manual question that the new split no
    longer produces is kept too, so a re-split never loses one."""
    manual = {r["id"]: r for r in old_records if r["metadata"].get("type") == "question" and is_manual(r)}
    merged = [manual.pop(r["id"], r) for r in new_records]
    for lost in manual.values():
        print(f"  WARNING: {lost['id']} is a manual record, but this split has no question "
              f"with that id. It is kept; check the file.")
        merged.append(lost)
    return merged


def unused_paper_id(records: list[dict], taken: set) -> list[dict]:
    """Two different papers can share subject, year and exam (or have no year
    at all). The second one used to overwrite the first one's file. It now
    gets "_2" (then "_3" ...) in its file name and in every record id."""
    base = records[0]["id"].removesuffix("_header")
    if base not in taken:
        return records
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    print(f"  NOTE: another paper is already stored as {base}; this one is stored as {base}_{n}.")
    return [{**r, "id": f"{base}_{n}" + r["id"][len(base):]} for r in records]


def process_all(pages_dir=PAGES_DIR, out_dir=OUT_DIR, yaml_paths=None, redo_all=False):
    pages_dir, out_dir = Path(pages_dir), Path(out_dir)
    yaml_paths = yaml_paths or sorted(str(p) for p in SUBJECTS_DIR.glob("*.y*ml"))
    if not yaml_paths:
        raise SystemExit(f"No subject YAMLs found in {SUBJECTS_DIR}")
    subject_yamls = load_subject_yamls(yaml_paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = papers_already_split(out_dir)
    taken = {f.name.removesuffix("_pyq.json") for f in out_dir.glob("*_pyq.json")}
    for pages_file in sorted(pages_dir.glob("*_pages.jsonl")):
        out_path = done.get(paper_source(pages_file))
        if out_path and not redo_all:
            print(f"Skipping {pages_file.name}: already split -> {out_path.name} "
                  f"(--all splits it again)\n")
            continue

        print(f"Processing {pages_file.name}...")
        records = process_pages_file(pages_file, subject_yamls)
        if not records:
            continue
        if out_path:                     # --all: same file again, human fixes kept
            old_records = json.loads(out_path.read_text(encoding="utf-8"))
            base = old_records[0]["id"].removesuffix("_header")
            new_base = records[0]["id"].removesuffix("_header")
            records = [{**r, "id": base + r["id"][len(new_base):]} for r in records]
            records = keep_manual_records(records, old_records)
        else:                            # a new paper: never on top of another paper's file
            records = unused_paper_id(records, taken)
            out_path = out_dir / f"{records[0]['id'].removesuffix('_header')}_pyq.json"
            taken.add(out_path.name.removesuffix("_pyq.json"))
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {len(records)} records -> {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="page-level PYQ jsonl -> one record per question")
    ap.add_argument("--pages-dir", default=str(PAGES_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="where the *_pyq.json files go")
    ap.add_argument("--all", action="store_true",
                    help="split every paper again, also the ones that already have a file "
                         "(records marked manual are kept as they are)")
    args = ap.parse_args()
    process_all(args.pages_dir, args.out_dir, redo_all=args.all)