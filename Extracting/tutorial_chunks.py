"""
Everything -- ingestion, diagram transcription, chunking, topic mapping and
output verification -- lives in this single file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pymupdf  # PyMuPDF
import yaml
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAX_CHUNK_WORDS = 300          # target words per composite chunk (spec)
MIN_IMAGE_DIMENSION = 80       # px -- filters icons/bullets/logos, keeps diagrams
INDENT_TOLERANCE = 8.0         # pt -- x-offset above which a line counts as indented

# Model the spec asks for. Kept as the primary; the client falls back when the
# API reports it is unavailable for the key in use.
DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = (
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-pro",
)

DIAGRAM_TRANSCRIPTION_PROMPT = (
    "This image is a figure from a computer-science tutorial / assignment sheet "
    "(for example a graph, tree, binary search tree, heap, hash table, matrix, "
    "linked-list diagram, or a state-space / automaton diagram).\n"
    "Transcribe its structure precisely and exhaustively as plain text so that a "
    "student could reconstruct the figure without ever seeing the image:\n"
    "  - list every node / vertex / cell and its label or value;\n"
    "  - list every edge / pointer / link, stating its direction (and weight if "
    "shown);\n"
    "  - describe any table, matrix, or array as rows and columns with values;\n"
    "  - include every visible text label, key, legend entry and annotation.\n"
    "Prefer a structured form (adjacency list, parent -> children mapping, "
    "key/value index, or a row/column table) over loose prose. Do not add "
    "commentary, explanation, or interpretation beyond the structure itself."
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG = logging.getLogger("ingest_tutorials")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #
# Strong marker: "Q1.", "Q.1.", "Q2)", "Q 3.", "Q1 )", "Question 2:", "Q.1.The"
STRONG_MARKER_RE = re.compile(
    r"^\s*Q(?:uestion)?\s*\.?\s*(\d{1,3})\s*[\).:\-]?\s*", re.IGNORECASE
)
# Weak marker: a bare "1)", "2." at the very start of a line (used by sheets
# that number questions without a "Q" prefix, e.g. DS_TUT_9).
WEAK_MARKER_RE = re.compile(r"^\s*(\d{1,3})\s*[\).]\s*")

# Fenced code block: ``` ... ``` (kept atomic -- never split across chunks).
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)

# A standalone "Solution" / "Answer" heading starts the worked-solutions block.
SOLUTION_HEADING_RE = re.compile(r"^(?:solutions?|answers?)\s*:?\s*$", re.IGNORECASE)

# Page-number / footer shapes.
PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?$", re.IGNORECASE)

# Lines that are always boilerplate regardless of position (belt-and-braces;
# most are already removed by the "drop everything before Q1" rule).
HEADER_LINE_RES = [
    re.compile(r"^tutorial\s+(?:and\s+assignment\s+)?sheet\b", re.IGNORECASE),
    re.compile(r"^tutorial\s*[–\-:]?\s*\d", re.IGNORECASE),
    re.compile(r"^tutorial\s*[–\-]?\s*\d+\s*[-–]\s*\d+", re.IGNORECASE),
    re.compile(r"^week\s*\d", re.IGNORECASE),
    re.compile(r"^topics?\s*:", re.IGNORECASE),
    re.compile(r"^instructions?\s*$", re.IGNORECASE),
    re.compile(r"^15B11CI\d{3}\b"),                       # course code
    re.compile(r"^algorithms and problem solving\b", re.IGNORECASE),
    re.compile(r"^data structures\b", re.IGNORECASE),
]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    """One physical text line with its page and top-left coordinate."""
    page: int
    x0: float
    y0: float
    y1: float
    text: str


@dataclass
class ImageRef:
    """An embedded image, located on the page so it can be tied to a question."""
    page: int
    x0: float
    y0: float
    path: Path
    sha1: str


@dataclass
class Unit:
    """An indivisible question/exercise block (or a lead-in prose block)."""
    text: str
    is_question: bool
    page: int
    y0: float
    images: list = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def has_code(self) -> bool:
        return bool(CODE_BLOCK_RE.search(self.text))


@dataclass
class Chunk:
    chunk_id: str
    description: str
    topics: list = field(default_factory=list)
    topic_ids: list = field(default_factory=list)
    cos: list = field(default_factory=list)          # unique CO labels the chunk covers
    co_details: list = field(default_factory=list)   # [{id, label, subject, text, source, via}]


# --------------------------------------------------------------------------- #
# PDF text extraction (with coordinates)
# --------------------------------------------------------------------------- #
def extract_lines(doc: "pymupdf.Document") -> list:
    """Return every non-empty text line of the PDF as a ``Line`` with its
    page index and bounding-box coordinates (needed for indentation-aware
    question detection and for associating figures with questions)."""
    lines: list = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(span["text"] for span in spans).rstrip()
                if not text.strip():
                    continue
                x0 = min(span["bbox"][0] for span in spans)
                y0 = min(span["bbox"][1] for span in spans)
                y1 = max(span["bbox"][3] for span in spans)
                lines.append(Line(page_index, x0, y0, y1, text))
    return lines


def body_left_margin(lines: list) -> float:
    """The left edge of body text = smallest x0 seen. Headers/titles are
    centred (larger x0), so the minimum reliably lands on the question text."""
    if not lines:
        return 0.0
    return min(line.x0 for line in lines)


def is_boilerplate_line(line: Line, page_height: float, running: set) -> bool:
    """True if a line is a header, footer, page number, or a line that repeats
    on every page (running header/footer)."""
    text = line.text.strip()
    if not text:
        return True
    if text in running:
        return True
    for pattern in HEADER_LINE_RES:
        if pattern.match(text):
            return True
    # Pure page numbers only when they sit in the top/bottom margin band.
    if PAGE_NUMBER_RE.match(text):
        if page_height and (line.y0 < page_height * 0.06 or line.y1 > page_height * 0.94):
            return True
    return False


def find_running_lines(lines: list, page_count: int) -> set:
    """Detect lines that repeat near the top or bottom of most pages (running
    headers/footers) so they can be stripped wherever they appear."""
    if page_count < 2:
        return set()
    counts: dict = {}
    for line in lines:
        key = re.sub(r"\s+", " ", line.text.strip()).casefold()
        if not key:
            continue
        counts.setdefault(key, set()).add(line.page)
    threshold = max(2, int(page_count * 0.5))
    return {key for key, pages in counts.items() if len(pages) >= threshold}


# --------------------------------------------------------------------------- #
# Question detection
# --------------------------------------------------------------------------- #
def detect_question_markers(lines: list, body_left: float) -> list:
    """Return the indices of lines that begin a new top-level question.

    Two marker shapes are recognised:
      * strong  -- a "Q"/"Question" prefix (unambiguous);
      * weak    -- a bare "N)" / "N." at the *body left margin* (not indented),
                   used by sheets that number questions without a "Q".

    A monotonic guard rejects out-of-order numbers: a marker is only accepted
    when its number is greater than the last accepted one. This cleanly drops
    false positives created by line wrapping, e.g. the continuation
    "... (Fig. 1 of\nQ1). Write a program ..." whose second line looks like a
    marker but re-uses an already-seen number.
    """
    markers: list = []
    last_number = 0
    for idx, line in enumerate(lines):
        text = line.text
        strong = STRONG_MARKER_RE.match(text)
        number: Optional[int] = None
        if strong:
            number = int(strong.group(1))
        else:
            weak = WEAK_MARKER_RE.match(text)
            if weak and line.x0 <= body_left + INDENT_TOLERANCE:
                number = int(weak.group(1))
        if number is None:
            continue
        if number > last_number:
            markers.append(idx)
            last_number = number
    return markers


def build_units(lines: list, markers: list) -> list:
    """Turn marker positions into ``Unit`` objects. Everything *before* the
    first marker is dropped (headers/titles/topics/instructions)."""
    if not markers:
        return []

    units: list = []
    for i, start in enumerate(markers):
        end = markers[i + 1] if i + 1 < len(markers) else len(lines)
        block = lines[start:end]
        text = "\n".join(line.text.strip() for line in block)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            continue
        units.append(Unit(
            text=text,
            is_question=True,
            page=block[0].page,
            y0=block[0].y0,
        ))
    return units


def strip_solution_sections(lines: list) -> list:
    """Drop everything from the first standalone "Solution"/"Answer" heading
    onward -- worked answers are not question content."""
    for idx, line in enumerate(lines):
        if SOLUTION_HEADING_RE.match(line.text.strip()):
            LOG.debug("  solution section detected at line %d -> truncating", idx)
            return lines[:idx]
    return lines


# --------------------------------------------------------------------------- #
# Image extraction & association
# --------------------------------------------------------------------------- #
def extract_images(doc: "pymupdf.Document", subject: str, pdf_stem: str,
                   image_root: Path) -> list:
    """Extract every embedded figure (>= MIN_IMAGE_DIMENSION px) and return
    ``ImageRef`` objects carrying the page + position used for association."""
    image_root.mkdir(parents=True, exist_ok=True)
    refs: list = []
    seen_xref: dict = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref")
            if xref is None or xref == 0:
                continue
            if info.get("width", 0) < MIN_IMAGE_DIMENSION or info.get("height", 0) < MIN_IMAGE_DIMENSION:
                continue
            if xref in seen_xref:
                path, sha1 = seen_xref[xref]
            else:
                try:
                    raw = doc.extract_image(xref)
                except Exception as exc:  # pragma: no cover - defensive
                    LOG.warning("  could not extract image xref=%s: %s", xref, exc)
                    continue
                data = raw["image"]
                ext = raw.get("ext", "png")
                path = image_root / f"{subject}_{pdf_stem}_p{page_index}_x{xref}.{ext}"
                path.write_bytes(data)
                sha1 = hashlib.sha1(data).hexdigest()
                seen_xref[xref] = (path, sha1)
            bbox = info["bbox"]
            refs.append(ImageRef(page=page_index, x0=bbox[0], y0=bbox[1],
                                 path=path, sha1=sha1))
    return refs


def associate_images_to_units(units: list, images: list) -> None:
    """Attach each figure to the question it belongs to: the last question that
    starts at or before the figure's (page, y) position. Mutates ``units``."""
    if not units or not images:
        return
    ordered = sorted(units, key=lambda u: (u.page, u.y0))
    for img in sorted(images, key=lambda i: (i.page, i.y0)):
        chosen = None
        for unit in ordered:
            if (unit.page, unit.y0) <= (img.page, img.y0):
                chosen = unit
            else:
                break
        if chosen is None:
            chosen = ordered[0]
        chosen.images.append(img)


# --------------------------------------------------------------------------- #
# Chunking (atomic questions, look-ahead word guard)
# --------------------------------------------------------------------------- #
def pack_units_into_chunks(units: list) -> list:
    """Greedily pack atomic question units into composite chunks.

    Rules (per spec):
      * the running word count is checked *before* adding the next question;
      * if adding it would exceed ``MAX_CHUNK_WORDS`` the current chunk is
        closed and the whole question moves to a fresh chunk;
      * a question is never split (so a code block inside it never splits);
      * a single question already over the cap becomes its own standalone chunk.
    """
    chunks: list = []
    current: list = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(current)
        current, current_words = [], 0

    for unit in units:
        if unit.word_count > MAX_CHUNK_WORDS:
            flush()
            chunks.append([unit])          # soft-cap exception
            continue
        if current and current_words + unit.word_count > MAX_CHUNK_WORDS:
            flush()
        current.append(unit)
        current_words += unit.word_count

    flush()
    return chunks


# --------------------------------------------------------------------------- #
# Gemini client (with model fallback, retry, and on-disk cache)
# --------------------------------------------------------------------------- #
class GeminiError(RuntimeError):
    pass


def _is_unavailable(err: Exception) -> bool:
    msg = str(err)
    return "NOT_FOUND" in msg or "no longer available" in msg or "not found" in msg.lower()


def _is_quota(err: Exception) -> bool:
    msg = str(err)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


def _is_transient(err: Exception) -> bool:
    msg = str(err)
    return "503" in msg or "UNAVAILABLE" in msg or "500" in msg or "deadline" in msg.lower()


def _retry_delay(err: Exception) -> Optional[float]:
    """Pull the server-suggested retry delay (seconds) out of a 429 message."""
    match = re.search(r"retry(?:Delay| in)\D*([\d.]+)\s*s", str(err), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            return None
    return None


class GeminiClient:
    """Thin wrapper around the official google-genai SDK.

    * loads ``GEMINI_API_KEY`` from a ``.env`` file (python-dotenv);
    * transparently falls back through ``FALLBACK_MODELS`` when the primary
      model has been retired for the key in use;
    * is rate-limit aware: on a 429 it cools the offending model down for the
      server-suggested delay and rotates to another model, so the free-tier
      "N requests / minute / model" ceiling is spread across models instead of
      hammering one;
    * caches every response on disk so re-runs cost nothing.
    """

    def __init__(self, api_key: str, primary_model: str = DEFAULT_MODEL,
                 fallback_models: Iterable = FALLBACK_MODELS,
                 cache_path: Optional[Path] = None,
                 request_interval: float = 0.0, max_attempts: int = 8):
        self._api_key = api_key
        self._models = [primary_model] + [m for m in fallback_models if m != primary_model]
        self._client = None
        self._cache_path = cache_path
        self._cache: dict = {}
        self._cooldown: dict = {m: 0.0 for m in self._models}
        self._request_interval = request_interval
        self._max_attempts = max_attempts
        self._last_call = 0.0
        self._load_cache()

    # -- cache ------------------------------------------------------------- #
    def _load_cache(self) -> None:
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _key(*parts: str) -> str:
        return hashlib.sha1("\u241f".join(parts).encode("utf-8")).hexdigest()

    # -- client ------------------------------------------------------------ #
    def _get_client(self):
        if self._client is None:
            from google import genai  # lazy import so --no-llm works without it
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _pick_model(self) -> str:
        """Least-recently-cooled available model (spreads load across models)."""
        now = time.time()
        available = [m for m in self._models if self._cooldown.get(m, 0.0) <= now]
        if not available:
            wait = min(self._cooldown.values()) - now
            wait = max(0.5, min(wait, 90.0))
            LOG.info("  all models rate-limited; waiting %.0fs", wait)
            time.sleep(wait)
            return self._pick_model()
        available.sort(key=lambda m: self._cooldown.get(m, 0.0))
        return available[0]

    def _generate_raw(self, contents, json_mode: bool) -> str:
        from google.genai import types
        client = self._get_client()
        config = types.GenerateContentConfig(temperature=0.0)
        if json_mode:
            config.response_mime_type = "application/json"

        last_err: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            model = self._pick_model()
            gap = time.time() - self._last_call
            if gap < self._request_interval:
                time.sleep(self._request_interval - gap)
            try:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config,
                )
                self._last_call = time.time()
                return (response.text or "").strip()
            except Exception as exc:  # noqa: BLE001 - we branch on the message
                self._last_call = time.time()
                last_err = exc
                if _is_unavailable(exc):
                    LOG.warning("  model '%s' unavailable; disabling it", model)
                    self._cooldown[model] = time.time() + 86400
                    continue
                if _is_quota(exc):
                    delay = _retry_delay(exc) or 60.0
                    self._cooldown[model] = time.time() + delay
                    LOG.warning("  model '%s' rate-limited; cooling %.0fs (attempt %d)",
                                model, delay, attempt)
                    continue
                if _is_transient(exc):
                    self._cooldown[model] = time.time() + 15.0
                    LOG.warning("  model '%s' transient error; retrying (attempt %d)",
                                model, attempt)
                    continue
                raise GeminiError(str(exc)) from exc
        raise GeminiError(f"all models failed after {self._max_attempts} attempts: {last_err}")

    # -- public API -------------------------------------------------------- #
    def transcribe_diagram(self, image_path: Path) -> str:
        data = image_path.read_bytes()
        key = self._key("diagram", hashlib.sha1(data).hexdigest())
        if key in self._cache:
            return self._cache[key]

        from google.genai import types
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        contents = [
            types.Part.from_bytes(data=data, mime_type=mime),
            DIAGRAM_TRANSCRIPTION_PROMPT,
        ]
        text = self._generate_raw(contents, json_mode=False)
        if not text:
            text = f"[Diagram transcription returned no text for {image_path.name}]"
        self._cache[key] = text
        self._save_cache()
        return text

    def classify_topics(self, subject_name: str, chunk_text: str,
                        topics: list) -> list:
        """Return a list of ``{"topic": name, "topic_id": id}`` for the chunk."""
        catalog = "\n".join(f"- id: {t['id']} | name: {t['name']}" for t in topics)
        allowed_ids = {t["id"] for t in topics}
        key = self._key("topics", subject_name, chunk_text, catalog)
        if key in self._cache:
            cached = self._cache[key]
            return [t for t in cached if t.get("topic_id") in allowed_ids]

        prompt = (
            f"You are tagging tutorial questions for the subject \"{subject_name}\".\n"
            "Below is a chunk of tutorial content (one or more questions, possibly "
            "with diagram descriptions). Choose the syllabus topic(s) that the chunk "
            "covers. A chunk may contain several short questions on DIFFERENT topics, "
            "so return EVERY topic that is genuinely covered -- but never a topic that "
            "the content does not support.\n\n"
            "You MUST choose from this exact list (use the exact id and exact name):\n"
            f"{catalog}\n\n"
            "Return ONLY JSON in this shape:\n"
            '{"topics": [{"topic": "<exact name>", "topic_id": "<exact id>"}]}\n'
            "If nothing in the list applies, return {\"topics\": []}.\n\n"
            "CHUNK:\n"
            f"{chunk_text}"
        )
        raw = self._generate_raw(prompt, json_mode=True)
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise GeminiError(f"topic classification returned non-JSON: {raw[:120]!r}") from exc
        result = []
        for item in parsed.get("topics", []):
            if not isinstance(item, dict):
                continue
            tid = item.get("topic_id")
            if tid in allowed_ids:
                name = next((t["name"] for t in topics if t["id"] == tid), item.get("topic"))
                result.append({"topic": name, "topic_id": tid})
        # de-duplicate, preserving order
        seen, unique = set(), []
        for item in result:
            if item["topic_id"] not in seen:
                seen.add(item["topic_id"])
                unique.append(item)
        self._cache[key] = unique
        self._save_cache()
        return unique


# --------------------------------------------------------------------------- #
# Keyword fallback classifier (used with --no-llm or when the LLM fails)
# --------------------------------------------------------------------------- #
_GENERIC_WORDS = {
    "data", "structure", "structures", "algorithm", "algorithms", "introduction",
    "intro", "advanced", "basic", "topic", "topics", "implementation", "concept",
    "concepts", "and", "the", "for", "with", "from", "using", "review",
}


def keyword_classify(chunk_text: str, topics: list) -> list:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()

    haystack = f" {norm(chunk_text)} "
    matches = []
    for topic in topics:
        aliases = set()
        for term in re.split(r"[,():/&+\-]+", topic["name"]):
            nt = norm(term)
            if nt:
                aliases.add(nt)
            for word in nt.split():
                if len(word) >= 4 and word not in _GENERIC_WORDS:
                    aliases.add(word)
        if any(f" {a} " in haystack for a in aliases):
            matches.append({"topic": topic["name"], "topic_id": topic["id"]})
    return matches


# --------------------------------------------------------------------------- #
# Chunk description assembly (inline diagram insertion)
# --------------------------------------------------------------------------- #
def build_description(chunk_units: list, transcriber) -> str:
    """Join the questions of a chunk, inserting each figure's description
    inline -- immediately after the question it belongs to -- wrapped in the
    ``[Diagram Context / Description: ...]`` marker."""
    parts = []
    for unit in chunk_units:
        parts.append(unit.text)
        for image in unit.images:
            description = transcriber(image.path)
            parts.append(f"[Diagram Context / Description: {description}]")
    return "\n".join(parts).strip()


# --------------------------------------------------------------------------- #
# Per-PDF processing
# --------------------------------------------------------------------------- #
def resolve_subject(pdf_path: Path, tutorials_root: Path) -> str:
    """Subject = immediate parent folder (``Tutorials/<Subject>/*.pdf``). A PDF
    sitting directly in the root falls back to its own filename stem."""
    parent = pdf_path.parent.resolve()
    if parent == tutorials_root.resolve():
        return pdf_path.stem
    return parent.name


def process_pdf(pdf_path: Path, subject: str, tutorials_root: Path,
                image_root: Path, keep_solutions: bool) -> list:
    doc = pymupdf.open(pdf_path)
    try:
        page_height = doc[0].rect.height if len(doc) else 0.0
        lines = extract_lines(doc)
        if not lines:
            LOG.warning("  %s: no text layer (scanned PDF?) -- skipping", pdf_path.name)
            return []

        running = find_running_lines(lines, len(doc))
        if running:
            LOG.debug("  running header/footer lines: %s", sorted(running))

        # Drop boilerplate lines, then (optionally) the solutions block.
        cleaned = [ln for ln in lines if not is_boilerplate_line(ln, page_height, running)]
        if not keep_solutions:
            cleaned = strip_solution_sections(cleaned)

        body_left = body_left_margin(cleaned)
        markers = detect_question_markers(cleaned, body_left)
        units = build_units(cleaned, markers)

        images = extract_images(doc, subject, pdf_path.stem, image_root / subject)
        associate_images_to_units(units, images)

        n_images = sum(len(u.images) for u in units)
        LOG.info("  %s: %d question(s), %d figure(s)", pdf_path.name, len(units), n_images)
        return units
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# Topic-map loading
# --------------------------------------------------------------------------- #
# Built-in fallback catalog, used when topic_map.yaml has no `subjects:` block,
# so the pipeline still runs against the standard curriculum. Keep the ids in
# sync with prerequisite_graph/subjects/{ds,algo}.yaml.
FALLBACK_SUBJECT_CATALOG = {
    "DS": {
        "yaml_subject": "DS",
        "subject_name": "Data Structures",
        "topics": [
            {"id": "DS_01", "name": "Linear Data Structures (Array, Linked List, Stack, Queue)"},
            {"id": "DS_02", "name": "Searching and Sorting"},
            {"id": "DS_03", "name": "Non-Linear DS: Multi List, Tree, Priority Queue (Heaps)"},
            {"id": "DS_04", "name": "Non-Linear DS: BST, AVL, RB Tree, B/B+ Tree"},
            {"id": "DS_05", "name": "Non-Linear DS: Graphs"},
            {"id": "DS_06", "name": "Advanced DS: Segment/Interval Tree, Suffix Tree/Array, Tries"},
            {"id": "DS_07", "name": "Hashing (Hash Tables, Collision Resolution)"},
        ],
    },
    "APS": {
        "yaml_subject": "ALGO",
        "subject_name": "Algorithms and Problem Solving",
        "topics": [
            {"id": "ALGO_01", "name": "Introduction: Asymptotic Analysis, Sorting/Searching Review"},
            {"id": "ALGO_02", "name": "Search Trees & Priority Queue (Segment/Interval/RB Tree, Binomial/Fibonacci Heap)"},
            {"id": "ALGO_03", "name": "Divide and Conquer"},
            {"id": "ALGO_04", "name": "Greedy Algorithms"},
            {"id": "ALGO_05", "name": "Backtracking Algorithms"},
            {"id": "ALGO_06", "name": "Dynamic Programming"},
            {"id": "ALGO_07", "name": "String Algorithms"},
            {"id": "ALGO_08", "name": "Problem Spaces and Search (BFS/DFS/A*)"},
            {"id": "ALGO_09", "name": "Tractable and Non-Tractable Problems (P, NP, NP-Complete)"},
        ],
    },
}


def load_topic_catalog(topic_map_path: Path, subjects_dir: Optional[Path]) -> dict:
    """Return ``{subject_folder: {"name": ..., "topics": [{"id","name"}, ...]}}``
    from the ``subjects:`` block of ``topic_map.yaml``. Cross-checks each
    subject's topic_ids against ``prerequisite_graph/subjects/*.yaml`` and warns
    on drift (the graph is the source of truth downstream)."""
    config = yaml.safe_load(topic_map_path.read_text(encoding="utf-8")) or {}
    subjects_block = config.get("subjects")
    if not subjects_block:
        LOG.warning(
            "'%s' has no 'subjects:' block -- falling back to the built-in catalog "
            "(DS + APS). Add a 'subjects:' block to customise subject/topic mapping.",
            topic_map_path,
        )
        subjects_block = FALLBACK_SUBJECT_CATALOG

    graph_ids: dict = {}
    if subjects_dir and subjects_dir.exists():
        for yml in sorted(subjects_dir.glob("*.y*ml")):
            data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            graph_ids[data.get("subject")] = {t["id"] for t in data.get("topics", [])}

    catalog: dict = {}
    for folder, block in subjects_block.items():
        topics = [{"id": t["id"], "name": t["name"]} for t in block.get("topics", [])]
        catalog[folder] = {"name": block.get("subject_name", folder),
                           "yaml_subject": block.get("yaml_subject"),
                           "topics": topics}
        yaml_subject = block.get("yaml_subject")
        if yaml_subject and yaml_subject in graph_ids:
            unknown = {t["id"] for t in topics} - graph_ids[yaml_subject]
            if unknown:
                LOG.warning("  topic_map subject '%s' lists ids not in %s.yaml: %s",
                            folder, yaml_subject, sorted(unknown))
    return catalog


# --------------------------------------------------------------------------- #
# Course outcomes (COs) + prerequisites
# --------------------------------------------------------------------------- #
def load_curriculum_graph(subjects_dir: Optional[Path]) -> dict:
    """Load course outcomes, the topic->CO map and the prerequisite edges from
    ``prerequisite_graph/subjects/*.yaml`` (the curriculum graph, which is the
    source of truth downstream).

    Returns a dict with four lookups::

        co_text       : {(subject, co_id): outcome text}
        topic_co      : {topic_id: [co_id, ...]}
        topic_subject : {topic_id: subject}
        prereqs       : {topic_id: [(requires_id, strength), ...]}
    """
    co_text: dict = {}
    topic_co: dict = {}
    topic_subject: dict = {}
    prereqs: dict = {}
    if subjects_dir and subjects_dir.exists():
        for yml in sorted(subjects_dir.glob("*.y*ml")):
            data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            subject = data.get("subject")
            for co_id, text in (data.get("course_outcomes") or {}).items():
                co_text[(subject, co_id)] = text
            for topic in data.get("topics", []):
                tid = topic.get("id")
                if not tid:
                    continue
                topic_subject[tid] = subject
                topic_co[tid] = list(topic.get("co") or [])
            for edge in data.get("prerequisites", []):
                tid, req = edge.get("topic"), edge.get("requires")
                if tid and req:
                    prereqs.setdefault(tid, []).append((req, edge.get("strength", "hard")))
    return {"co_text": co_text, "topic_co": topic_co,
            "topic_subject": topic_subject, "prereqs": prereqs}


def collect_cos(topic_ids: list, graph: dict, depth: int = 1,
                own_subject: Optional[str] = None,
                include_cross_subject: bool = False) -> tuple:
    """Resolve the course outcomes (COs) a chunk touches.

    Starts from the chunk's own ``topic_ids`` (each contributes its ``co`` list)
    and then walks the prerequisite edges up to ``depth`` levels
    (``depth < 0`` = full transitive closure; ``depth == 0`` = own topics only).
    COs are de-duplicated by ``(subject, co_id)`` and returned in discovery
    order: own-topic COs first, then prerequisite COs breadth-first.

    Because prerequisites cross subjects (e.g. ``DS_01`` requires ``SDF1_03``)
    and CO ids are only unique *within* a subject (DS ``CO1`` != SDF1 ``CO1``),
    the ``own_subject`` / ``include_cross_subject`` pair controls scope:

    * ``include_cross_subject=False`` (default) -- keep only COs whose subject
      matches ``own_subject``, so ``cos`` stays a clean list of that subject's
      own CO ids (``CO1``..``CO5``).
    * ``include_cross_subject=True`` -- keep COs from every subject reached;
      ``cos`` entries are then qualified as ``SUBJECT:CO`` whenever a plain id
      would be ambiguous.

    Returns ``(cos, co_details)`` where ``cos`` is the ordered list of unique CO
    labels and ``co_details`` is the matching list of dicts
    ``{id, label, subject, text, source, via}`` (``source`` is ``"topic"`` or
    ``"prerequisite"``; ``via`` is the topic id that pulled the CO in).
    """
    co_text = graph["co_text"]
    topic_co = graph["topic_co"]
    topic_subject = graph["topic_subject"]
    prereqs = graph["prereqs"]

    ordered: list = []
    seen: set = set()

    def add(topic_id: str, source: str) -> None:
        subject = topic_subject.get(topic_id)
        if own_subject and not include_cross_subject and subject != own_subject:
            return
        for co_id in topic_co.get(topic_id, []):
            key = (subject, co_id)
            if key in seen:
                continue
            seen.add(key)
            ordered.append({
                "id": co_id,
                "subject": subject,
                "text": co_text.get(key, ""),
                "source": source,
                "via": topic_id,
            })

    for tid in topic_ids:
        add(tid, "topic")

    if depth != 0:
        visited = set(topic_ids)
        frontier = list(topic_ids)
        level = 0
        while frontier and (depth < 0 or level < depth):
            level += 1
            nxt: list = []
            for tid in frontier:
                for req_id, _strength in prereqs.get(tid, []):
                    if req_id in visited:
                        continue
                    visited.add(req_id)
                    add(req_id, "prerequisite")
                    nxt.append(req_id)
            frontier = nxt

    # Qualify the label with the subject only when a plain id is ambiguous.
    id_counts: dict = {}
    for entry in ordered:
        id_counts[entry["id"]] = id_counts.get(entry["id"], 0) + 1
    for entry in ordered:
        entry["label"] = (f'{entry["subject"]}:{entry["id"]}'
                          if id_counts[entry["id"]] > 1 else entry["id"])

    cos = [entry["label"] for entry in ordered]
    return cos, ordered


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def find_api_key(env_paths: Iterable[Path]) -> Optional[str]:
    for path in env_paths:
        if path and path.exists():
            load_dotenv(path, override=False)
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def write_subject_jsonl(subject: str, chunks: list, out_root: Path) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", subject).strip("_")
    out_path = out_root / f"{slug}_tutorial_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "description": chunk.description,
                "topics": chunk.topics,
                "topic_ids": chunk.topic_ids,
                "cos": chunk.cos,
                "co_details": chunk.co_details,
            }, ensure_ascii=False) + "\n")
    return out_path


# --------------------------------------------------------------------------- #
# Output verification (merged in from the standalone verifier so the whole
# pipeline ships as a single file:  python ingest_tutorials.py --verify)
# --------------------------------------------------------------------------- #
REQUIRED_KEYS = {"chunk_id", "description", "topics", "topic_ids", "cos", "co_details"}


def verify_outputs(out_dir: str) -> int:
    """Sanity-check every ``*_tutorial_chunks.jsonl`` under ``out_dir``.

    Validates each line against the required minimal schema and reports
    chunk/word/topic statistics. Returns 0 when clean, 1 when any line is
    malformed (or when no output files are found).
    """
    root = Path(out_dir)
    files = sorted(root.glob("*_tutorial_chunks.jsonl"))
    if not files:
        print(f"no *_tutorial_chunks.jsonl found under {root}")
        return 1

    problems = 0
    grand_chunks = 0
    for path in files:
        n = 0
        words = 0
        no_topic = 0
        no_co = 0
        diagram_chunks = 0
        ids = set()
        co_ids = set()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  {path.name}:{lineno} invalid JSON: {exc}")
                problems += 1
                continue
            if set(obj.keys()) != REQUIRED_KEYS:
                print(f"  {path.name}:{lineno} wrong keys: {sorted(obj.keys())}")
                problems += 1
            if not isinstance(obj.get("topics"), list) or not isinstance(obj.get("topic_ids"), list):
                print(f"  {path.name}:{lineno} topics/topic_ids must be lists")
                problems += 1
            if len(obj.get("topics", [])) != len(obj.get("topic_ids", [])):
                print(f"  {path.name}:{lineno} topics/topic_ids length mismatch")
                problems += 1
            if not isinstance(obj.get("cos"), list) or not isinstance(obj.get("co_details"), list):
                print(f"  {path.name}:{lineno} cos/co_details must be lists")
                problems += 1
            elif [d.get("label", d.get("id")) for d in obj.get("co_details", [])] != obj.get("cos", []):
                print(f"  {path.name}:{lineno} 'cos' does not match co_details labels")
                problems += 1
            if not obj.get("chunk_id", "").startswith(path.name.split("_")[0]):
                print(f"  {path.name}:{lineno} chunk_id prefix mismatch: {obj.get('chunk_id')}")
                problems += 1
            n += 1
            words += len(obj.get("description", "").split())
            if "[Diagram Context / Description:" in obj.get("description", ""):
                diagram_chunks += 1
            if not obj.get("topic_ids"):
                no_topic += 1
            if not obj.get("cos"):
                no_co += 1
            ids.update(obj.get("topic_ids", []))
            co_ids.update(obj.get("cos", []))
        grand_chunks += n
        avg = (words // n) if n else 0
        print(f"{path.name}: {n} chunks | avg {avg} words | {diagram_chunks} with diagrams "
              f"| {no_topic} untagged | {no_co} without COs "
              f"| topic_ids={sorted(ids)} | cos={sorted(co_ids)}")

    print(f"\nTotal: {grand_chunks} chunks across {len(files)} file(s); "
          f"{problems} problem(s).")
    return 1 if problems else 0


def find_repo_root(start: Path) -> Path:
    """Locate the ``Question_Paper_Generator`` project root.

    The root is the folder that contains the ``Extracting/`` and/or
    ``prerequisite_graph/`` directories. We walk *up* from the script's own
    location, so the script works whether it is placed at the repo root
    (``Question_Paper_Generator/ingest_tutorials.py``) or inside ``Extracting/``
    (``Question_Paper_Generator/Extracting/ingest_tutorials.py``).

    This is the fix for the "path not found" bug: previously ``repo_root`` was
    hard-coded as ``script_dir.parent``, which is only correct when the script
    sits at the repo root. When it sat in ``Extracting/`` the derived paths
    became ``Extracting/Extracting/topic_map.yaml`` etc. and nothing resolved.
    """
    markers = ("Extracting", "prerequisite_graph")
    for base in [start, *start.parents]:
        if any((base / m).is_dir() for m in markers):
            return base
    # Last resort: assume the script sits at the repo root.
    return start


def project_paths(script_dir: Path) -> dict:
    """Return every canonical project path, derived from the detected root.

    Layout (script lives in ``Extracting/``)::

        Question_Paper_Generator/          <- repo_root
        |-- Tutorials/<Subject>/*.pdf      <- input  (repo_root/Tutorials)
        `-- Extracting/
            |-- ingest_tutorials.py        <- this file (script_dir)
            |-- .env                       <- GEMINI_API_KEY
            |-- topic_map.yaml
            |-- tutorial_images/
            `-- Processed/                 <- output (script_dir/Processed)
    """
    repo_root = find_repo_root(script_dir)
    extracting = script_dir if (script_dir / "topic_map.yaml").exists() else repo_root / "Extracting"
    return {
        "repo_root": repo_root,
        "extracting": extracting,
        "tutorials": repo_root / "Tutorials",          # main folder, next to Extracting/
        "output": extracting / "Processed",            # Extracting/Processed
        "topic_map": extracting / "topic_map.yaml",
        "subjects_dir": repo_root / "prerequisite_graph" / "subjects",
        "images": extracting / "tutorial_images",
        "env": extracting / ".env",
        "cache": extracting / ".cache",
    }


def candidate_tutorial_dirs(script_dir: Path, repo_root: Path) -> list:
    """Every plausible location of the tutorial PDFs, most-canonical first.

    The canonical location is the repo-root ``Tutorials/`` folder
    (``repo/Question_Paper_Generator/Tutorials/<Subject>/*.pdf``); the others are
    fallbacks so the script keeps working if it is run from a different layout.
    """
    return [
        repo_root / "Tutorials",            # repo/Question_Paper_Generator/Tutorials  (canonical)
        script_dir / "Tutorials",           # .../Extracting/Tutorials
        Path.cwd() / "Tutorials",           # ./Tutorials from the current working dir
    ]


def resolve_tutorials_dir(requested: str, script_dir: Path, repo_root: Path) -> Path:
    """Resolve the tutorials directory.

    If the caller passed an explicit ``--tutorials-dir`` we honour it exactly
    (and fail loudly if it is missing). Otherwise we auto-detect the first
    existing candidate so the default "just works" regardless of layout.
    """
    # An explicit path (user typed --tutorials-dir) must exist as given.
    if requested and requested.strip():
        requested_path = Path(requested)
        if requested_path.exists():
            return requested_path.resolve()
        LOG.warning("--tutorials-dir '%s' does not exist; auto-detecting instead", requested)

    for cand in candidate_tutorial_dirs(script_dir, repo_root):
        if cand.exists():
            LOG.info("Using auto-detected tutorials dir: %s", cand.resolve())
            return cand.resolve()

    tried = "\n".join(f"    - {c}" for c in candidate_tutorial_dirs(script_dir, repo_root))
    raise SystemExit(
        f"tutorials directory not found.\n"
        f"  --tutorials-dir was: {requested}\n"
        f"  tried these locations:\n{tried}\n"
        f"Pass --tutorials-dir <path-to-Tutorials> explicitly."
    )


def run(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    paths = project_paths(script_dir)
    repo_root = paths["repo_root"]
    LOG.info("Project root: %s", repo_root)

    tutorials_root = resolve_tutorials_dir(args.tutorials_dir, script_dir, repo_root)
    out_root = Path(args.output_dir).resolve()
    image_root = Path(args.image_dir).resolve()
    topic_map_path = Path(args.topic_map).resolve()
    subjects_dir = Path(args.subjects_dir).resolve() if args.subjects_dir else None

    if not topic_map_path.exists():
        raise SystemExit(f"topic map not found: {topic_map_path}")

    out_root.mkdir(parents=True, exist_ok=True)
    catalog = load_topic_catalog(topic_map_path, subjects_dir)
    graph = load_curriculum_graph(subjects_dir)
    if graph["co_text"]:
        LOG.info("Loaded %d course outcome(s) and %d prerequisite edge(s) from %s",
                 len(graph["co_text"]),
                 sum(len(v) for v in graph["prereqs"].values()),
                 subjects_dir)
    else:
        LOG.warning("no course outcomes loaded from %s -- chunks will have empty 'cos'",
                    subjects_dir)

    pdf_files = sorted(tutorials_root.rglob("*.pdf"))
    if args.limit:
        pdf_files = pdf_files[: args.limit]
    if not pdf_files:
        raise SystemExit(f"no PDF files found under {tutorials_root}")
    LOG.info("Found %d tutorial PDF(s) under %s", len(pdf_files), tutorials_root)

    # --- Gemini client ---------------------------------------------------- #
    client = None
    if not args.no_llm:
        api_key = find_api_key([
            paths["env"],                       # Extracting/.env  (canonical)
            script_dir / ".env",                # next to the script
            repo_root / ".env",                 # repo root
            Path.cwd() / ".env",                # current working dir
        ])
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY not found. Put it in a .env file (see the module "
                "docstring) or run with --no-llm for an offline smoke test."
            )
        cache_path = paths["cache"] / "gemini_cache.json"
        client = GeminiClient(api_key, primary_model=args.model, cache_path=cache_path,
                              request_interval=args.request_interval)

        def transcribe(path: Path) -> str:  # noqa: E306
            return client.transcribe_diagram(path)

        def classify(subject_name: str, text: str, topics: list):  # noqa: E306
            return client.classify_topics(subject_name, text, topics)
    else:
        LOG.warning("--no-llm: using placeholder diagram text and keyword topic tagging")

        def transcribe(path: Path) -> str:  # noqa: E306
            return f"[diagram omitted in --no-llm mode: {path.name}]"

        def classify(subject_name: str, text: str, topics: list):  # noqa: E306
            return keyword_classify(text, topics)

    # --- process each PDF; chunk within the PDF so a chunk never mixes two
    #     different tutorial sheets (they are separate weeks/topics) -------- #
    chunks_by_subject: dict = {}
    for pdf_path in pdf_files:
        subject = resolve_subject(pdf_path, tutorials_root)
        if subject not in catalog:
            LOG.warning("  no topic_map entry for subject '%s' (from %s) -- skipping",
                        subject, pdf_path.name)
            continue
        LOG.info("[%s] %s", subject, pdf_path.name)
        try:
            units = process_pdf(pdf_path, subject, tutorials_root, image_root,
                                args.keep_solutions)
        except Exception as exc:  # noqa: BLE001 - keep going across files
            LOG.error("  FAILED %s: %s", pdf_path.name, exc)
            continue
        if not units:
            continue

        # de-duplicate identical questions (normalised) within this sheet
        seen, unique_units = set(), []
        for unit in units:
            norm = re.sub(r"\s+", " ", unit.text).strip().casefold()
            if norm not in seen:
                seen.add(norm)
                unique_units.append(unit)

        subject_name = catalog[subject]["name"]
        topics = catalog[subject]["topics"]
        for group in pack_units_into_chunks(unique_units):
            description = build_description(group, transcribe)
            try:
                matches = classify(subject_name, description, topics)
            except Exception as exc:  # noqa: BLE001 - fall back to keywords
                LOG.warning("  topic classification failed (%s); keyword fallback", exc)
                matches = keyword_classify(description, topics)
            topic_ids = [m["topic_id"] for m in matches]
            own_subject = catalog[subject].get("yaml_subject", subject)
            cos, co_details = collect_cos(topic_ids, graph, args.prereq_depth,
                                          own_subject=own_subject,
                                          include_cross_subject=(args.co_scope == "all"))
            chunks_by_subject.setdefault(subject, []).append(Chunk(
                chunk_id="",  # assigned per subject below
                description=description,
                topics=[m["topic"] for m in matches],
                topic_ids=topic_ids,
                cos=cos,
                co_details=co_details,
            ))

    # --- write one JSONL per subject, renumbering chunk ids ------------- #
    total = 0
    for subject in sorted(chunks_by_subject):
        chunks = chunks_by_subject[subject]
        for index, chunk in enumerate(chunks, start=1):
            chunk.chunk_id = f"{subject}_tut_chunk_{index:03d}"
        out_path = write_subject_jsonl(subject, chunks, out_root)
        words = [len(c.description.split()) for c in chunks]
        LOG.info("Wrote %d chunk(s) for '%s' -> %s (avg %d words/chunk)",
                 len(chunks), subject, out_path,
                 (sum(words) // len(words)) if words else 0)
        total += len(chunks)

    if client is not None:
        client._save_cache()

    LOG.info("Done. %d tutorial chunk(s) across %d subject(s).",
             total, len(chunks_by_subject))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    paths = project_paths(script_dir)

    parser = argparse.ArgumentParser(
        description="Ingest tutorial PDFs into per-subject JSONL chunks (RAG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tutorials-dir", default=str(paths["tutorials"]),
                        help="Root of tutorial PDFs, organised as <dir>/<Subject>/*.pdf")
    parser.add_argument("--output-dir", default=str(paths["output"]),
                        help="Directory to write <Subject>_tutorial_chunks.jsonl into")
    parser.add_argument("--topic-map", default=str(paths["topic_map"]),
                        help="Extracting/topic_map.yaml -- its subjects: block holds the "
                             "per-subject topic catalog (id + name)")
    parser.add_argument("--subjects-dir", default=str(paths["subjects_dir"]),
                        help="prerequisite_graph/subjects dir -- source of course "
                             "outcomes (COs) and prerequisite edges for each chunk")
    parser.add_argument("--prereq-depth", type=int, default=1,
                        help="How many prerequisite levels to fold into a chunk's COs "
                             "(0 = own topics only, 1 = direct prerequisites, "
                             "-1 = full transitive closure)")
    parser.add_argument("--co-scope", choices=("subject", "all"), default="subject",
                        help="Which COs to attach: 'subject' = only the chunk's own "
                             "subject's COs (clean CO ids); 'all' = also COs of "
                             "cross-subject prerequisite topics (labels become "
                             "SUBJECT:CO when ambiguous)")
    parser.add_argument("--image-dir", default=str(paths["images"]),
                        help="Where extracted figures are written")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Primary Gemini model (falls back automatically)")
    parser.add_argument("--request-interval", type=float, default=0.0,
                        help="Minimum seconds between Gemini calls (rate-limit helper)")
    parser.add_argument("--keep-solutions", action="store_true",
                        help="Keep worked 'Solution' sections instead of stripping them")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip all Gemini calls (placeholder diagrams + keyword topics)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N PDFs (0 = all)")
    parser.add_argument("--verify", action="store_true",
                        help="Do not ingest; just validate existing "
                             "<Subject>_tutorial_chunks.jsonl files in --output-dir")
    parser.add_argument("--show-paths", action="store_true",
                        help="Print every resolved path (tutorials, output, topic-map, "
                             ".env) and exit -- use this to debug 'path not found'")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser


def show_paths() -> int:
    """Print the resolved paths the script would use, then exit."""
    script_dir = Path(__file__).resolve().parent
    paths = project_paths(script_dir)
    repo_root = paths["repo_root"]
    print("ingest_tutorials.py -- resolved paths")
    print("-" * 60)
    print(f"script_dir     : {script_dir}")
    print(f"repo_root      : {repo_root}   (auto-detected)")
    print(f"cwd            : {Path.cwd()}")
    print()
    print("tutorial dir candidates (first existing wins):")
    for cand in candidate_tutorial_dirs(script_dir, repo_root):
        print(f"  [{'OK ' if cand.exists() else '   '}] {cand}")
    print()
    for label, path in [
        ("output-dir   ", paths["output"]),
        ("topic-map    ", paths["topic_map"]),
        ("subjects-dir ", paths["subjects_dir"]),
        ("image-dir    ", paths["images"]),
        (".env         ", paths["env"]),
    ]:
        print(f"{label}: [{'OK ' if path.exists() else 'MISSING'}] {path}")
    print("-" * 60)
    return 0


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.show_paths:
        return show_paths()
    if args.verify:
        return verify_outputs(args.output_dir)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
