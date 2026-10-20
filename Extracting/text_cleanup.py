"""
text_cleanup.py -- turn a book's PDF pages into clean text before chunking.

    from text_cleanup import clean_book_pages, cut_back_matter
    pages = clean_book_pages(doc)                  # one clean string per page
    text = cut_back_matter("\\n".join(pages[3:9]))  # one section, without its reference list

Used by extract.py. What was wrong with the plain page.get_text() text, and
what is done about it here (numbers measured on the committed chunk files):

1. LIGATURES. The DS and APS books store "fi", "ff", "ffi", "fl" as one
   special character each ("eﬃcient"): 70% of DS chunks and 63% of APS
   chunks had them. The embedding model most likely reads such a word as
   unknown. -> replaced by the plain letters.

2. MATHS FONTS. The APS book (CLRS) sets its maths in MathTime Pro 2 fonts
   (MT2...), and the PDF does not say which real symbol each glyph is, so
   "n1 = q - p + 1" came out as "n1 D q \\x00 p C 1", "A[p..q]" as
   "AŒp : : q\\x8d" and "Θ(n²)" as "‚.n2/" (70% of APS chunks). Each MT2 font
   has its own table below, read off the book itself: every character was
   listed with the line it sits in (the tables follow TeX's standard maths
   font layouts, with the changes MathTime makes). The DS book's big
   brackets (CMEX10) use the same extension table.

3. CONTROL CHARACTERS left after that (figures drawn with tiny "T1".."T63"
   fonts in the DS book) -> removed.

4. RUNNING HEADERS AND FOOTERS: page numbers, "Handbook of Data Structures
   and Applications", "© 2005 by Chapman & Hall/CRC", chapter titles at the
   top of every page. -> a line in the top or bottom MARGIN_BAND of a page
   is removed when it is a page number, or when the same text (numbers
   ignored) is in that band on RUNNING_MIN_PAGES or more pages.

4b. CONTENTS LINES: every DS handbook chapter opens with a box listing its
   sections ("Stacks and Queues ........ 2-12", "Stack Implementation •
   Queue Implementation"); one was the prerequisite source for Linear DS.
   -> dot-leader lines (and the page number after one) and "•" outlines
   are removed, except in a code font.

5. FIGURE LABELS: a figure's node numbers and letters come out one per line
   ("20 / 10 / 15 / 5 / 8 / 30 ..."), and some chunks were mostly that.
   -> a run of LABEL_RUN_MIN or more such short lines is removed, unless
   it is set in a code font (program output must stay).

6. BACK MATTER of a chapter: reference lists, acknowledgments, CLRS's
   "Chapter notes" (history and citations). 6% of DS chunks were reference
   lists, and one was retrieved for "Heaps". -> cut_back_matter() cuts a
   section at such a heading. "References" only counts when citations
   follow it: in the C++ book it is a real chapter about reference variables.

Nothing here opens a file or needs anything but the page objects PyMuPDF
gives, so the tests run it on made-up pages.
"""
import re
from dataclasses import dataclass

try:
    import pymupdf
    TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT   # the settings page.get_text() uses
except ImportError:                       # the tests use made-up pages
    TEXT_FLAGS = None

MARGIN_BAND = 0.08         # top / bottom share of a page where running headers and footers sit
RUNNING_MIN_PAGES = 3      # the same margin text on this many pages = a running header/footer
LABEL_RUN_MIN = 5          # this many short label lines in a row = figure labels

# ------------------------------------------------------------------ characters
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
                           "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# MathTime Pro 2 symbol fonts: MT2SYT (text size), MT2SYS (script), MT2SYF,
# MT2BSYT / MT2BSYS (bold). Mostly TeX's symbol layout; MathTime puts "+" and
# "=" at C and D, and a few more symbols elsewhere.
MT2_SYMBOL = {
    "\x00": "−", "\x01": "·", "\x02": "×", "\x03": "*", "\x0f": "•",
    "\x11": "≡", "\x12": "⊆", "\x14": "≤", "\x15": "≥", "\x19": "≈", "\x1a": "⊂",
    "\x1c": "≈", "\x1d": "≪",
    "!": "→", '"': "↑", "#": "↓", "$": "↔", "(": "⇐", ")": "⇒", "-": "↖",
    "0": "′", "1": "∞", "2": "∈", "6": "!",          # "6" is the "not" stroke: 6D -> "!=" (≠)
    ":": "¬", ";": "∅", "[": "∪", "\\": "∩", "^": "∧", "_": "∨",
    "b": "⌊", "c": "⌋", "d": "⌈", "e": "⌉", "f": "{", "g": "}", "h": "⟨", "i": "⟩",
    "j": "|", "k": "‖", "p": "√", "}": "⋄",
    "C": "+", "D": "=", "I": ";", "W": ":", "N": "¯", "y": "^",
    "¤": "≠", "˚": "⊕", "˝": "⊗", "†": "∠", "…": "∉", "˙": "", "\x8f": "",
}

# MathTime Pro 2 maths italic: MT2MIT, MT2MIS, MT2MIF, MT2BMIT, MT2BMIS.
# Letters and digits are themselves; the brackets, punctuation and Greek
# letters are not ("f .n/" is f(n), "‚.n2/" is Θ(n²), "AŒ1 : : n\x8d" is A[1..n]).
MT2_MATH_ITALIC = {
    ".": "(", "/": ")", ";": ",", ":": ".", "=": "/", "Œ": "[", "\x8d": "]", "Š": "!",
    "\x0f": "ε", '"': "ε", "\x10": "ζ", "\x11": "η", "\x15": "λ", "\x16": "μ", "\x17": "v",
    "\x19": "π", "\x1a": "ρ", "\x1b": "σ", "\x1e": "φ", "!": "ω",
    "˛": "α", "ˇ": "β", "ı": "δ", "´": "z",
    "‚": "Θ", "\x7f": "Ω", "ˆ": "Φ", "†": "Σ", "…": "Π", "\x81": "Δ",
    "?": "⋆", "‹": "",
}

# Big brackets, sums and integrals: MathTime's MT2EXA / MT2EXE / MT2EXF and
# Computer Modern's CMEX10 (DS book). TeX's extension layout: one character
# per size, plus the pieces tall brackets are built from.
TEX_EXTENSION = {
    "\x00": "(", "\x01": ")", "\x02": "[", "\x03": "]", "\x04": "⌊", "\x05": "⌋", "˘": "⌋",
    "\x06": "⌈", "˙": "⌈", "\x07": "⌉", "\x08": "{", "˚": "{", "\t": "}", "\n": "⟨", "\x0b": "⟩",
    "\x0c": "|", "ˇ": "|", "\x0d": "‖",
    "\x10": "(", "\x11": ")", "\x12": "(", "\x13": ")", "\x14": "[", "\x15": "]",
    "\x16": "⌊", "\x17": "⌋", "\x18": "⌈", "\x19": "⌉", "\x1a": "{", "\x1b": "}",
    "\x1c": "⟨", "\x1d": "⟩",
    "!": ")", '"': "[", "#": "]", "$": "⌊", "%": "⌋", "&": "⌈", "'": "⌉", "(": "{", ")": "}",
    "P": "∑", "X": "∑", "Q": "∏", "Y": "∏", "R": "∫", "Z": "∫", "S": "⋃", "[": "⋃",
    "T": "⋂", "\\": "⋂", "V": "⋀", "^": "⋀", "W": "⋁", "_": "⋁", "M": "⨁",
    "p": "√", "q": "√", "r": "√", "s": "√", "b": "^", "c": "^", "d": "^", "e": "~", "f": "~", "g": "~",
    "h": "[", "i": "]", "j": "⌊", "k": "⌋", "l": "⌈", "m": "⌉", "n": "{", "o": "}",
    "’": "", "„": "", "œ": "", "“": "", "—": "", "‚": "", "ˆ": "",
}

BLACKBOARD = {"Z": "ℤ", "R": "ℝ", "N": "ℕ", "C": "ℂ", "Q": "ℚ"}

# (font name prefix, table). The first prefix that matches is used.
FONT_TABLES = [
    ("MT2SYA", {"−": "∤"}),           # "d ∤ a": d does not divide a
    ("MT2SY", MT2_SYMBOL), ("MT2BSY", MT2_SYMBOL),
    ("MT2MI", MT2_MATH_ITALIC), ("MT2BMI", MT2_MATH_ITALIC),
    ("MT2EX", TEX_EXTENSION), ("CMEX", TEX_EXTENSION),
    ("MT2HRB", BLACKBOARD),
    ("MT2MS", {"\t": "ℐ"}),           # script letters: the matroid's family of sets
]

SPACED_DOTS_RE = re.compile(r"\.(?: \.)+")      # ". . ." from maths "dots" -> "..."


def font_table(font):
    return next((table for prefix, table in FONT_TABLES if font.startswith(prefix)), None)


def clean_span(font, text):
    """One run of text in one font, with its real characters."""
    table = font_table(font)
    if table:
        text = "".join(table.get(ch, ch) for ch in text)
    return CONTROL_RE.sub("", text.translate(LIGATURES))


def clean_text(text):
    """For text with no font information (e.g. a section title)."""
    return CONTROL_RE.sub("", text.translate(LIGATURES))


# ------------------------------------------------------------------ lines and pages
MONO_FONT_RE = re.compile(r"courier|mono|cmtt|consol", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^(?:\d{1,4}|\d{1,3}-\d{1,4}|[ivxlcdm]{1,7})$", re.IGNORECASE)
LABEL_RE = re.compile(r"^(?:[A-Za-z0-9]{1,3}|\d+(?:\.\d+)?|\d+,\d+)$")   # "4,10": an interval heap node
# a contents line: "Stacks and Queues ........................ 2-12"
DOT_LEADER_RE = re.compile(r"(?:\.\s?){5,}\s*\S{0,8}\s*$")
# a chapter outline: "Stack Implementation • Queue Implementation" (not a "• item" list)
OUTLINE_RE = re.compile(r"\S\s+•\s+\S")


@dataclass
class PageLine:
    text: str
    y0: float
    y1: float
    code: bool          # set in a code (monospace) font


def page_lines(page):
    """The page's lines, each with its real characters and where it sits.
    Uses the same extraction settings as page.get_text(), so a page with
    nothing to fix gives exactly the same lines."""
    data = page.get_text("dict", flags=TEXT_FLAGS)
    lines = []
    for block in data["blocks"]:
        if block.get("type", 0) != 0:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = SPACED_DOTS_RE.sub(lambda m: m.group(0).replace(" ", ""),
                                      "".join(clean_span(s["font"], s["text"]) for s in spans))
            lines.append(PageLine(text, line["bbox"][1], line["bbox"][3],
                                  any(MONO_FONT_RE.search(s["font"]) for s in spans)))
    return lines


def margin_key(text):
    """How margin lines are compared: numbers, case and spacing do not matter."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.strip())).casefold()


def in_margin(line, height):
    return line.y1 < height * MARGIN_BAND or line.y0 > height * (1 - MARGIN_BAND)


def find_running_lines(pages):
    """pages: [(height, [PageLine, ...]), ...] for the whole book. The margin
    texts that repeat on RUNNING_MIN_PAGES or more pages, as margin_key()s."""
    seen = {}
    for number, (height, lines) in enumerate(pages):
        for line in lines:
            if line.text.strip() and in_margin(line, height):
                seen.setdefault(margin_key(line.text), set()).add(number)
    return {key for key, where in seen.items() if len(where) >= RUNNING_MIN_PAGES}


def is_header_or_footer(line, height, running):
    if not in_margin(line, height):
        return False
    text = line.text.strip()
    return bool(PAGE_NUMBER_RE.match(text)) or margin_key(text) in running


def drop_label_runs(lines):
    """Remove runs of LABEL_RUN_MIN or more short label lines (figure node
    numbers and letters), except in a code font."""
    out, run = [], []
    for line in lines + [None]:
        if line is not None and not line.code and LABEL_RE.match(line.text.strip()):
            run.append(line)
            continue
        if len(run) < LABEL_RUN_MIN:
            out += run
        run = []
        if line is not None:
            out.append(line)
    return out


def drop_contents_lines(lines):
    """Remove table-of-contents lines: dot leaders ("Introduction ...... 1-1",
    and the page number on the line after one) and chapter outlines
    ("Stack Implementation • Queue Implementation"). Every DS handbook
    chapter opens with such a box; it names sections but states no fact.
    Lines in a code font are kept."""
    out, after_leader = [], False
    for line in lines:
        text = line.text.strip()
        if after_leader and PAGE_NUMBER_RE.match(text):
            after_leader = False
            continue
        after_leader = False
        if not line.code and DOT_LEADER_RE.search(text):
            after_leader = True
            continue
        if not line.code and OUTLINE_RE.search(text):
            continue
        out.append(line)
    return out


def clean_page(height, lines, running):
    """One page's clean text: no headers/footers, no contents lines, no
    figure-label runs, no blank lines."""
    kept = [ln for ln in lines if ln.text.strip() and not is_header_or_footer(ln, height, running)]
    return "".join(ln.text + "\n" for ln in drop_label_runs(drop_contents_lines(kept)))


def clean_book_pages(doc):
    """Every page of the book as clean text (index = page number - 1).
    Headers are found from the whole book first, so this needs all pages."""
    pages = [(doc[p].rect.height, page_lines(doc[p])) for p in range(len(doc))]
    running = find_running_lines(pages)
    return [clean_page(height, lines, running) for height, lines in pages]


# ------------------------------------------------------------------ end of a chapter
BACK_MATTER_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?(references|bibliography|further reading|"
                            r"acknowledge?ments?|chapter notes)\s*$", re.IGNORECASE)
CITATION_RE = re.compile(r"^(?:\[\d+\]|\d{1,3}\.\s+[A-Z])")
YEAR_RE = re.compile(r"\b(?:1[89]|20)\d\d\b")


def back_matter_starts(lines, i):
    """Does lines[i] start a reference list / acknowledgments / chapter notes?"""
    m = BACK_MATTER_RE.match(lines[i].strip())
    if not m:
        return False
    after = [ln.strip() for ln in lines[i + 1:i + 8] if ln.strip()]
    if not after:
        return True
    if m.group(1).lower() in ("references", "bibliography"):
        # a list of citations must follow: "[1] D. Knuth, ... 1997." -- not a
        # C++ chapter on reference variables, and not a table of contents
        return (any(CITATION_RE.match(ln) for ln in after[:3])
                and any(YEAR_RE.search(ln) for ln in after[:6]))
    # acknowledgments, chapter notes, further reading: a paragraph follows
    # (in a table of contents a page number follows instead)
    return len(after[0]) >= 30 and not PAGE_NUMBER_RE.match(after[0])


def cut_back_matter(text):
    """A section's text up to its reference list, acknowledgments or chapter
    notes -- they end a chapter, and none of them is material to set a
    question from."""
    lines = text.split("\n")
    for i in range(len(lines)):
        if back_matter_starts(lines, i):
            return "\n".join(lines[:i]).rstrip("\n") + ("\n" if i else "")
    return text
