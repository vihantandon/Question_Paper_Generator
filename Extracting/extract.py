import fitz
import json
import math
import os
from pathlib import Path
import re
from dataclasses import asdict, dataclass

# The embedding model (all-MiniLM-L6-v2) reads only the first 256 tokens of a
# chunk and ignores the rest, so a chunk must fit in that window or its second
# half can never be found by search. 2 of the 256 are the model's own markers.
EMBED_MAX_TOKENS = 256
MAX_CHUNK_TOKENS = EMBED_MAX_TOKENS - 2
MIN_CHUNK_TOKENS = 80          # a shorter last chunk is evened out with the one before

# Tokens are not words: every punctuation mark is one token and a long or rare
# word is several, so code is far "longer" than its word count
# ("for(i=0;i<n;i++)" is 1 word but 14 tokens). Counting pieces -- a word, a
# number, or one punctuation mark -- follows the real tokenizer closely on all
# four books (real tokens per piece: median 1.05, 95% of the time under 1.2)
# and needs no model to be loaded here.
_PIECE_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
TOKENS_PER_PIECE = 1.2

@dataclass
class Chunk:
    chunk_id: str
    subject: str
    source_file: str
    section_title: str
    page_start: int
    page_end: int
    text: str

def has_text_layer(doc) -> bool:
        """Quick check: if extracted text from the first few pages is near-empty,
        this PDF is scanned/image-based and needs the vision-LLM transcription
        approach instead of this script."""

        sample_text = "".join(doc[p].get_text() for p in range(min(5,len(doc))))
        return len(sample_text.strip()) > 200

def get_toc_sections(doc):
      toc = doc.get_toc()
      if not toc:
            return None

      # A page belongs to exactly ONE section. Several TOC entries often start
      # on the same page (a heading and its sub-headings), and a section used
      # to run on to the next entry's start page as well -- so one page's text
      # was stored again under every one of those titles. Now each page that
      # starts an entry opens one section (named after the first entry on it),
      # which ends just before the next such page. Going by page rather than
      # by TOC order also keeps a broken bookmark (DS: "53.3.3 Corner Block
      # List" points at page 1) from swallowing the rest of the book.
      first_title = {}
      for level,title,page in toc:
            if 1 <= page <= len(doc):
                  first_title.setdefault(page-1,title.strip())

      starts = sorted(first_title)
      sections = []
      for i , start_page in enumerate(starts):
            end_page = (starts[i+1]-1) if i+1 < len(starts) else len(doc)-1
            sections.append((first_title[start_page],start_page,end_page))

      return sections

def get_heading_sections_by_font(doc):
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(round(span["size"]))
    if not sizes:
        return None
 
    body_size = max(set(sizes), key=sizes.count)  # most common size = body text
    heading_threshold = body_size + 2
 
    sections = []
    current_title, current_start = "Untitled", 0
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    is_bold = "Bold" in span.get("font", "")
                    text = span["text"].strip()
                    if (span["size"] >= heading_threshold or is_bold) and 3 < len(text) < 120:
                        sections.append((current_title, current_start,
                                          page_num - 1 if page_num > current_start else page_num))
                        current_title, current_start = text, page_num
    sections.append((current_title, current_start, len(doc) - 1))
    return sections[1:] if len(sections) > 1 else sections  # drop the placeholder "Untitled" head

def extract_section_text(doc, start_page, end_page):
     return "\n".join(doc[p].get_text() for p in range(start_page, end_page + 1))

def estimate_tokens(text):
    """About how many tokens the embedding model will make of this text
    (deliberately a little high, so a chunk that passes really fits)."""
    return math.ceil(len(_PIECE_RE.findall(text)) * TOKENS_PER_PIECE)

def split_into_units(text, max_tokens):
    """The pieces chunks are built from: the text's lines. A line that is too
    long on its own (a table row, a listing with no line breaks) is cut
    between words."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if estimate_tokens(line) <= max_tokens:
            units.append(line)
            continue
        run = []
        for word in line.split():
            if run and estimate_tokens(" ".join(run + [word])) > max_tokens:
                units.append(" ".join(run))
                run = []
            run.append(word)
        units.append(" ".join(run))
    return units

def split_into_chunks(text, max_tokens = MAX_CHUNK_TOKENS):
    """Split a section's text into chunks the embedding model can read in
    full. Lines are packed one after another until the next would not fit, so
    a cut only ever falls between two lines.

    (This used to cut at blank lines and aim for 350 words. In three of the
    four books a blank line only appears between PAGES, so a chunk was a whole
    page -- two to three times what the model reads -- and a chapter's title
    page, being short, became a chunk of its own that matched every search
    for that chapter's topic.)"""

    units = [(unit, estimate_tokens(unit)) for unit in split_into_units(text, max_tokens)]
    size = lambda group: sum(tokens for _, tokens in group)

    groups,current = [],[]
    for unit in units:
        if current and size(current) + unit[1] > max_tokens:
            groups.append(current)
            current = []
        current.append(unit)
    if current:
        groups.append(current)

    # A short last chunk (it only exists because it did not fit into the one
    # before): share the lines of the last two chunks out evenly instead.
    if len(groups) > 1 and size(groups[-1]) < MIN_CHUNK_TOKENS:
        both = groups[-2] + groups[-1]
        cut = 1
        while size(both[:cut + 1]) <= size(both) / 2:
            cut += 1
        if size(both[cut:]) <= max_tokens:
            groups[-2:] = [both[:cut], both[cut:]]

    return ["\n".join(unit for unit, _ in group) for group in groups]

def process_pdf(path , subject):
    doc = fitz.open(path)

    if not has_text_layer(doc):
        raise ValueError(f"{path} appears to be scanned/image-based. "
        "Use the vision-LLM transcription approach instead of this script.")

    sections = get_toc_sections(doc) or get_heading_sections_by_font(doc)

    all_chunks = []
    for idx, (title, start_page, end_page) in enumerate(sections):
        section_text = extract_section_text(doc,start_page,end_page)
        if len(section_text.strip()) < 20:
            continue 

        for j,chunk_text in enumerate(split_into_chunks(section_text)):
            all_chunks.append(Chunk(
                chunk_id = f"{subject}_{idx:03d}_{j:02d}",
                subject = subject,
                source_file = path,
                section_title = title,
                page_start = start_page+1,
                page_end = end_page+1,
                text = chunk_text,
            ))
    return all_chunks

HERE = Path(__file__).resolve().parent     # Extracting/
ROOT = HERE.parent                          # repo root


def process_all_books(books_dir = ROOT / "Books", output_dir = HERE / "Processed"):
    """Books/*.pdf -> Extracting/Processed/<book>_chunks.jsonl.
    Paths are found relative to this file, so it runs from any folder.
    NOTE: this rewrites the chunk files and drops the topic_id fields added
    by tag_topics.py -- run  python tag_topics.py  again afterwards."""
    books_path = Path(books_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .pdf files in the Books directory
    pdf_files = list(books_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{books_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process...\n")

    for pdf_file in pdf_files:
        # Use filename (without extension) as subject/identifier
        subject_name = pdf_file.stem
        output_file = output_path / f"{subject_name}_chunks.jsonl"

        print(f"Processing: {pdf_file.name}...")

        try:
            chunks = process_pdf(str(pdf_file), subject=subject_name)

            with open(output_file, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(asdict(c)) + "\n")

            unique_sections = len(set(c.section_title for c in chunks))
            print(
                f"  Saved {len(chunks)} chunks across {unique_sections} sections to '{output_file}'\n"
            )

        except Exception as e:
            print(f"  Failed to process {pdf_file.name}: {e}\n")


if __name__ == "__main__":
    process_all_books()