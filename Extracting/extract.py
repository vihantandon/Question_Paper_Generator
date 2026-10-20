import fitz
import json
import os
import sys
from pathlib import Path
from dataclasses import asdict, dataclass

# How big a chunk may be (the embedding model reads only 256 tokens) is one
# rule shared with the tutorial extractor: see token_budget.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_budget import (  # noqa: E402,F401  (re-exported: extract.estimate_tokens etc.)
    EMBED_MAX_TOKENS, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, TOKENS_PER_PIECE,
    estimate_tokens, split_into_chunks, split_into_units)
# Ligatures, the maths fonts' real symbols, running headers, figure labels and
# reference lists are dealt with before chunking: see text_cleanup.py.
from text_cleanup import clean_book_pages, clean_text, cut_back_matter  # noqa: E402

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

def extract_section_text(pages, start_page, end_page):
    """The section's clean text (pages from clean_book_pages), without the
    reference list / acknowledgments / chapter notes that end a chapter."""
    return cut_back_matter("\n".join(pages[start_page:end_page + 1]))

def display_path(path):
    """The book's path as stored in the chunk files: relative to the repo
    root with forward slashes ("Books/DS.pdf"), so the files do not carry
    the full folder path of whoever ran the extraction."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:                      # a PDF outside the repo: its name is enough
        return resolved.name

def process_pdf(path , subject):
    doc = fitz.open(path)

    if not has_text_layer(doc):
        raise ValueError(f"{path} appears to be scanned/image-based. "
        "Use the vision-LLM transcription approach instead of this script.")

    sections = get_toc_sections(doc) or get_heading_sections_by_font(doc)
    pages = clean_book_pages(doc)

    all_chunks = []
    for idx, (title, start_page, end_page) in enumerate(sections):
        section_text = extract_section_text(pages,start_page,end_page)
        if len(section_text.strip()) < 20:
            continue

        for j,chunk_text in enumerate(split_into_chunks(section_text)):
            all_chunks.append(Chunk(
                chunk_id = f"{subject}_{idx:03d}_{j:02d}",
                subject = subject,
                source_file = display_path(path),
                section_title = clean_text(title),
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
                    f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

            unique_sections = len(set(c.section_title for c in chunks))
            print(
                f"  Saved {len(chunks)} chunks across {unique_sections} sections to '{output_file}'\n"
            )

        except Exception as e:
            print(f"  Failed to process {pdf_file.name}: {e}\n")


if __name__ == "__main__":
    process_all_books()