import fitz
import json
import re
from dataclasses import asdict, dataclass

TARGET_CHUNK_WORDS = 350
MIN_CHUNK_WORDS = 60

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

      sections = []
      for i , (level,title,page) in enumerate(toc):
            start_page = page-1
            end_page = (toc[i+1][2]-1) if i+1 < len(toc) else len(doc)-1
            end_page = max(start_page,end_page)
            sections.append((title.strip(),start_page,end_page))

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

def split_into_chunks(text, target_words = TARGET_CHUNK_WORDS):         
    """Split a section's text into sub-chunks along paragraph breaks,
    respecting a target word count per chunk."""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks,current,current_len = [],[],0

    for para in paragraphs:
        para_len = len(para.split())
        if current_len + para_len > target_words and current:
            chunks.append(" ".join(current))
            current,current_len = [],0
        current.append(para)
        current_len += para_len
         
    if current:
         chunks.append(" ".join(current))

    if len(chunks) > 1 and len(chunks[-1].split()) < MIN_CHUNK_WORDS:
         chunks[-2] += " " + chunks[-1]
         chunks.pop()

    return chunks

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

if __name__ == "__main__":
    chunks = process_pdf("/home/vihan-tandon/Desktop/Question_Paper_Generator/Books/Let us c - yashwantkanetkar.pdf", subject = "SDF-1")
    with open("Processed", "w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    print(f"Extracted {len(chunks)} chunks from {len(set(c.section_title for c in chunks))} sections.")