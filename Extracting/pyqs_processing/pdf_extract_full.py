# pdf_extract_full.py
"""
Step 1 of the PYQ pipeline: PYQ PDFs -> one jsonl of page texts per paper.

    PYQs/*.pdf  ->  Extracting/Processed/pyq_pages/<paper>_pages.jsonl

All paths are found relative to this file, so it runs from any folder:
    python Extracting/pyqs_processing/pdf_extract_full.py
    python Extracting/pyqs_processing/pdf_extract_full.py --input-dir <folder of PDFs>

Needs OPENROUTER_API_KEY in the environment (vision model for scanned
pages / diagrams). Papers that already have a pages file are skipped.
Next step: split_pyq_questions.py
"""
import argparse
import fitz  # PyMuPDF
import json
import io
import os
import time
import base64
from pathlib import Path
from PIL import Image
from openai import OpenAI

HERE = Path(__file__).resolve().parent            # Extracting/pyqs_processing
ROOT = HERE.parent.parent                          # repo root
DEFAULT_INPUT_DIR = ROOT / "PYQs"
DEFAULT_OUTPUT_DIR = ROOT / "Extracting" / "Processed" / "pyq_pages"

# OpenRouter is OpenAI-API-compatible; just point the client at their base_url.
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# OpenRouter's individual free model slugs rotate on/off frequently (models get
# pulled off the free tier with no warning, as you just hit). "openrouter/free:free"
# is a router that auto-picks a currently-free model and filters for whatever
# capability the request needs (here: image understanding), so it's the most
# stable choice for a script like this. If you want to pin a specific model
# instead, check https://openrouter.ai/models?max_price=0&modality=text%2Bimage-%3Etext
# for what's currently free and swap it in.
MODEL_NAME = "openrouter/free:free"

# Optional but recommended by OpenRouter for free-tier usage/attribution.
EXTRA_HEADERS = {
    "HTTP-Referer": "https://localhost",   # any URL identifying your app
    "X-Title": "PYQ PDF Extractor",
}

MIN_TEXT_LEN_PER_PAGE = 40      # below this, treat page as scanned/sparse
DELAY_BETWEEN_CALLS = 4          # seconds; raise this if you hit rate-limit errors

VISION_PROMPT = """Transcribe this page completely and precisely. Follow these rules:
1. Plain text: transcribe verbatim, preserving question numbers, sub-parts (a)/(b)/(c), and marks in brackets exactly as printed.
2. Any diagram, graph, tree, or figure: describe it in words inside a [DIAGRAM: ...] block. State what kind of diagram it is and its content (e.g. "[DIAGRAM: undirected graph, 5 nodes A-E; adjacency list: {A->{B(4), C(5)}, B->{C(2), E(3)}, C->D(7)}").
3. Any C/C++/pseudocode: reproduce it exactly, preserving indentation, in a fenced code block like ```c ... ``` or ```cpp ... ```.
4. Do not answer any questions shown, do not add commentary — transcription only."""


def has_meaningful_text(page) -> bool:
    return len(page.get_text().strip()) >= MIN_TEXT_LEN_PER_PAGE


def has_images(page) -> bool:
    return len(page.get_images(full=True)) > 0


def page_to_pil_image(page, dpi=200) -> Image.Image:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.open(io.BytesIO(pix.tobytes("png")))


def pil_image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def transcribe_with_vision(page, retries=3) -> str:
    img = page_to_pil_image(page)
    data_url = pil_image_to_data_url(img)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                extra_headers=EXTRA_HEADERS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = (2 ** attempt) * 5
            print(f"    retrying after error ({e}); waiting {wait}s...")
            time.sleep(wait)


def extract_page(page) -> tuple[str, str]:
    """Returns (text, method). Escalates to vision model whenever the page
    has any embedded image (diagram) or too little extractable text (scanned)."""
    if has_meaningful_text(page) and not has_images(page):
        return page.get_text().strip(), "text_layer"
    text = transcribe_with_vision(page)
    time.sleep(DELAY_BETWEEN_CALLS)  # stay under free-tier rate limits
    return text, "vision_llm"


def _display_path(path: Path) -> str:
    """Path relative to the repo root (e.g. 'PYQs/T1 APS.pdf'), so the
    records don't contain one teammate's absolute local path."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _norm_stem(stem: str) -> str:
    """'T2 SDF 2' and 'T2_SDF_2' are the same paper."""
    return "".join(ch for ch in stem.lower() if ch.isalnum())


def process_pdf(path: str, subject: str) -> list[dict]:
    doc = fitz.open(path)
    records = []
    for i, page in enumerate(doc):
        text, method = extract_page(page)
        records.append({
            "page_id": f"{subject}_{Path(path).stem}_p{i+1:03d}",
            "subject": subject,
            "source_file": _display_path(path),
            "page_num": i + 1,
            "extraction_method": method,
            "text": text,
        })
        print(f"  page {i+1}/{len(doc)} -> {method}")
    return records


def process_all_pdfs(input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    input_path, output_path = Path(input_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Existing pages files, keyed by a normalized name, so a renamed file
    # ("T2_SDF_2_pages.jsonl" for "T2 SDF 2.pdf") still counts as done.
    done = {_norm_stem(f.name[:-len("_pages.jsonl")]): f
            for f in output_path.glob("*_pages.jsonl")}

    pdf_files = sorted(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in '{input_dir}'")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    for pdf_file in pdf_files:
        subject = pdf_file.stem.split("_")[0]
        out_file = done.get(_norm_stem(pdf_file.stem),
                            output_path / f"{pdf_file.stem}_pages.jsonl")

        if out_file.exists():
            print(f"Skipping {pdf_file.name} — output already exists at {out_file}")
            print("  (delete that file first if you want to re-extract it)\n")
            continue

        print(f"Processing {pdf_file.name}...")
        try:
            records = process_pdf(str(pdf_file), subject)
        except Exception as e:
            print(f"  FAILED on {pdf_file.name}: {e}")
            print(f"  Already-processed PDFs are safe — their .jsonl files are untouched.\n")
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(records)} pages -> {out_file}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PYQ PDFs -> page-level jsonl")
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="folder with the PYQ PDFs (default: <repo>/PYQs)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="where *_pages.jsonl go (default: Extracting/Processed/pyq_pages)")
    args = ap.parse_args()
    process_all_pdfs(args.input_dir, args.output_dir)