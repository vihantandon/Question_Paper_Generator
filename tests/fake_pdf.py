"""
Made-up PDF pages for the extractor tests (test_extract, test_housekeeping,
test_text_cleanup). Just enough of a PyMuPDF page for extract.py and
Extracting/text_cleanup.py:

    page.get_text()          the plain text
    page.get_text("dict")    lines with their font and position
    page.rect.height

    FakePage("first line\\nsecond line")       every line in the body of the page
    FakePage(lines=[("Handbook of ...", 20),   (text, y): a header in the top margin
                    ([("MT2MIT", "f .n/"), ("MT2SYT", " D "), ("MT2MIT", "1")], 300),
                    ("int x = 0;", 320, "Courier")])       (text, y, font)
"""
import types

PAGE_HEIGHT = 800.0
BODY_TOP, BODY_BOTTOM = 100.0, 700.0      # well inside text_cleanup's top / bottom margins


class FakePage:
    def __init__(self, text=None, lines=None, height=PAGE_HEIGHT):
        if lines is None:
            rows = (text or "").split("\n")
            step = min(10.0, (BODY_BOTTOM - BODY_TOP) / max(1, len(rows)))
            lines = [(row, BODY_TOP + i * step) for i, row in enumerate(rows)]
        self.lines = [self._spans(entry) for entry in lines]
        self.rect = types.SimpleNamespace(height=height)
        self.text = text if text is not None else "".join(
            "".join(t for _, t in spans) + "\n" for spans, _ in self.lines)

    @staticmethod
    def _spans(entry):
        content, y = entry[0], entry[1]
        font = entry[2] if len(entry) > 2 else "Times-Roman"
        spans = [(font, content)] if isinstance(content, str) else list(content)
        return spans, y

    def get_text(self, kind="text", **kwargs):
        if kind != "dict":
            return self.text
        return {"blocks": [{"type": 0, "lines": [
            {"bbox": (50.0, y, 500.0, y + 8.0),
             "spans": [{"font": font, "text": t} for font, t in spans]}
            for spans, y in self.lines]}]}
