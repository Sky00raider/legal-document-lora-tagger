"""Turn a .txt or .pdf document into a list of clauses.

Indian public contracts (GeM terms, GCCs, tenders) number their clauses: "7", "7.1",
"5.7.1", "3)", or roman numerals "xi." for top-level items. In PDF text the number
often sits on its own line, and short headings ("Performance security") stand alone
between the number and the body. So:

  1. Clean each page: drop page numbers and footers ("6 | P a g e", "[53]", "Page 3 of 9")
     and running headers/footers (any short line that repeats on many pages).
  2. Start a new clause at every numbered marker (1. / 1.1 / 2.3.4 / 3) / xi.).
     Lettered sub-items ((a), b., (iv)) stay inside their parent clause: a list of
     acceptable forms of security is one clause, not five.
  3. A "clause" that is only a short heading is not tagged on its own: it becomes the
     title of the next clause ("Performance security. Within 21 days ..."). That is
     also how LEDGAR clauses look (they often begin with their heading).
  4. Drop fragments too short to tag (fewer than `min_words` words).

For .txt files, blank lines also separate clauses (people type paragraphs).
"""
import re
from collections import Counter
from pathlib import Path

NUMBER = r"(?:\d{1,2}(?:\.\d{1,2}){0,3}[.)]?|[ivx]{1,5}[.)]|\((?:\d{1,2}|[ivx]{1,5})\))"
MARKER_ALONE = re.compile(rf"^{NUMBER}$", re.IGNORECASE)
MARKER_START = re.compile(rf"^({NUMBER})\s+(?=[A-Z(\"'])")
PAGE_FURNITURE = re.compile(r"^(\d{1,4}|\[\d{1,4}\]|\d+\s*\|\s*P\s*a\s*g\s*e|page\s+\d+(\s+of\s+\d+)?|-\s*\d+\s*-)$",
                            re.IGNORECASE)
TOC_LEADER = re.compile(r"\.{6,}|(?:\. ){5,}")
HEADING_MAX_WORDS = 10


def clean_text(text: str) -> str:
    """Normalise quotes/dashes that PDFs mangle, collapse whitespace."""
    text = text.replace("�", "'").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return " ".join(text.split())


def page_lines(pages: list[str]) -> list[str]:
    """All non-empty lines, minus page numbers and lines repeated on many pages (headers/footers)."""
    per_page = [[clean_text(line) for line in page.splitlines()] for page in pages]
    per_page = [[line for line in lines if line] for lines in per_page]
    repeats = Counter(line for lines in per_page for line in set(lines) if len(line) < 80)
    min_repeats = max(3, len(pages) // 3)
    out = []
    for lines in per_page:
        for i, line in enumerate(lines):
            # A lone number is a page number only at the top or bottom of a page; elsewhere it is a clause number.
            at_edge = i < 2 or i >= len(lines) - 2
            if at_edge and PAGE_FURNITURE.match(line) and not _next_is_marker(lines, i):
                continue
            if repeats[line] >= min_repeats and not MARKER_ALONE.match(line):
                continue
            if TOC_LEADER.search(line):                  # table of contents: "Warranty ........ 60"
                continue
            out.append(line)
    return out


def _next_is_marker(lines: list[str], i: int) -> bool:
    """True if line i is a bare number that is followed by text, not by another number (so it is a clause number)."""
    if not MARKER_ALONE.match(lines[i]) or i + 1 >= len(lines):
        return False
    nxt = lines[i + 1]
    return not (MARKER_ALONE.match(nxt) or MARKER_START.match(nxt) or PAGE_FURNITURE.match(nxt))


def _is_heading(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= HEADING_MAX_WORDS and not text.rstrip().endswith((".", ";", ":", ","))


def split_lines(lines: list[str], min_words: int = 6) -> list[dict]:
    """Group lines into clauses at numbered markers. Returns [{"number", "text"}]."""
    raw: list[dict] = []
    current = {"number": "", "lines": []}
    pending_number = None
    for line in lines:
        if line == "":                                   # paragraph break (txt only)
            if current["lines"]:
                raw.append(current)
            current = {"number": "", "lines": []}
            continue
        if MARKER_ALONE.match(line):
            pending_number = line
            continue
        m = MARKER_START.match(line)
        if pending_number is not None or m:
            if current["lines"]:
                raw.append(current)
            # A number written inside the line wins over a bare number on the line before.
            number, body = (m.group(1), line[m.end():]) if m else (pending_number, line)
            current = {"number": number, "lines": [body]}
            pending_number = None
        else:
            current["lines"].append(line)
    if current["lines"]:
        raw.append(current)

    clauses, heading = [], None
    for c in raw:
        text = clean_text(" ".join(c["lines"]))
        if _is_heading(text):                            # "Performance security" -> title of next clause
            heading = text if heading is None else f"{heading}. {text}"
            continue
        if heading:
            text = f"{heading}. {text}"
            heading = None
        if len(text.split()) >= min_words:
            clauses.append({"number": c["number"].strip("()").rstrip(".)"), "text": text})
    return clauses


def split_clauses(text: str, min_words: int = 6) -> list[dict]:
    """Split plain text (blank lines and numbered markers both separate clauses)."""
    lines = [clean_text(line) for line in text.splitlines()]
    return split_lines(lines, min_words)


def read_pdf_pages(path: Path) -> list[str]:
    import pymupdf  # only needed for PDFs

    with pymupdf.open(path) as doc:
        return [page.get_text() for page in doc]


def document_clauses(path: str | Path, min_words: int = 6) -> list[dict]:
    """Clauses of a .txt or .pdf file, each with its number, text and a running id."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        clauses = split_lines(page_lines(read_pdf_pages(path)), min_words)
    elif suffix in (".txt", ".md"):
        clauses = split_clauses(path.read_text(encoding="utf-8", errors="replace"), min_words)
    else:
        raise ValueError(f"unsupported file type {suffix!r}: use .txt or .pdf")
    return [{"id": i + 1, **c} for i, c in enumerate(clauses)]
