"""PDF parser for 10-K filings. Extracts text + tables + section structure."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from loguru import logger


SECTION_PATTERNS = [
    (r"^\s*ITEM\s+(\d+[A-Z]?)\.?\s+(.+?)$", "item"),
    (r"^\s*PART\s+([IVX]+)\.?\s*(.*)$", "part"),
]


@dataclass
class PageContent:
    page_num: int
    text: str
    tables: list[list[list[str]]]


@dataclass
class ParsedDocument:
    pages: list[PageContent]
    sections: dict[int, str]  # page_num -> section title

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def _detect_section(line: str) -> str | None:
    for pattern, kind in SECTION_PATTERNS:
        m = re.match(pattern, line.strip(), re.IGNORECASE)
        if m:
            if kind == "item":
                return f"Item {m.group(1)}. {m.group(2).strip()}"[:120]
            return f"Part {m.group(1)} {m.group(2).strip()}".strip()[:120]
    return None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_pdf(path: str | Path) -> ParsedDocument:
    """Parse a 10-K PDF, returning per-page text + extracted tables + section map."""
    path = Path(path)
    pages: list[PageContent] = []
    sections: dict[int, str] = {}
    current_section: str | None = None

    logger.info(f"Parsing {path.name}")
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            for line in raw_text.splitlines()[:10]:
                detected = _detect_section(line)
                if detected:
                    current_section = detected
                    break
            if current_section:
                sections[idx] = current_section

            tables = []
            try:
                for tbl in page.extract_tables() or []:
                    cleaned = [[(c or "").strip() for c in row] for row in tbl if any(row)]
                    if cleaned and len(cleaned) > 1:
                        tables.append(cleaned)
            except Exception as e:
                logger.debug(f"Table extraction failed on page {idx}: {e}")

            pages.append(PageContent(page_num=idx, text=_clean_text(raw_text), tables=tables))

    logger.info(f"Parsed {len(pages)} pages, {sum(len(p.tables) for p in pages)} tables")
    return ParsedDocument(pages=pages, sections=sections)


def table_to_markdown(table: list[list[str]]) -> str:
    """Convert extracted table to markdown for LLM-friendly representation."""
    if not table or len(table) < 2:
        return ""
    header = table[0]
    rows = table[1:]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        normalized = row + [""] * (len(header) - len(row))
        md += "| " + " | ".join(normalized[: len(header)]) + " |\n"
    return md
