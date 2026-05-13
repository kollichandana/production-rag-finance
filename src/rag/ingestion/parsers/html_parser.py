"""HTML parser for SEC EDGAR filings (which are typically served as HTML)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class HtmlSection:
    title: str
    text: str
    tables_markdown: list[str]


def _table_to_md(table_tag) -> str:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [re.sub(r"\s+", " ", td.get_text(strip=True)) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    if not rows or len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    md = "| " + " | ".join(rows[0]) + " |\n"
    md += "| " + " | ".join(["---"] * width) + " |\n"
    for r in rows[1:]:
        md += "| " + " | ".join(r) + " |\n"
    return md


def parse_html(path: str | Path) -> list[HtmlSection]:
    """Parse an EDGAR HTML filing into sections keyed by Item headings."""
    path = Path(path)
    logger.info(f"Parsing HTML {path.name}")
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")

    for s in soup(["script", "style"]):
        s.decompose()

    sections: list[HtmlSection] = []
    current_title = "Front Matter"
    current_text: list[str] = []
    current_tables: list[str] = []

    for el in soup.body.descendants if soup.body else []:
        if getattr(el, "name", None) in {"h1", "h2", "h3", "h4", "b", "strong"}:
            text = el.get_text(strip=True)
            if re.match(r"^(ITEM|PART)\s+", text, re.IGNORECASE) and len(text) < 200:
                if current_text or current_tables:
                    sections.append(
                        HtmlSection(
                            title=current_title,
                            text="\n".join(current_text).strip(),
                            tables_markdown=current_tables,
                        )
                    )
                current_title = text[:120]
                current_text = []
                current_tables = []
                continue
        if getattr(el, "name", None) == "table":
            md = _table_to_md(el)
            if md:
                current_tables.append(md)
        if getattr(el, "name", None) == "p":
            txt = el.get_text(" ", strip=True)
            if txt:
                current_text.append(txt)

    if current_text or current_tables:
        sections.append(
            HtmlSection(
                title=current_title,
                text="\n".join(current_text).strip(),
                tables_markdown=current_tables,
            )
        )

    sections = [s for s in sections if s.text or s.tables_markdown]
    logger.info(f"Extracted {len(sections)} sections from {path.name}")
    return sections
