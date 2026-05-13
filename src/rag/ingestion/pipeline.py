"""End-to-end ingestion: parse → chunk → embed → upsert into Qdrant."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from rag.ingestion.chunkers.semantic_chunker import ChunkerConfig, SemanticChunker
from rag.ingestion.parsers.html_parser import parse_html
from rag.ingestion.parsers.pdf_parser import parse_pdf, table_to_markdown
from rag.retrieval.vector_store import VectorStore
from rag.schemas import Chunk, DocumentMetadata, DocumentType
from rag.settings import get_settings


def _doc_id_for(path: Path) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem.replace(' ', '_')}_{h}"


def _infer_metadata(path: Path) -> DocumentMetadata:
    """Infer metadata from filename. Expected pattern: TICKER_COMPANY_YEAR_10K.{pdf,html}"""
    stem = path.stem
    parts = stem.split("_")
    ticker = parts[0].upper() if parts else None
    company = parts[1] if len(parts) > 1 else stem
    fiscal_year = None
    for p in parts:
        if p.isdigit() and len(p) == 4:
            fiscal_year = int(p)
            break
    return DocumentMetadata(
        doc_id=_doc_id_for(path),
        company=company,
        ticker=ticker,
        filing_type=DocumentType.FILING_10K,
        fiscal_year=fiscal_year,
        source_path=str(path),
    )


def _chunk_pdf(path: Path, meta: DocumentMetadata, chunker: SemanticChunker) -> list[Chunk]:
    parsed = parse_pdf(path)
    chunks: list[Chunk] = []
    current_section = None
    section_buffer: list[str] = []
    section_page = None

    for page in parsed.pages:
        page_section = parsed.sections.get(page.page_num)
        if page_section and page_section != current_section:
            if section_buffer and current_section:
                chunks.extend(
                    chunker.chunk_section(
                        "\n\n".join(section_buffer),
                        doc_id=meta.doc_id,
                        section=current_section,
                        page=section_page,
                    )
                )
            current_section = page_section
            section_page = page.page_num
            section_buffer = [page.text]
        else:
            section_buffer.append(page.text)

        for tbl in page.tables:
            md = table_to_markdown(tbl)
            if md:
                chunks.append(
                    chunker.chunk_table(md, doc_id=meta.doc_id, section=current_section, page=page.page_num)
                )

    if section_buffer:
        chunks.extend(
            chunker.chunk_section(
                "\n\n".join(section_buffer),
                doc_id=meta.doc_id,
                section=current_section or "Document",
                page=section_page,
            )
        )

    for c in chunks:
        c.metadata.update(
            {
                "company": meta.company,
                "ticker": meta.ticker,
                "fiscal_year": meta.fiscal_year,
                "filing_type": meta.filing_type.value,
            }
        )
    return chunks


def _chunk_html(path: Path, meta: DocumentMetadata, chunker: SemanticChunker) -> list[Chunk]:
    sections = parse_html(path)
    chunks: list[Chunk] = []
    for sec in sections:
        chunks.extend(
            chunker.chunk_section(sec.text, doc_id=meta.doc_id, section=sec.title)
        )
        for tbl_md in sec.tables_markdown:
            chunks.append(chunker.chunk_table(tbl_md, doc_id=meta.doc_id, section=sec.title))
    for c in chunks:
        c.metadata.update(
            {
                "company": meta.company,
                "ticker": meta.ticker,
                "fiscal_year": meta.fiscal_year,
                "filing_type": meta.filing_type.value,
            }
        )
    return chunks


def ingest_path(input_path: Path, store: VectorStore | None = None) -> dict:
    """Ingest all PDFs/HTML under a directory or a single file."""
    settings = get_settings()
    chunker = SemanticChunker(
        ChunkerConfig(
            target_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
    )
    store = store or VectorStore()
    store.ensure_collection()

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            [p for p in input_path.rglob("*") if p.suffix.lower() in {".pdf", ".html", ".htm"}]
        )

    if not files:
        logger.warning(f"No PDF/HTML files under {input_path}")
        return {"docs": 0, "chunks": 0}

    summary = {"docs": 0, "chunks": 0, "files": []}
    all_chunks: list[Chunk] = []

    for path in tqdm(files, desc="Documents"):
        meta = _infer_metadata(path)
        try:
            if path.suffix.lower() == ".pdf":
                chunks = _chunk_pdf(path, meta, chunker)
            else:
                chunks = _chunk_html(path, meta, chunker)
        except Exception as e:
            logger.exception(f"Failed to ingest {path}: {e}")
            continue

        logger.info(f"{path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
        summary["docs"] += 1
        summary["files"].append({"path": str(path), "chunks": len(chunks)})

        # Persist parsed chunks for debugging / eval reuse
        out_path = settings.processed_data_dir / f"{meta.doc_id}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for c in chunks:
                f.write(json.dumps(c.model_dump()) + "\n")

    if all_chunks:
        logger.info(f"Upserting {len(all_chunks)} chunks to Qdrant")
        store.upsert_chunks(all_chunks)

    summary["chunks"] = len(all_chunks)
    return summary
