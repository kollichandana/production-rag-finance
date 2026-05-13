"""Hierarchical + semantic chunker tuned for financial filings.

Strategy:
1. Split on section boundaries (Item N.) first — these are semantic units.
2. Within sections, split on paragraph boundaries.
3. Pack paragraphs into ~target-token chunks with overlap.
4. Tables become their own chunks (with section context prepended).

We avoid pure-embedding semantic chunking here because it's expensive at
ingestion and the structural signals in 10-Ks are already strong.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import tiktoken

from rag.schemas import Chunk, ChunkType


@dataclass
class ChunkerConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 64
    max_tokens: int = 768


class SemanticChunker:
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.cfg = config or ChunkerConfig()
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text, disallowed_special=()))

    def _split_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _pack_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Greedy packing with paragraph-level overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if para_tokens > self.cfg.max_tokens:
                # Sentence-split a giant paragraph
                for piece in self._split_oversized(para):
                    chunks.append(piece)
                continue

            if current_tokens + para_tokens > self.cfg.target_tokens and current:
                chunks.append("\n\n".join(current))
                # overlap: keep the tail paragraph(s) until we reach overlap_tokens
                overlap: list[str] = []
                overlap_tokens = 0
                for p in reversed(current):
                    t = self._count_tokens(p)
                    if overlap_tokens + t > self.cfg.overlap_tokens:
                        break
                    overlap.insert(0, p)
                    overlap_tokens += t
                current = overlap
                current_tokens = overlap_tokens

            current.append(para)
            current_tokens += para_tokens

        if current and current_tokens >= self.cfg.min_tokens:
            chunks.append("\n\n".join(current))
        elif current and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(current)
        elif current:
            chunks.append("\n\n".join(current))

        return chunks

    def _split_oversized(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sent in sentences:
            t = self._count_tokens(sent)
            if current_tokens + t > self.cfg.target_tokens and current:
                result.append(" ".join(current))
                current = [sent]
                current_tokens = t
            else:
                current.append(sent)
                current_tokens += t
        if current:
            result.append(" ".join(current))
        return result

    def chunk_section(
        self,
        text: str,
        doc_id: str,
        section: str | None = None,
        page: int | None = None,
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        packed = self._pack_paragraphs(paragraphs)

        chunks: list[Chunk] = []
        for i, body in enumerate(packed):
            # Prepend section header as light context boost
            contextualized = f"[{section}]\n\n{body}" if section else body
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::sec{i}::{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    text=contextualized,
                    chunk_type=ChunkType.TEXT,
                    section=section,
                    page=page,
                    token_count=self._count_tokens(contextualized),
                    metadata={"position": i},
                )
            )
        return chunks

    def chunk_table(
        self,
        table_md: str,
        doc_id: str,
        section: str | None = None,
        page: int | None = None,
    ) -> Chunk:
        contextualized = f"[Table from {section}]\n\n{table_md}" if section else table_md
        return Chunk(
            chunk_id=f"{doc_id}::tbl::{uuid.uuid4().hex[:8]}",
            doc_id=doc_id,
            text=contextualized,
            chunk_type=ChunkType.TABLE,
            section=section,
            page=page,
            token_count=self._count_tokens(contextualized),
            metadata={"is_table": True},
        )
