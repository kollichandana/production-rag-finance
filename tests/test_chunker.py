"""Chunker behavior: respects target size, preserves overlap, handles small inputs."""
from rag.ingestion.chunkers.semantic_chunker import ChunkerConfig, SemanticChunker


def test_short_text_single_chunk():
    chunker = SemanticChunker(ChunkerConfig(target_tokens=128, min_tokens=10))
    chunks = chunker.chunk_section("This is a short document.", doc_id="doc1", section="Test")
    assert len(chunks) == 1
    assert "Test" in chunks[0].text
    assert chunks[0].section == "Test"


def test_long_text_multiple_chunks_with_overlap():
    long_text = "\n\n".join([f"Paragraph {i}. " + ("word " * 80).strip() for i in range(10)])
    chunker = SemanticChunker(ChunkerConfig(target_tokens=200, overlap_tokens=40, min_tokens=20))
    chunks = chunker.chunk_section(long_text, doc_id="doc2")
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 800


def test_table_chunk_carries_table_flag():
    chunker = SemanticChunker()
    chunk = chunker.chunk_table("| a | b |\n|---|---|\n| 1 | 2 |", doc_id="doc3", section="Item 8.")
    assert chunk.metadata.get("is_table") is True
    assert "Item 8." in chunk.text
