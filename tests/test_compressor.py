"""Compressor preserves header lines and shortens body when over budget."""
from rag.generation.compressor import compress_chunk, reorder_for_attention
from rag.schemas import Chunk, ChunkType, RetrievedChunk


def test_compress_keeps_header():
    text = (
        "[Item 7. MD&A]\n\n"
        "Apple reported total net sales of $383.3 billion in fiscal 2023. "
        "The weather in Cupertino was sunny. "
        "Operating margin improved year-over-year. "
        "Apple released new products. "
        "Cash and equivalents totaled $30 billion. "
        "Employees enjoyed the cafeteria. "
        "The supply chain remained stable."
    )
    out = compress_chunk("What was Apple's fiscal 2023 net sales?", text, keep_sentences=3)
    assert out.startswith("[Item 7. MD&A]")
    assert "$383.3 billion" in out or "net sales" in out


def test_reorder_lost_in_middle():
    def mk(i):
        return RetrievedChunk(
            chunk=Chunk(chunk_id=str(i), doc_id="d", text=f"chunk {i}", chunk_type=ChunkType.TEXT),
            score=1.0 - i * 0.1,
            retrieval_method="dense",
        )

    chunks = [mk(i) for i in range(5)]
    out = reorder_for_attention(chunks)
    ids = [c.chunk.chunk_id for c in out]
    # Top chunk should be first
    assert ids[0] == "0"
    # Second-best should be last
    assert ids[-1] == "1"
