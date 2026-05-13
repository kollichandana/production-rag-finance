"""RRF: ordering changes when fusing multiple ranked lists."""
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.schemas import Chunk, ChunkType, RetrievedChunk


def _make(chunk_id: str, score: float, method: str = "dense") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d", text="t", chunk_type=ChunkType.TEXT),
        score=score,
        retrieval_method=method,
    )


def test_rrf_promotes_items_present_in_both_lists():
    dense = [_make("A", 0.9), _make("B", 0.8), _make("C", 0.7)]
    sparse = [_make("B", 3.0, "sparse"), _make("A", 2.0, "sparse"), _make("D", 1.0, "sparse")]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    # A and B appear in both — they should rank above C and D
    ids = [r.chunk.chunk_id for r in fused]
    assert ids[0] in {"A", "B"}
    assert ids[1] in {"A", "B"}
    assert set(ids[:2]) == {"A", "B"}


def test_rrf_handles_single_list():
    only = [_make("X", 0.9), _make("Y", 0.5)]
    fused = reciprocal_rank_fusion(only, k=60)
    assert [r.chunk.chunk_id for r in fused] == ["X", "Y"]
