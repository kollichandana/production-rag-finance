"""Streamlit frontend for the Production RAG demo.

Designed for Streamlit Community Cloud deployment:
- Runs the pipeline in-process (no separate API server needed in the simple deploy)
- Reads config from st.secrets when on Streamlit Cloud, falls back to .env locally
- Lazy-loads the pipeline so cold starts don't time out the page
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Production RAG — SEC 10-K Q&A",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve st.secrets → environment so pydantic-settings picks them up.
# Only touch st.secrets if a secrets.toml is actually present — otherwise it
# logs noisy FileNotFoundError messages. Locally we fall back to .env.
_secrets_paths = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path(__file__).parent / ".streamlit" / "secrets.toml",
]
if any(p.exists() for p in _secrets_paths):
    for key in [
        "ANTHROPIC_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_COLLECTION",
        "GENERATION_MODEL",
        "FALLBACK_MODEL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]:
        try:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = st.secrets[key]
        except Exception:
            pass


@st.cache_resource(show_spinner="Warming up retrieval & rerank models...")
def _get_pipeline():
    from rag.pipeline import get_pipeline

    return get_pipeline()


@st.cache_data(ttl=60)
def _get_stats():
    from rag.retrieval.vector_store import VectorStore

    store = VectorStore()
    chunks = store.get_all_chunks()
    if not chunks:
        return {"chunks": 0, "documents": 0, "companies": []}
    companies = sorted({c.metadata.get("company") for c in chunks if c.metadata.get("company")})
    docs = {c.doc_id for c in chunks}
    return {
        "chunks": len(chunks),
        "documents": len(docs),
        "companies": companies,
        "tables": sum(1 for c in chunks if c.chunk_type.value == "table"),
    }


# -------- Sidebar --------
with st.sidebar:
    st.title(":bar_chart: Production RAG")
    st.caption("SEC 10-K filings Q&A — hybrid retrieval + HyDE + grounded generation")

    st.divider()
    st.subheader("Corpus")
    try:
        stats = _get_stats()
        st.metric("Chunks", f"{stats['chunks']:,}")
        st.metric("Documents", stats["documents"])
        st.metric("Tables extracted", stats.get("tables", 0))
        if stats["companies"]:
            with st.expander("Companies", expanded=False):
                for c in stats["companies"]:
                    st.markdown(f"- {c}")
    except Exception as e:
        st.warning(f"Vector store unreachable: {e}")

    st.divider()
    st.subheader("Retrieval settings")
    use_hyde = st.toggle("HyDE query expansion", value=True, help="Embed a hypothetical answer for better recall")
    use_decomposition = st.toggle(
        "Decompose multi-part questions", value=False, help="Break complex Qs into sub-queries"
    )
    use_reranker = st.toggle("Cross-encoder reranker", value=True)
    use_grounding = st.toggle("Grounding check (post-hoc)", value=True)
    top_k = st.slider("Top-K chunks", min_value=3, max_value=12, value=5)

    st.divider()
    st.subheader("Filters")
    company_filter = st.multiselect(
        "Filter by company",
        options=stats.get("companies", []) if "stats" in locals() else [],
        default=[],
    )

    st.divider()
    show_naive_comparison = st.toggle(
        "Side-by-side: Naive vs Production", value=False, help="Runs the same query with naive baseline next to it"
    )

    st.divider()
    st.caption(":hammer_and_wrench: Source on [GitHub](https://github.com)")


# -------- Main --------
st.title("Ask the 10-Ks")
st.caption(
    "Hybrid retrieval (dense + BM25 + RRF) → cross-encoder rerank → "
    "contextual compression → Claude with grounded citations."
)

if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("meta"):
            with st.expander(":mag: Retrieved chunks & metadata"):
                st.json(turn["meta"])

query = st.chat_input("e.g. What were Apple's reportable segments in fiscal 2023?")
if query:
    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    pipeline = _get_pipeline()
    pipeline.cfg.use_hyde = use_hyde
    pipeline.cfg.use_decomposition = use_decomposition
    pipeline.cfg.use_reranker = use_reranker
    pipeline.cfg.use_grounding_check = use_grounding
    pipeline.cfg.top_k = top_k

    filters = {"company": company_filter} if company_filter else None

    with st.chat_message("assistant"):
        if show_naive_comparison:
            col_n, col_p = st.columns(2)
            with col_n:
                st.markdown("##### :wrench: Naive RAG")
                with st.spinner("Naive..."):
                    naive_pipeline = pipeline.__class__.naive(pipeline.retriever)
                    naive_pipeline.cfg.use_cache = False
                    t0 = time.perf_counter()
                    naive_resp = naive_pipeline.run(query, filters=filters)
                    naive_lat = (time.perf_counter() - t0) * 1000
                st.markdown(naive_resp.answer)
                st.caption(f"Latency: {naive_lat:.0f} ms · Citations: {len(naive_resp.citations)}")
            with col_p:
                st.markdown("##### :rocket: Production RAG")
                with st.spinner("Production..."):
                    t0 = time.perf_counter()
                    prod_resp = pipeline.run(query, filters=filters)
                    prod_lat = (time.perf_counter() - t0) * 1000
                st.markdown(prod_resp.answer)
                st.caption(
                    f"Latency: {prod_lat:.0f} ms · "
                    f"Citations: {len(prod_resp.citations)} · "
                    f"Faithfulness: {prod_resp.faithfulness_score or '—'} · "
                    f"Cached: {prod_resp.cached}"
                )
            response = prod_resp
        else:
            with st.spinner("Retrieving and generating..."):
                response = pipeline.run(query, filters=filters)

            st.markdown(response.answer)

            cols = st.columns(4)
            cols[0].metric("Latency", f"{response.latency_ms:.0f} ms")
            cols[1].metric("Citations", len(response.citations))
            cols[2].metric(
                "Faithfulness",
                f"{response.faithfulness_score:.2f}" if response.faithfulness_score else "—",
            )
            cols[3].metric("Cached", "yes" if response.cached else "no")

        # Citations
        if response.citations:
            st.markdown("##### Sources")
            for i, cit in enumerate(response.citations, start=1):
                with st.expander(
                    f"[{i}] {cit.company or 'Unknown'} — {cit.section or 'n/a'}"
                    + (f" (p. {cit.page})" if cit.page else "")
                ):
                    st.write(cit.snippet)

        if response.rewritten_query and response.rewritten_query != query:
            with st.expander(":pencil2: Query rewrite"):
                st.code(f"Original:  {query}\nRewritten: {response.rewritten_query}")
        if response.sub_queries:
            with st.expander(":scissors: Sub-queries"):
                for s in response.sub_queries:
                    st.markdown(f"- {s}")

        meta = {
            "model": response.model,
            "token_usage": response.token_usage,
            "retrieved_chunks": [
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "company": rc.chunk.metadata.get("company"),
                    "section": rc.chunk.section,
                    "page": rc.chunk.page,
                    "score": round(rc.score, 4),
                    "method": rc.retrieval_method,
                }
                for rc in response.retrieved_chunks
            ],
        }

        st.session_state.chat.append(
            {"role": "assistant", "content": response.answer, "meta": meta}
        )


if not st.session_state.chat:
    st.markdown("##### :bulb: Try these")
    examples = [
        "What were Apple's total net sales in fiscal 2023?",
        "What are Microsoft's three reportable operating segments?",
        "Compare R&D spending between Apple and Microsoft in their most recent fiscal year.",
        "What risk factors does Apple disclose related to supply chain concentration?",
    ]
    for e in examples:
        if st.button(e, use_container_width=True):
            st.session_state["_seed_query"] = e
            st.rerun()
