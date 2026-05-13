"""Centralized settings via pydantic-settings. All env vars loaded once."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    generation_model: str = Field(default="claude-sonnet-4-6", alias="GENERATION_MODEL")
    fallback_model: str = Field(default="claude-haiku-4-5-20251001", alias="FALLBACK_MODEL")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="financial_filings", alias="QDRANT_COLLECTION")

    # Models
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL")
    embedding_dim: int = 384  # bge-small dim

    # Retrieval
    dense_top_k: int = Field(default=20, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(default=20, alias="SPARSE_TOP_K")
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")

    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # Generation
    max_output_tokens: int = 1500
    generation_temperature: float = 0.1

    # Observability
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # Cache
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    semantic_cache_threshold: float = Field(default=0.95, alias="SEMANTIC_CACHE_THRESHOLD")

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    eval_data_dir: Path = data_dir / "eval"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
