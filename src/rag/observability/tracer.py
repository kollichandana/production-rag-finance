"""Optional Langfuse tracing. No-ops if credentials aren't configured."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from loguru import logger

from rag.settings import get_settings


_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    s = get_settings()
    if not s.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
        logger.info("Langfuse tracing enabled")
        return _client
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")
        return None


@contextmanager
def trace(name: str, metadata: dict[str, Any] | None = None):
    client = _get_client()
    if client is None:
        yield None
        return
    try:
        t = client.trace(name=name, metadata=metadata or {})
        yield t
    except Exception as e:
        logger.debug(f"Tracing failed: {e}")
        yield None


def log_generation(
    trace_obj,
    name: str,
    model: str,
    input_data: Any,
    output_data: Any,
    usage: dict[str, int] | None = None,
) -> None:
    if trace_obj is None:
        return
    try:
        trace_obj.generation(
            name=name,
            model=model,
            input=input_data,
            output=output_data,
            usage=usage or {},
        )
    except Exception as e:
        logger.debug(f"log_generation failed: {e}")


def flush() -> None:
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
