"""Anthropic client wrapper with prompt caching, fallback chain, and retry."""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from anthropic import Anthropic, APIError, APIStatusError, RateLimitError
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag.settings import get_settings


class LLMClient:
    """Thin wrapper around the Anthropic SDK.

    Key features:
    - Prompt caching on system prompt (huge win for repeated context like
      retrieval guidelines that don't change between requests)
    - Automatic fallback to a cheaper model on rate limit / 5xx
    - Bounded retries with exponential backoff
    """

    def __init__(self) -> None:
        s = get_settings()
        if not s.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY is empty — LLM calls will fail")
        self.client = Anthropic(api_key=s.anthropic_api_key)
        self.primary_model = s.generation_model
        self.fallback_model = s.fallback_model
        self.default_max_tokens = s.max_output_tokens
        self.default_temperature = s.generation_temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        reraise=True,
    )
    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool = True,
    ) -> dict[str, Any]:
        """Return {'text', 'input_tokens', 'output_tokens', 'cache_read', 'cache_creation', 'model'}."""
        model = model or self.primary_model
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        system_param: Any = None
        if isinstance(system, str) and system:
            if cache_system and len(system) > 1024:
                system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            else:
                system_param = system
        elif isinstance(system, list):
            system_param = system

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            if system_param is not None:
                kwargs["system"] = system_param
            resp = self.client.messages.create(**kwargs)
        except APIStatusError as e:
            if e.status_code in (500, 503, 529) and model == self.primary_model:
                logger.warning(f"{model} returned {e.status_code}, falling back to {self.fallback_model}")
                return self.complete(
                    messages=messages,
                    system=system,
                    model=self.fallback_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    cache_system=cache_system,
                )
            raise

        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        usage = resp.usage
        return {
            "text": text,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "model": model,
            "stop_reason": resp.stop_reason,
        }

    def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterable[str]:
        model = model or self.primary_model
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        system_param: Any = None
        if isinstance(system, str) and system:
            if len(system) > 1024:
                system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            else:
                system_param = system

        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            system=system_param,
        ) as stream:
            yield from stream.text_stream


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return LLMClient()
