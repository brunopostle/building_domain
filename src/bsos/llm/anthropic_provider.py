"""Anthropic LLMProvider using instructor for structured output."""
import sys
import structlog
import instructor
from anthropic import Anthropic, APIStatusError, APIConnectionError
from pydantic import BaseModel

from bsos.llm.retry import call_with_retry, NonRetryableError
from bsos.llm.cache import LLMResponseCache

log = structlog.get_logger()

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache: LLMResponseCache | None = None,
    ):
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._cache = cache
        raw = Anthropic(api_key=api_key, timeout=timeout)
        self._client = instructor.from_anthropic(raw)
        self._raw = raw

    @property
    def model_id(self) -> str:
        return self._model

    def extract(self, prompt: str, schema: type[BaseModel], *, entity_name: str | None = None) -> BaseModel:
        if self._cache is not None:
            cached = self._cache.get(self._model, prompt)
            if cached is not None:
                log.debug("llm_cache_hit", model=self._model, entity=entity_name)
                return schema.model_validate(cached)

        def _call() -> BaseModel:
            try:
                return self._client.messages.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=schema,
                    max_tokens=self._max_tokens,
                )
            except APIStatusError as exc:
                if exc.status_code in {400, 401, 403}:
                    log.error("llm_non_retryable_error", status=exc.status_code, model=self._model)
                    sys.exit(f"Fatal LLM error (HTTP {exc.status_code}): {exc.message}")
                raise
            except APIConnectionError:
                raise

        try:
            result = call_with_retry(_call)
        except NonRetryableError as exc:
            sys.exit(str(exc))

        if self._cache is not None:
            self._cache.put(self._model, prompt, result.model_dump(mode="json"))

        return result

    def classify(self, prompt: str, options: list[str]) -> str:
        options_text = "\n".join(f"- {o}" for o in options)
        full_prompt = (
            f"{prompt}\n\nRespond with exactly one of these options and nothing else:\n{options_text}"
        )

        def _call() -> str:
            try:
                response = self._raw.messages.create(
                    model=self._model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=16,
                )
                return response.content[0].text.strip()
            except APIStatusError as exc:
                if exc.status_code in {400, 401, 403}:
                    log.error("llm_non_retryable_error", status=exc.status_code, model=self._model)
                    sys.exit(f"Fatal LLM error (HTTP {exc.status_code}): {exc.message}")
                raise

        try:
            text = call_with_retry(_call)
        except NonRetryableError as exc:
            sys.exit(str(exc))

        for opt in options:
            if opt.lower() == text.lower():
                return opt
        log.warning("llm_classify_unrecognised", response=text, options=options)
        return options[0]
