"""Ollama LLMProvider — uses instructor with Ollama's OpenAI-compatible API."""
import sys
import structlog
import instructor
from openai import OpenAI, APIStatusError, APIConnectionError
from pydantic import BaseModel

from bsos.llm.retry import call_with_retry, NonRetryableError
from bsos.llm.cache import LLMResponseCache

log = structlog.get_logger()

DEFAULT_TIMEOUT = 120.0
DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        cache: LLMResponseCache | None = None,
    ):
        self._model = model
        self._timeout = timeout
        self._cache = cache
        raw = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout)
        self._client = instructor.from_openai(raw, mode=instructor.Mode.JSON)

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
                return self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=schema,
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
                response = self._client._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                return response.choices[0].message.content.strip()
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
