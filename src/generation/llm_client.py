"""Async Ollama LLM client via httpx."""

import json
from typing import Optional

import httpx

from src.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.1:8b"
_GENERATE_PATH = "/api/generate"


class OllamaClient:
    """Asynchronous client for the Ollama REST API.

    Reads ``OLLAMA_HOST`` from the environment (via the loaded config) and
    calls ``POST /api/generate`` for single-turn completions.

    Args:
        host: Base URL for the Ollama server (default ``"http://localhost:11434"``).
        model: Model name to use for generation (default ``"llama3.1:8b"``).
        timeout: HTTP request timeout in seconds (default 120).
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        model: str = _DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        host = host.rstrip("/")
        # Guard against unresolved ${OLLAMA_HOST} placeholder or bare hostnames.
        if not host.startswith(("http://", "https://")):
            log.warning(
                "OllamaClient: host '{}' has no scheme — prepending http://", host
            )
            host = "http://" + host
        self.host = host
        self.model = model
        self.timeout = timeout
        log.debug("OllamaClient: host={} model={}", self.host, self.model)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system: Optional[str] = None,
    ) -> str:
        """Send *prompt* to Ollama and return the generated text.

        Args:
            prompt: User prompt string.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            system: Optional system prompt prepended to the conversation.

        Returns:
            Generated text stripped of leading/trailing whitespace.

        Raises:
            httpx.ConnectError: If Ollama is not reachable.
            httpx.TimeoutException: If the request exceeds *timeout* seconds.
            RuntimeError: If Ollama returns a non-200 status code.
        """
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system

        url = self.host + _GENERATE_PATH
        log.debug("OllamaClient.generate: POST {} model={}", url, self.model)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
            except httpx.ConnectError as exc:
                log.error("OllamaClient: cannot connect to Ollama at '{}'", self.host)
                raise httpx.ConnectError(
                    f"Cannot connect to Ollama at '{self.host}'. "
                    "Is it running?  Set OLLAMA_HOST if using a non-default address."
                ) from exc
            except httpx.TimeoutException as exc:
                log.error("OllamaClient: request timed out after {}s", self.timeout)
                raise

        if response.status_code != 200:
            body = response.text[:500]
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {body}"
            )

        data = response.json()
        text: str = data.get("response", "")
        log.debug(
            "OllamaClient.generate: received {} chars",
            len(text),
        )
        return text.strip()

    @classmethod
    def from_config(cls, cfg: dict) -> "OllamaClient":
        """Construct an :class:`OllamaClient` from the loaded YAML config dict.

        Expects a ``cfg["llm"]`` section with optional keys ``host`` and
        ``model``.

        Args:
            cfg: Top-level config dict (as returned by
                 :func:`src.utils.config.load_config`).

        Returns:
            Configured :class:`OllamaClient` instance.
        """
        llm_cfg = cfg.get("llm", {})
        host = llm_cfg.get("host", _DEFAULT_HOST)
        model = llm_cfg.get("model", _DEFAULT_MODEL)
        return cls(host=host, model=model)
