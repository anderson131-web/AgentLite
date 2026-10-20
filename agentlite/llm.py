"""Provider-agnostic LLM layer.

Agents never import a provider SDK directly. They create an `LLM` and call
`LLM.invoke(messages, tools)`, which returns an `LLMResponse` carrying the
model's message plus the metadata the trace needs (provider, model, tokens,
latency). Switching from Ollama to a cloud model is then a config change, not a
code change.

Providers:
- ollama        (default, local, no key)          <- Phase 1
- azure_openai  (optional, needs keys in .env)     <- Phase 9
- gemini        (optional, needs key in .env)      <- Phase 8
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from agentlite.config import get_settings

SUPPORTED_PROVIDERS: tuple[str, ...] = ("ollama",)

# Models this project is tuned and evaluated for. Other pulled models still work.
RECOMMENDED_OLLAMA_MODELS: tuple[str, ...] = (
    "qwen2.5:3b",   # default: fits fully in 4 GB VRAM, fast
    "qwen2.5:7b",   # better quality, partially offloads to CPU, slower
    "llama3.2:3b",
    "qwen3:4b",
)


class LLMError(RuntimeError):
    """Raised when a model can't be built or called. Message is safe to show users."""


@dataclass
class LLMResponse:
    message: AIMessage
    provider: str
    model: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def content(self) -> str:
        # Some providers return a list of content parts; we only care about text.
        content = self.message.content
        return content if isinstance(content, str) else str(content)

    @property
    def tool_calls(self) -> list[dict]:
        return list(self.message.tool_calls or [])


class LLM:
    """A chat model plus the provider/model names, behind one small interface."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = provider or settings.provider
        self.model = model or settings.model
        self.temperature = settings.temperature if temperature is None else temperature
        self._chat = _build_chat_model(self.provider, self.model, self.temperature)

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
    ) -> LLMResponse:
        """Send messages (and optionally tool schemas) to the model and time the call."""
        runnable = self._chat.bind_tools(list(tools)) if tools else self._chat
        start = time.perf_counter()
        try:
            message = runnable.invoke(list(messages))
        except Exception as exc:  # network errors, timeouts, provider errors
            raise LLMError(f"{self.provider}/{self.model} call failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = message.usage_metadata or {}
        return LLMResponse(
            message=message,
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


def _build_chat_model(provider: str, model: str, temperature: float) -> BaseChatModel:
    settings = get_settings()
    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=settings.ollama_num_ctx,
            client_kwargs={"timeout": settings.request_timeout_s},
        )
    raise LLMError(
        f"Unknown provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


# --- Ollama helpers -----------------------------------------------------------


def list_ollama_models(base_url: str | None = None) -> list[str]:
    """Names of models pulled into the local Ollama server. Empty list if unreachable."""
    url = (base_url or get_settings().ollama_base_url).rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=3)
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    return sorted(m["name"] for m in response.json().get("models", []))


def check_ollama_model(model: str, base_url: str | None = None) -> None:
    """Fail early with a helpful message instead of a stack trace mid-run."""
    base_url = base_url or get_settings().ollama_base_url
    pulled = list_ollama_models(base_url)
    if not pulled:
        raise LLMError(
            f"Can't reach Ollama at {base_url} (or no models are pulled). "
            "Start the Ollama app, then run: ollama pull qwen2.5:3b"
        )
    # "qwen2.5:3b" matches "qwen2.5:3b"; "llama3.2" matches "llama3.2:latest".
    if model not in pulled and f"{model}:latest" not in pulled:
        raise LLMError(
            f"Model '{model}' is not pulled. Run: ollama pull {model}  "
            f"(pulled: {', '.join(pulled)})"
        )


def available_models() -> dict[str, list[str]]:
    """Models per enabled provider. Used by the CLI now and the API's GET /models later."""
    return {"ollama": list_ollama_models()}
