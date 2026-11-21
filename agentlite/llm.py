"""Provider-agnostic LLM layer.

Agents never import a provider SDK directly. They create an `LLM` and call
`LLM.invoke(messages, tools)`, which returns an `LLMResponse` carrying the
model's message plus the metadata the trace needs (provider, model, tokens,
latency). Switching from Ollama to a cloud model is then a config change, not a
code change.

Providers:
- ollama        default, local, no key
- gemini        optional, needs GOOGLE_API_KEY (Google AI Studio free tier)
- azure_openai  optional, needs the AZURE_OPENAI_* settings (pay per token)

Cloud SDKs are imported inside _build_chat_model, so an Ollama-only setup
never needs them configured.
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

SUPPORTED_PROVIDERS: tuple[str, ...] = ("ollama", "gemini", "azure_openai")

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
        # Gemini returns a list of content blocks; .text joins the text parts for any provider.
        return self.message.text

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


def enabled_providers() -> list[str]:
    """Ollama is always on; cloud providers only when their settings are present."""
    settings = get_settings()
    providers = ["ollama"]
    if settings.google_api_key:
        providers.append("gemini")
    if settings.azure_openai_api_key and settings.azure_openai_endpoint and settings.azure_openai_deployment:
        providers.append("azure_openai")
    return providers


def _build_chat_model(provider: str, model: str, temperature: float) -> BaseChatModel:
    settings = get_settings()
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
    if provider not in enabled_providers():
        raise LLMError(f"Provider '{provider}' is not configured. Add its keys to .env (see .env.example).")

    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=settings.ollama_num_ctx,
            client_kwargs={"timeout": settings.request_timeout_s},
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            timeout=settings.request_timeout_s,
            max_retries=2,  # free-tier rate limits return 429s; retry with backoff
        )
    # azure_openai: "model" is the DEPLOYMENT name you chose in Azure AI Foundry.
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=model,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=temperature,
        timeout=settings.request_timeout_s,
        max_retries=2,
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
    """Chat models per enabled provider (embedding models are left out)."""
    settings = get_settings()
    models: dict[str, list[str]] = {"ollama": [m for m in list_ollama_models() if "embed" not in m]}
    if "gemini" in enabled_providers():
        # The configured model first, plus the cheaper Flash-Lite as an option.
        models["gemini"] = list(dict.fromkeys([settings.gemini_model, "gemini-3.5-flash-lite"]))
    if "azure_openai" in enabled_providers():
        models["azure_openai"] = [settings.azure_openai_deployment]
    return {provider: names for provider, names in models.items() if names}
