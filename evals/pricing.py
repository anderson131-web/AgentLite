"""Token prices used to estimate cost, in USD per 1 million tokens (input, output).

Local Ollama models cost $0 per token (you pay in electricity and hardware).
Hosted prices change; check the provider's pricing page before quoting numbers.
Free-tier usage is still reported at list price, so the comparison is fair.
"""

from __future__ import annotations

# key: "provider/model"; a "provider/*" entry is the fallback for that provider.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "ollama/*": (0.0, 0.0),
    # Google's pricing page, checked 2026-10-20. 3.8 Flash doubles on 2027-01-01 ($1.50 / $7.50).
    "gemini/gemini-3.8-flash": (0.75, 3.75),
    "gemini/gemini-3.5-flash-lite": (0.30, 2.50),
    # Third-party listings for gpt-4.1-nano (Microsoft's page is region-specific); verify
    # on the Azure pricing calculator. Azure "models" are deployment names, so this
    # fallback assumes your deployment runs gpt-4.1-nano.
    "azure_openai/*": (0.10, 0.40),
}


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES_PER_MILLION.get(f"{provider}/{model}") or PRICES_PER_MILLION.get(f"{provider}/*")
    if price is None:
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
