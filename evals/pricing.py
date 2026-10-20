"""Token prices used to estimate cost, in USD per 1 million tokens (input, output).

Local Ollama models cost $0 per token (you pay in electricity and hardware).
Hosted prices change; check the provider's pricing page before quoting numbers.
Free-tier usage is still reported at list price, so the comparison is fair.
"""

from __future__ import annotations

# key: "provider/model"; a "provider/*" entry is the fallback for that provider.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "ollama/*": (0.0, 0.0),
}


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES_PER_MILLION.get(f"{provider}/{model}") or PRICES_PER_MILLION.get(f"{provider}/*")
    if price is None:
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
