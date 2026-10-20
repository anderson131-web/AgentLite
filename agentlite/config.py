"""Central configuration.

Every setting comes from environment variables (optionally loaded from a .env
file in the project root). Nothing here requires a paid API key: the defaults
point at a local Ollama server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# On a 4 GB GPU the KV cache grows with the context window. Above ~8k tokens a
# 3B model starts spilling out of VRAM and gets much slower, so we cap it here
# no matter what the .env says.
MAX_NUM_CTX = 8192


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    temperature: float
    ollama_base_url: str
    ollama_num_ctx: int
    request_timeout_s: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read settings once and cache them (call get_settings.cache_clear() in tests)."""
    return Settings(
        provider=os.getenv("LLM_PROVIDER", "ollama"),
        model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
        # 0 = deterministic-ish output, which small models need to follow tool formats.
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_num_ctx=min(int(os.getenv("OLLAMA_NUM_CTX", str(MAX_NUM_CTX))), MAX_NUM_CTX),
        request_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "120")),
    )
