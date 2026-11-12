"""Generation-subsystem settings. Separate from kavach.config.Settings so Phase 1.5 stays untouched."""
from __future__ import annotations

import functools
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class GenSettings(BaseSettings):
    """Override any field with KAVACH_GEN_<FIELD> (e.g. KAVACH_GEN_NUM_CTX=3072)."""

    model_config = SettingsConfigDict(env_prefix="KAVACH_GEN_", extra="ignore")

    # Ollama endpoint and model
    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "kavach-tier5"
    base_model: str = "qwen2.5:7b-instruct-q4_K_M"
    tokenizer_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # Decoding and context budget
    num_ctx: int = 4096
    num_predict: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repeat_penalty: float = 1.0
    seed: int = 0
    keep_alive: str = "30m"
    context_token_cap: int = 1536
    prompt_safety_margin: int = 48

    # Network resilience (Full Jitter)
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 180.0
    max_attempts: int = 5
    backoff_base_s: float = 1.5
    backoff_cap_s: float = 30.0

    # VRAM guardrails (MiB, as reported by nvidia-smi memory.used)
    vram_target_mib: int = 4710
    vram_hard_limit_mib: int = 5478
    vram_poll_interval_s: float = 0.5

    # Judge
    judge_backend: Literal["auto", "gemini", "ollama"] = "auto"
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_api_key: str | None = None
    gemini_rpm: int = 15
    gemini_thinking_budget: int = 1024
    judge_num_predict: int = 768


@functools.lru_cache(maxsize=1)
def get_gen_settings() -> GenSettings:
    return GenSettings()
