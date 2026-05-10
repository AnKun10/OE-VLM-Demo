from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator

import yaml

from .providers.base import VLMProvider
from .providers.openai_compatible import OpenAICompatibleProvider
from .providers.qwen_vllm import QwenVLLMProvider

PROVIDER_MAP: dict[str, type[VLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
    "qwen_vllm": QwenVLLMProvider,
}


class VLMManager:
    """Loads model configs from YAML and routes generation requests."""

    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self.providers: dict[str, VLMProvider] = {}
        self.default_model: str = ""

    def load(self) -> None:
        yaml_path = Path(__file__).parent / "models.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for entry in config["models"]:
            model_id = entry["id"]
            provider_name = entry["provider"]

            if provider_name not in PROVIDER_MAP:
                print(f"[VLMManager] Unknown provider '{provider_name}' for model '{model_id}', skipping.")
                continue

            api_key_env = entry.get("api_key_env")
            if api_key_env:
                api_key = os.environ.get(api_key_env, "")
                if not api_key:
                    print(f"[VLMManager] Warning: env var '{api_key_env}' not set for model '{model_id}'.")
                    api_key = "none"
            else:
                api_key = "none"

            provider_cls = PROVIDER_MAP[provider_name]
            provider_kwargs: dict[str, Any] = {
                "base_url": entry["base_url"],
                "api_key": api_key,
                "model_id": entry["model_id"],
            }
            provider_kwargs.update(provider_cls.extra_kwargs_from_entry(entry))
            provider = provider_cls(**provider_kwargs)

            self.models[model_id] = entry
            self.providers[model_id] = provider

            if not self.default_model:
                self.default_model = model_id

        print(f"[VLMManager] Loaded {len(self.providers)} model(s): {list(self.providers.keys())}")

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": model_id,
                "name": cfg["name"],
                "capabilities": {
                    "vision": bool(cfg.get("capabilities", {}).get("vision", False)),
                },
            }
            for model_id, cfg in self.models.items()
        ]

    def _resolve(self, model_id: str | None) -> tuple[VLMProvider, dict[str, Any]]:
        resolved_id = model_id if model_id and model_id in self.providers else self.default_model
        if not resolved_id or resolved_id not in self.providers:
            raise RuntimeError("No VLM models are configured.")
        return self.providers[resolved_id], self.models[resolved_id]

    def _prepare_messages(self, config: dict[str, Any], messages: list[dict]) -> list[dict]:
        system_prompt = config.get("system_prompt", "").strip()
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + messages
        return messages

    async def stream(self, model_id: str | None, messages: list[dict]) -> AsyncIterator[str]:
        provider, config = self._resolve(model_id)
        prepared = self._prepare_messages(config, messages)
        async for delta in provider.stream(
            messages=prepared,
            max_tokens=config.get("max_tokens", 256),
            temperature=config.get("temperature", 0),
        ):
            yield delta

    async def generate(self, model_id: str | None, messages: list[dict]) -> str:
        chunks: list[str] = []
        async for delta in self.stream(model_id, messages):
            chunks.append(delta)
        return "".join(chunks).strip()

    async def stream_raw(
        self,
        model_id: str | None,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream tokens from the provider WITHOUT prepending the model's
        per-yaml system prompt. The caller supplies its own messages list
        verbatim. Used by the image compressor (caption + router calls have
        their own system prompts).
        """
        provider, _ = self._resolve(model_id)
        async for delta in provider.stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield delta

    async def generate_raw(
        self,
        model_id: str | None,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Non-streaming wrapper around `stream_raw`. Returns the joined+
        stripped text for the call."""
        chunks: list[str] = []
        async for delta in self.stream_raw(
            model_id, messages,
            max_tokens=max_tokens, temperature=temperature,
        ):
            chunks.append(delta)
        return "".join(chunks).strip()
