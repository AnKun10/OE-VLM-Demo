from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .providers.base import VLMProvider
from .providers.openai_compatible import OpenAICompatibleProvider

# Maps provider name in YAML -> provider class
PROVIDER_MAP: dict[str, type[VLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
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

            # Resolve API key from environment
            api_key_env = entry.get("api_key_env")
            if api_key_env:
                api_key = os.environ.get(api_key_env, "")
                if not api_key:
                    print(f"[VLMManager] Warning: env var '{api_key_env}' not set for model '{model_id}'.")
                    api_key = "none"
            else:
                api_key = "none"

            provider_cls = PROVIDER_MAP[provider_name]
            provider = provider_cls(
                base_url=entry["base_url"],
                api_key=api_key,
                model_id=entry["model_id"],
            )

            self.models[model_id] = entry
            self.providers[model_id] = provider

            if not self.default_model:
                self.default_model = model_id

        print(f"[VLMManager] Loaded {len(self.providers)} model(s): {list(self.providers.keys())}")

    def list_models(self) -> list[dict[str, str]]:
        return [
            {"id": model_id, "name": cfg["name"]}
            for model_id, cfg in self.models.items()
        ]

    def generate(self, model_id: str | None, messages: list[dict]) -> str:
        resolved_id = model_id if model_id and model_id in self.providers else self.default_model

        if not resolved_id or resolved_id not in self.providers:
            raise RuntimeError("No VLM models are configured.")

        config = self.models[resolved_id]
        provider = self.providers[resolved_id]

        # Prepend system prompt
        system_prompt = config.get("system_prompt", "").strip()
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        return provider.generate(
            messages=messages,
            max_tokens=config.get("max_tokens", 256),
            temperature=config.get("temperature", 0),
        )
