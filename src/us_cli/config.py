"""
Config loader — lê settings.json como o Qwen CLI faz.
Suporta apenas auth type 'local' (GGUF direto via llama-cpp-python).
"""

import json
import os
from pathlib import Path
from typing import Any


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge de dois dicionários (override ganha)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(workspace_dir: str | None = None) -> dict[str, Any]:
    """
    Carrega settings.json dos mesmos escopos que o Qwen CLI:
    1. User settings: ~/.qwen/settings.json
    2. Workspace settings: <workspace>/.qwen/settings.json
    """
    home = Path.home()
    user_settings_path = home / ".qwen" / "settings.json"

    settings = {}

    # User settings
    if user_settings_path.exists():
        with open(user_settings_path) as f:
            settings = deep_merge(settings, json.load(f))

    # Workspace settings
    if workspace_dir:
        ws_path = Path(workspace_dir) / ".qwen" / "settings.json"
        if ws_path.exists():
            with open(ws_path) as f:
                settings = deep_merge(settings, json.load(f))

    return settings


def resolve_model_config(settings: dict) -> dict:
    """
    Extrai a config do modelo dos settings.
    Suporta:
    - Ollama via openai auth (baseUrl localhost)
    - GGUF local via llama-cpp-python (model_path)
    """
    auth_type = settings.get("security", {}).get("auth", {}).get("selectedType", "")

    # Se auth é 'openai' com localhost -> usa Ollama API
    if auth_type == "openai":
        providers = settings.get("modelProviders", {}).get("openai", [])
        if providers:
            p = providers[0]
            return {
                "backend": "ollama",
                "base_url": p.get("baseUrl", "http://localhost:11434/v1"),
                "model": p.get("id", settings.get("model", {}).get("name", "")),
                "api_key": os.environ.get(p.get("envKey", ""), "ollama"),
                "context_size": p.get("generationConfig", {}).get("contextWindowSize", 4096),
                "max_tokens": p.get("generationConfig", {}).get("samplingParams", {}).get("max_tokens", 4096),
                "temperature": p.get("generationConfig", {}).get("samplingParams", {}).get("temperature", 0.7),
            }

    # Se tem model_path -> GGUF direto
    model_name = settings.get("model", {}).get("name", "")
    if model_name and model_name.endswith(".gguf"):
        return {
            "backend": "llama_cpp",
            "model_path": model_name,
            "context_size": settings.get("model", {}).get("context_size", 4096),
            "max_tokens": settings.get("model", {}).get("max_tokens", 4096),
            "temperature": settings.get("model", {}).get("temperature", 0.7),
        }

    # Fallback: tenta achar GGUF em path comum
    return {
        "backend": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model": model_name or "qwen2.5-coder:3b",
        "api_key": "ollama",
        "context_size": 4096,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
