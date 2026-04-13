"""
LLM Layer — camada de inferência.

Suporta dois backends:
1. llama-cpp-python: GGUF direto no processo (sem servidor, 1 processo)
2. Ollama API: fallback via HTTP (compatibilidade com setup atual)

Sem telemetria. Sem nuvem. Sem OAuth.
"""

import json
import requests
from typing import AsyncGenerator


class Message:
    """Uma mensagem na conversa."""
    def __init__(self, role: str, content: str, tool_calls: list | None = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []

    def to_openai(self) -> dict:
        msg = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class ToolCall:
    """Uma chamada de ferramenta solicitada pelo modelo."""
    def __init__(self, id: str, name: str, arguments: dict):
        self.id = id
        self.name = name
        self.arguments = arguments

    @classmethod
    def from_openai(cls, tc: dict) -> "ToolCall":
        return cls(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=json.loads(tc["function"]["arguments"]),
        )


class LLMResponse:
    """Resposta do modelo."""
    def __init__(self, content: str, tool_calls: list[ToolCall] | None = None,
                 finish_reason: str = "stop"):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason


class LlamaCppBackend:
    """
    Backend GGUF direto via llama-cpp-python.
    1 processo. Sem servidor. Sem HTTP.
    """
    def __init__(self, model_path: str, context_size: int = 4096,
                 n_threads: int = 4, n_gpu_layers: int = 0):
        from llama_cpp import Llama

        print(f"🧠 Carregando GGUF: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )
        self.context_size = context_size
        print(f"✅ Modelo carregado ({context_size} ctx, {n_threads} threads)")

    def chat(self, messages: list[dict], system_prompt: str,
             temperature: float = 0.7, max_tokens: int = 4096,
             tools: list[dict] | None = None) -> LLMResponse:
        """
        Gera resposta via llama-cpp-python.
        Formata mensagens no estilo OpenAI chat completions.
        """
        # Monta o prompt no formato chat
        prompt = self._format_chat(messages, system_prompt, tools)

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "\n\nHuman:", "\n\nUser:"],
            echo=False,
        )

        text = output["choices"][0]["text"].strip()
        return LLMResponse(content=text)

    def _format_chat(self, messages: list[dict], system_prompt: str,
                     tools: list[dict] | None) -> str:
        """Formata mensagens para o formato de prompt do modelo."""
        prompt = f"<|system|>\n{system_prompt}\n"
        if tools:
            prompt += f"\nFerramentas disponíveis:\n{json.dumps(tools, indent=2)}\n"
        prompt += "<|end|>\n"

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue  # já injetamos acima
            elif role == "user":
                prompt += f"<|user|>\n{content}\n<|end|>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n<|end|>\n"
            elif role == "tool":
                prompt += f"<|tool|>\n{content}\n<|end|>\n"

        prompt += "<|assistant|>\n"
        return prompt


class OllamaBackend:
    """
    Backend Ollama via API HTTP localhost.
    Compatibilidade com setup atual (ollama serve separado).
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str = "qwen2.5-coder:3b", api_key: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        print(f"🔌 Conectando ao Ollama: {base_url} ({model})")

    def chat(self, messages: list[dict], system_prompt: str,
             temperature: float = 0.7, max_tokens: int = 4096,
             tools: list[dict] | None = None) -> LLMResponse:
        """Gera resposta via Ollama API OpenAI-compatible."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]
        content = choice.get("content", "")
        tool_calls = []

        if "tool_calls" in choice:
            for tc in choice["tool_calls"]:
                tool_calls.append(ToolCall.from_openai(tc))

        finish_reason = data["choices"][0].get("finish_reason", "stop")
        return LLMResponse(content=content, tool_calls=tool_calls,
                           finish_reason=finish_reason)


def create_backend(model_config: dict):
    """
    Factory: cria o backend baseado na config.
    - Se tem 'model_path' -> llama-cpp-python (GGUF direto)
    - Se tem 'base_url' -> Ollama API
    """
    if model_config.get("backend") == "llama_cpp":
        return LlamaCppBackend(
            model_path=model_config["model_path"],
            context_size=model_config.get("context_size", 4096),
            n_threads=model_config.get("n_threads", 4),
            n_gpu_layers=model_config.get("n_gpu_layers", 0),
        )
    else:
        return OllamaBackend(
            base_url=model_config.get("base_url", "http://localhost:11434/v1"),
            model=model_config.get("model", "qwen2.5-coder:3b"),
            api_key=model_config.get("api_key", "ollama"),
        )
