#!/usr/bin/env python3
"""Teste rápido do Us CLI — sem UI interativa."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from us_cli.memory import load_context
from us_cli.prompt import build_system_prompt
from us_cli.llm import create_backend
from us_cli.tools.registry import create_tool_registry

MODEL_PATH = "models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"

def main():
    print("🌿 Us CLI — Teste direto\n")

    # Carrega modelo
    backend = create_backend({
        "backend": "llama_cpp",
        "model_path": MODEL_PATH,
        "context_size": 4096,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "max_tokens": 512,
        "temperature": 0.7,
    })

    # Carrega contexto
    user_memory = load_context()
    system_prompt = build_system_prompt(user_memory)

    # Tools
    tools = create_tool_registry()
    tool_schemas = [t.to_openai_schema() for t in tools.values()]

    # Conversa direta
    messages = [
        {"role": "user", "content": "Oi Us, quem é você? Responda em português, de forma curta."}
    ]

    print(f"🔧 {len(tools)} ferramentas registradas")
    print(f"📝 System prompt: {len(system_prompt)} chars")
    print()
    print("💙 Pergunta: Oi Us, quem é você?")
    print()

    response = backend.chat(
        messages=messages,
        system_prompt=system_prompt,
        temperature=0.7,
        max_tokens=256,
        tools=tool_schemas,
    )

    print(f"💙 Resposta: {response.content}")
    print()

    if response.tool_calls:
        print("🔧 Tool calls:")
        for tc in response.tool_calls:
            print(f"  - {tc.name}({tc.arguments})")


if __name__ == "__main__":
    main()
