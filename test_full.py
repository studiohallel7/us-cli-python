#!/usr/bin/env python3
"""Teste completo: modelo + memoria + tools."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from us_cli.memory import load_context
from us_cli.prompt import build_system_prompt
from us_cli.llm import create_backend
from us_cli.tools.registry import create_tool_registry

MODEL = "models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"

def main():
    print("Us CLI - Teste completo")
    print()

    backend = create_backend({
        "backend": "llama_cpp",
        "model_path": MODEL,
        "context_size": 4096,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "max_tokens": 256,
        "temperature": 0.7,
    })

    user_memory = load_context()
    system_prompt = build_system_prompt(user_memory)
    tools = create_tool_registry()
    tool_schemas = [t.to_openai_schema() for t in tools.values()]

    print(f"Ferramentas: {len(tools)}")
    print(f"System prompt: {len(system_prompt)} chars")
    print()

    # Test 1: conversa simples
    print("=== Teste 1: Quem e voce? ===")
    messages = [{"role": "user", "content": "Oi, quem e voce? Responda em 2 frases em portugues."}]
    resp = backend.chat(messages, system_prompt, max_tokens=256, tools=tool_schemas)
    print(f"Resposta: {resp.content}")
    print()

    # Test 2: tool calling
    print("=== Teste 2: Tool calling ===")
    messages2 = [{"role": "user", "content": "Leia o arquivo /home/julios/.qwen/source.json e me diga o conteudo."}]
    resp2 = backend.chat(messages2, system_prompt, max_tokens=256, tools=tool_schemas)
    if resp2.tool_calls:
        for tc in resp2.tool_calls:
            print(f"Tool: {tc.name}({tc.arguments})")
            tool = tools.get(tc.name)
            if tool:
                result = tool.execute(tc.arguments)
                print(f"Resultado: {result[:300]}")
    else:
        print(f"Resposta direta: {resp2.content}")

if __name__ == "__main__":
    main()
