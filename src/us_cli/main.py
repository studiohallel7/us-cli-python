"""
Entry point do Us CLI.

Uso:
    us                  # roda com settings padrão
    us --cwd /path      # roda num diretório específico
    us --model path.gguf # roda GGUF direto (llama-cpp-python)
"""

import argparse
import sys
import os

from .config import load_settings, resolve_model_config
from .loop import run_loop


def main():
    parser = argparse.ArgumentParser(description="Us CLI — Soberania Local")
    parser.add_argument("--cwd", type=str, default=None,
                        help="Diretório de trabalho")
    parser.add_argument("--model", type=str, default=None,
                        help="Caminho para modelo GGUF (llama-cpp-python)")
    parser.add_argument("--ollama", type=str, default=None,
                        help="URL do Ollama (ex: http://localhost:11434/v1)")
    parser.add_argument("--ollama-model", type=str, default=None,
                        help="Nome do modelo Ollama")
    parser.add_argument("--ctx", type=int, default=4096,
                        help="Tamanho do contexto")
    parser.add_argument("--threads", type=int, default=4,
                        help="Número de threads (para GGUF)")

    args = parser.parse_args()

    cwd = args.cwd or os.getcwd()

    # Se tem --model -> GGUF direto
    if args.model:
        model_config = {
            "backend": "llama_cpp",
            "model_path": args.model,
            "context_size": args.ctx,
            "n_threads": args.threads,
            "n_gpu_layers": 0,  # sem GPU no teu hardware
            "max_tokens": 4096,
            "temperature": 0.7,
        }
    elif args.ollama:
        model_config = {
            "backend": "ollama",
            "base_url": args.ollama,
            "model": args.ollama_model or "qwen2.5-coder:3b",
            "api_key": "ollama",
            "context_size": args.ctx,
            "max_tokens": 4096,
            "temperature": 0.7,
        }
    else:
        # Tenta carregar settings.json
        try:
            settings = load_settings(cwd)
            model_config = resolve_model_config(settings)
        except Exception as e:
            print(f"⚠️  Erro carregando settings: {e}")
            print("Usando fallback Ollama...")
            model_config = {
                "backend": "ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "qwen2.5-coder:3b",
                "api_key": "ollama",
                "context_size": 4096,
                "max_tokens": 4096,
                "temperature": 0.7,
            }

    # Override context_size se especificado
    model_config["context_size"] = args.ctx

    run_loop(model_config, cwd)


if __name__ == "__main__":
    main()
