# us-cli-python

Qwen CLI portado para Python — soberania local, sem telemetria, sem nuvem.

## Arquitetura

```
Processo único Python:
├── llama-cpp-python (GGUF direto na RAM) — ou Ollama API como fallback
├── Memory loader (QWEN.md, memory/, context files)
├── Prompt builder (system prompt + user memory)
├── Tool registry (read_file, write_file, edit, shell, glob, grep, etc.)
├── Conversation loop (input → LLM → tool calls → results → repeat)
└── Zero telemetria, zero OAuth, zero nuvem
```

## Instalação

```bash
# Cria venv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Install llama-cpp-python (build from source)
CMAKE_ARGS="-DLLAMA_BLAS=OFF" pip install llama-cpp-python --no-cache-dir
```

## Uso

```bash
# Com Ollama (backend atual)
python -m us_cli.main --ollama http://localhost:11434/v1 --ollama-model qwen2.5-coder:3b

# Com GGUF direto (soberania total, 1 processo)
python -m us_cli.main --model /caminho/para/modelo.gguf --ctx 4096 --threads 4

# Num diretório específico (carrega QWEN.md do projeto)
python -m us_cli.main --cwd /home/julios/Projetos/meu-projeto
```

## Comandos no REPL

- `/quit` ou `/exit` — sai do CLI
- `/reset` — limpa o histórico da conversa

## Tools disponíveis

| Tool | Descrição |
|------|-----------|
| `read_file` | Lê conteúdo de um arquivo |
| `write_file` | Cria ou sobrescreve um arquivo |
| `edit` | Substitui texto exato em um arquivo |
| `run_shell_command` | Executa comando shell |
| `glob` | Busca arquivos por pattern |
| `grep` | Busca conteúdo por regex |
| `list_directory` | Lista diretórios |
| `todo_write` | Gerencia lista de tarefas |
| `save_memory` | Salva fatos na memória de longo prazo |

## Estrutura do projeto

```
src/us_cli/
├── main.py          # Entry point + argparse
├── config.py        # Settings loader (settings.json)
├── memory.py        # Context discovery + import processor
├── prompt.py        # System prompt builder
├── llm.py           # LLM backends (llama-cpp + Ollama)
├── loop.py          # Conversation loop
└── tools/
    └── registry.py  # Tool definitions + implementations
```

## Diferenças do Qwen CLI original

| Qwen CLI (Node) | Us CLI (Python) |
|-----------------|-----------------|
| Telemetria Aliyun | Zero telemetria |
| OAuth Qwen Cloud | Local apenas |
| 2+ processos (ollama serve + CLI) | 1 processo (GGUF direto) |
| Código Google/Alibaba forkado | Código nosso, do zero |
| React Ink UI | Rich + prompt-toolkit |

## Licença

Código nosso. Sem Google. Sem Alibaba. Sem Aliyun.
Soberania. 💙
