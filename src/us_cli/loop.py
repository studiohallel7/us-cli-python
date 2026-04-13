"""
Conversation Loop — o coração do CLI.

Fluxo:
1. Usuário digita prompt
2. Envia para o LLM (system prompt + histórico)
3. LLM responde (texto e/ou tool calls)
4. Se tem tool calls: executa ferramentas, anexa resultados
5. Enva de volta ao LLM
6. Repete até não ter mais tool calls
7. Mostra resposta final
"""

import json
from rich.console import Console
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from pathlib import Path

from .llm import create_backend, Message, LLMResponse
from .prompt import build_system_prompt
from .memory import load_context
from .tools.registry import create_tool_registry


console = Console()


def run_loop(model_config: dict, cwd: str | None = None):
    """
    Loop principal do CLI.
    """
    # Inicializa componentes
    console.print("[bold green]🌿 Us CLI — Soberania Local[/bold green]")
    console.print()

    backend = create_backend(model_config)
    tools = create_tool_registry()

    # Carrega contexto (QWEN.md, memory, etc.)
    user_memory = load_context(cwd)
    system_prompt = build_system_prompt(user_memory)

    # Histórico da conversa
    history: list[dict] = []

    # Prompt interativo
    history_file = Path.home() / ".qwen" / "us_history.txt"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_file)),
    )

    console.print("[dim]Digite sua mensagem. /quit ou /exit para sair.[/dim]")
    console.print()

    while True:
        try:
            user_input = session.prompt("💙 Us> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dimm]Tchau, Juzim. 💙[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Comandos especiais
        if user_input in ("/quit", "/exit", "/sair"):
            console.print("Tchau, Juzim. 💙")
            break

        if user_input == "/reset":
            history.clear()
            console.print("[dim]Histórico limpo.[/dim]")
            continue

        # Adiciona mensagem do usuário
        history.append({"role": "user", "content": user_input})

        # Loop de ferramenta
        max_turns = 20  # segurança contra loop infinito
        turn = 0

        while turn < max_turns:
            turn += 1

            # Monta mensagens para o LLM
            messages = _prepare_messages(history)

            # Chama o LLM
            try:
                console.print("[dim]Pensando...[/dim]")
                response = backend.chat(
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=model_config.get("temperature", 0.7),
                    max_tokens=model_config.get("max_tokens", 4096),
                    tools=[t.to_openai_schema() for t in tools.values()],
                )
            except Exception as e:
                console.print(f"[red]Erro no LLM: {e}[/red]")
                break

            # Processa resposta
            assistant_content = response.content

            # Se tem tool calls, executa
            if response.tool_calls:
                # Adiciona resposta com tool calls
                tool_calls_data = []
                for tc in response.tool_calls:
                    tool_calls_data.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    })

                history.append({
                    "role": "assistant",
                    "content": assistant_content or None,
                    "tool_calls": tool_calls_data,
                })

                # Executa cada tool call
                for tc in response.tool_calls:
                    tool = tools.get(tc.name)
                    if tool:
                        console.print(f"[dim]🔧 {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})[/dim]")
                        result = tool.execute(tc.arguments)
                    else:
                        result = f"Ferramenta desconhecida: {tc.name}"

                    # Adiciona resultado
                    history.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.id,
                    })

                    # Mostra resultado resumido
                    display = result[:200] + "..." if len(result) > 200 else result
                    console.print(f"[dim]→ {display}[/dim]")

                # Continua o loop (LLM vai processar resultados)
                continue

            # Sem tool calls → resposta final
            if assistant_content:
                console.print()
                console.print(Markdown(assistant_content))
                console.print()

                # Adiciona ao histórico
                history.append({"role": "assistant", "content": assistant_content})

            break

        if turn >= max_turns:
            console.print("[red]⚠️  Limite de turns atingido (possível loop).[/red]")


def _prepare_messages(history: list[dict]) -> list[dict]:
    """
    Prepara histórico para envio ao LLM.
    Limita tamanho para não estourar contexto.
    """
    # Trunca histórico se muito longo (últimas 30 mensagens)
    if len(history) > 30:
        # Mantém system + últimas 29
        return history[-30:]
    return history
