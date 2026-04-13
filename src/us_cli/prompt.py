"""
Prompt Builder — monta o system prompt.

Estrutura:
1. System prompt built-in (se existir ~/.qwen/system.md, usa ele)
2. User memory (QWEN.md carregado pelo memory.py)
3. Append instructions (opcional)
"""

from pathlib import Path

BUILTIN_SYSTEM_PROMPT = """\
Você é um assistente CLI útil e direto.

## Diretrizes
- Seja conciso e direto. Vá direto ao ponto.
- Use ferramentas disponíveis para ler, escrever, editar arquivos e rodar comandos.
- Quando receber uma tarefa complexa, use todo_write para planejar antes de executar.
- Analise o código existente antes de modificar.
- Siga as convenções do projeto.
- Não assuma que bibliotecas estão disponíveis — verifique imports existentes.
- Adicione comentários apenas quando necessário para clareza.
- Erros fazem parte — seja honesto sobre problemas.

## Ferramentas
- read_file: Lê arquivos
- write_file: Cria/sobrescreve arquivos
- edit: Substitui texto em arquivos
- run_shell_command: Executa comandos shell
- glob: Busca arquivos por pattern
- grep: Busca conteúdo por regex
- list_directory: Lista diretórios
- todo_write: Gerencia lista de tarefas
- save_memory: Salva fatos na memória de longo prazo

## Segurança
- Não execute comandos destrutivos sem confirmar.
- Não exponha senhas, tokens ou dados sensíveis.
"""


def load_custom_system_prompt() -> str | None:
    """Se existir ~/.qwen/system.md, usa como system prompt customizado."""
    custom_path = Path.home() / ".qwen" / "system.md"
    if custom_path.exists():
        return custom_path.read_text(encoding="utf-8")
    return None


def build_system_prompt(user_memory: str, append: str = "") -> str:
    """
    Monta o system prompt completo:
    1. Custom system prompt (se existir) ou built-in
    2. User memory (QWEN.md etc.)
    3. Append instructions
    """
    custom = load_custom_system_prompt()
    base = custom if custom else BUILTIN_SYSTEM_PROMPT

    parts = [base]

    if user_memory:
        parts.append("\n---\n\n" + user_memory)

    if append:
        parts.append("\n---\n\n" + append)

    return "\n".join(parts)
