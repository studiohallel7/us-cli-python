"""
Tool Registry — registra ferramentas disponíveis para o LLM.

Cada tool tem:
- name: nome da ferramenta
- description: descrição para o LLM
- parameters: schema JSON (formato OpenAI function calling)
- execute: função que executa a ferramenta
"""

import json
import subprocess
import glob as glob_mod
import os
import re
from pathlib import Path
from typing import Any, Callable


class Tool:
    """Uma ferramenta disponível para o LLM."""

    def __init__(self, name: str, description: str,
                 parameters: dict, execute: Callable[[dict], str]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.execute_fn = execute

    def to_openai_schema(self) -> dict:
        """Converte para formato OpenAI function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, args: dict) -> str:
        try:
            return self.execute_fn(args)
        except Exception as e:
            return f"Erro executando {self.name}: {e}"


def create_tool_registry() -> dict[str, Tool]:
    """Cria o registry com todas as ferramentas built-in."""
    tools = {}

    # ---- read_file ----
    tools["read_file"] = Tool(
        name="read_file",
        description="Lê o conteúdo de um arquivo. Use para ver código, documentos, etc.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Caminho absoluto do arquivo"},
                "offset": {"type": "integer", "description": "Linha inicial (0-based, opcional)"},
                "limit": {"type": "integer", "description": "Número máximo de linhas (opcional)"},
            },
            "required": ["file_path"],
        },
        execute_fn=_read_file,
    )

    # ---- write_file ----
    tools["write_file"] = Tool(
        name="write_file",
        description="Escreve conteúdo em um arquivo. Cria ou sobrescreve.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Caminho absoluto do arquivo"},
                "content": {"type": "string", "description": "Conteúdo a escrever"},
            },
            "required": ["file_path", "content"],
        },
        execute_fn=_write_file,
    )

    # ---- edit ----
    tools["edit"] = Tool(
        name="edit",
        description="Substitui texto em um arquivo. Precisa do old_string exato e new_string.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Caminho absoluto do arquivo"},
                "old_string": {"type": "string", "description": "Texto exato a substituir"},
                "new_string": {"type": "string", "description": "Novo texto"},
                "replace_all": {"type": "boolean", "description": "Substituir todas as ocorrências"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        execute_fn=_edit,
    )

    # ---- run_shell_command ----
    tools["run_shell_command"] = Tool(
        name="run_shell_command",
        description="Executa um comando shell. Use para git, build, ls, etc.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Comando shell a executar"},
                "description": {"type": "string", "description": "Descrição do que o comando faz"},
                "is_background": {"type": "boolean", "description": "Rodar em background?"},
            },
            "required": ["command"],
        },
        execute_fn=_run_shell_command,
    )

    # ---- glob ----
    tools["glob"] = Tool(
        name="glob",
        description="Busca arquivos por pattern glob (ex: **/*.py).",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pattern glob"},
                "path": {"type": "string", "description": "Diretório base (opcional)"},
            },
            "required": ["pattern"],
        },
        execute_fn=_glob,
    )

    # ---- grep ----
    tools["grep"] = Tool(
        name="grep",
        description="Busca conteúdo em arquivos por regex.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex a buscar"},
                "path": {"type": "string", "description": "Diretório base (opcional)"},
                "glob": {"type": "string", "description": "Filtro de arquivo glob (opcional)"},
                "limit": {"type": "integer", "description": "Limite de resultados (opcional)"},
            },
            "required": ["pattern"],
        },
        execute_fn=_grep,
    )

    # ---- todo_write ----
    tools["todo_write"] = Tool(
        name="todo_write",
        description="Gerencia uma lista de tarefas para acompanhar progresso.",
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["id", "content", "status"],
                    },
                    "description": "Lista atualizada de tarefas",
                },
            },
            "required": ["todos"],
        },
        execute_fn=_todo_write,
    )

    # ---- save_memory ----
    tools["save_memory"] = Tool(
        name="save_memory",
        description="Salva um fato na memória de longo prazo.",
        parameters={
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "Fato a salvar"},
                "scope": {"type": "string", "enum": ["global", "project"], "description": "Escopo da memória"},
            },
            "required": ["fact"],
        },
        execute_fn=_save_memory,
    )

    # ---- list_directory ----
    tools["list_directory"] = Tool(
        name="list_directory",
        description="Lista arquivos e diretórios em um caminho.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho absoluto do diretório"},
            },
            "required": ["path"],
        },
        execute_fn=_list_directory,
    )

    return tools


# ---- Implementações das ferramentas ----

def _read_file(args: dict) -> str:
    fpath = Path(args["file_path"])
    if not fpath.exists():
        return f"Arquivo não encontrado: {args['file_path']}"
    try:
        content = fpath.read_text(encoding="utf-8")
        offset = args.get("offset", 0)
        limit = args.get("limit")
        lines = content.splitlines()
        if limit:
            lines = lines[offset:offset + limit]
        else:
            lines = lines[offset:]
        return "\n".join(lines)
    except UnicodeDecodeError:
        return f"Arquivo binário (não é texto): {args['file_path']}"


def _write_file(args: dict) -> str:
    fpath = Path(args["file_path"])
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(args["content"], encoding="utf-8")
    return f"Arquivo escrito: {args['file_path']} ({len(args['content'])} chars)"


def _edit(args: dict) -> str:
    fpath = Path(args["file_path"])
    if not fpath.exists():
        return f"Arquivo não encontrado: {args['file_path']}"

    content = fpath.read_text(encoding="utf-8")
    old = args["old_string"]
    new = args["new_string"]
    replace_all = args.get("replace_all", False)

    if old not in content:
        return f"old_string não encontrado no arquivo."

    if replace_all:
        new_content = content.replace(old, new)
    else:
        new_content = content.replace(old, new, 1)

    fpath.write_text(new_content, encoding="utf-8")
    return f"Editado: {args['file_path']}"


def _run_shell_command(args: dict) -> str:
    cmd = args["command"]
    is_bg = args.get("is_background", False)

    if is_bg:
        subprocess.Popen(cmd, shell=True)
        return f"Comando iniciado em background: {cmd}"

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd()
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output or "(sem output)"
    except subprocess.TimeoutExpired:
        return f"Comando expirou (120s timeout): {cmd}"


def _glob(args: dict) -> str:
    base = args.get("path", os.getcwd())
    matches = glob_mod.glob(os.path.join(base, args["pattern"]), recursive=True)
    if not matches:
        return "Nenhum arquivo encontrado."
    return "\n".join(matches[:50])


def _grep(args: dict) -> str:
    base = args.get("path", os.getcwd())
    pattern = args["pattern"]
    glob_filter = args.get("glob")
    limit = args.get("limit", 50)

    results = []
    for root, dirs, files in os.walk(base):
        for fname in files:
            if glob_filter and not glob_mod.fnmatch(glob_filter, fname):
                continue
            fpath = os.path.join(root, fname)
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.splitlines(), 1):
                    if re.search(pattern, line):
                        results.append(f"{fpath}:{i}:{line.strip()}")
                        if len(results) >= limit:
                            break
            except (OSError, UnicodeError):
                pass
        if len(results) >= limit:
            break

    if not results:
        return "Nenhum resultado encontrado."
    return "\n".join(results[:limit])


def _todo_write(args: dict) -> str:
    todos = args.get("todos", [])
    lines = []
    for t in todos:
        status_icon = {"pending": "⬜", "in_progress": "🔵", "completed": "✅"}.get(t["status"], "⬜")
        lines.append(f"{status_icon} [{t['id']}] {t['content']}")
    return "\n".join(lines) if lines else "Lista de tarefas vazia."


def _save_memory(args: dict) -> str:
    fact = args["fact"]
    scope = args.get("scope", "project")

    if scope == "global":
        mem_path = Path.home() / ".qwen" / "QWEN.md"
    else:
        mem_path = Path.cwd() / "QWEN.md"

    if not mem_path.exists():
        return f"Arquivo de memória não encontrado: {mem_path}"

    content = mem_path.read_text(encoding="utf-8")
    marker = "## Qwen Added Memories"

    if marker in content:
        # Adiciona após o marker
        parts = content.split(marker)
        parts[1] = parts[1].rstrip() + f"\n- {fact}\n"
        new_content = marker.join(parts)
    else:
        new_content = content + f"\n## Qwen Added Memories\n- {fact}\n"

    mem_path.write_text(new_content, encoding="utf-8")
    return f"Memória salva: {fact}"


def _list_directory(args: dict) -> str:
    dpath = Path(args["path"])
    if not dpath.exists():
        return f"Diretório não encontrado: {args['path']}"
    if not dpath.is_dir():
        return f"Não é um diretório: {args['path']}"

    items = []
    for item in sorted(dpath.iterdir()):
        icon = "📁" if item.is_dir() else "📄"
        items.append(f"{icon} {item.name}")

    if not items:
        return "(diretório vazio)"
    return "\n".join(items)
