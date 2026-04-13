"""
Memory Discovery — descobre e carrega arquivos de contexto como o Qwen CLI.

Algoritmo:
1. Global: ~/.qwen/QWEN.md
2. Home: ~/QWEN.md
3. Workspace: sobe do CWD até project root (.git), coletando QWEN.md
4. Processa imports (@caminho) recursivamente
5. Concatena com marcadores
"""

import os
import re
from pathlib import Path


# Nomes padrão para arquivos de contexto (como no Qwen CLI)
CONTEXT_FILENAMES = ["QWEN.md", "AGENTS.md"]


def find_project_root(start: Path) -> Path | None:
    """Sobe diretórios até achar .git ou /."""
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    return None


def discover_context_files(cwd: str | None = None) -> list[Path]:
    """
    Descobre arquivos QWEN.md/AGENTS.md na ordem do Qwen CLI:
    1. ~/.qwen/QWEN.md (global)
    2. ~/QWEN.md (home direto)
    3. CWD → project root (upward scan)
    """
    working = Path(cwd) if cwd else Path.cwd()
    files: list[Path] = []
    seen: set[Path] = set()

    # 1. Global: ~/.qwen/QWEN.md
    global_qwen = Path.home() / ".qwen" / "QWEN.md"
    if global_qwen.exists():
        files.append(global_qwen)
        seen.add(global_qwen.resolve())

    # 1b. ~/.qwen/AGENTS.md
    global_agents = Path.home() / ".qwen" / "AGENTS.md"
    if global_agents.exists() and global_agents.resolve() not in seen:
        files.append(global_agents)
        seen.add(global_agents.resolve())

    # 2. Home direto: ~/QWEN.md (apenas no root do home)
    home_qwen = Path.home() / "QWEN.md"
    if home_qwen.exists() and home_qwen.resolve() not in seen:
        files.append(home_qwen)
        seen.add(home_qwen.resolve())

    # 3. Upward scan: do CWD até project root
    project_root = find_project_root(working)
    current = working
    upward_files: list[Path] = []

    while True:
        for name in CONTEXT_FILENAMES:
            candidate = current / name
            if candidate.exists() and candidate.resolve() not in seen:
                upward_files.append(candidate)
                seen.add(candidate.resolve())

        if current == project_root or current == current.parent:
            break
        current = current.parent

    # Inverte: mais distante primeiro, mais local por último
    files.extend(reversed(upward_files))

    return files


def find_imports(content: str) -> list[str]:
    """
    Encontra imports no formato @caminho (sem espaços).
    Detectado caractere por caractere como no Qwen CLI.
    """
    imports = []
    i = 0
    while i < len(content):
        if content[i] == "@":
            # Começa a capturar o path
            j = i + 1
            while j < len(content) and content[j] not in (" ", "\n", "\t", "\r"):
                j += 1
            path = content[i + 1:j]
            if path and not path.startswith("http://") and not path.startswith("https://"):
                imports.append(path)
            i = j
        else:
            i += 1
    return imports


def validate_import_path(import_path: str, base_dir: Path, project_root: Path | None) -> Path | None:
    """
    Valida que o import não sai do project root (security).
    Retorna o path absoluto se válido, None se inválido.
    """
    # Resolve relativo ao base_dir
    candidate = (base_dir / import_path).resolve()

    # Path traversal check
    if project_root:
        resolved_root = project_root.resolve()
        if not str(candidate).startswith(str(resolved_root)):
            return None

    if candidate.exists() and candidate.is_file():
        return candidate

    return None


def process_imports(content: str, base_dir: Path, project_root: Path | None,
                    processed: set[Path] | None = None, depth: int = 0) -> str:
    """
    Processa imports @caminho recursivamente, substituindo inline.
    Formato de saída:
    <!-- Imported from: ./path -->
    content
    <!-- End of import from: ./path -->
    """
    if processed is None:
        processed = set()
    if depth > 5:  # maxDepth
        return content

    imports = find_imports(content)
    result = content

    for imp in imports:
        resolved = validate_import_path(imp, base_dir, project_root)
        if resolved is None or resolved in processed:
            continue

        processed.add(resolved)

        try:
            imp_content = resolved.read_text(encoding="utf-8")
            # Processa imports do import recursivamente
            imp_content = process_imports(imp_content, resolved.parent, project_root,
                                          processed, depth + 1)

            replacement = (
                f"\n<!-- Imported from: {imp} -->\n"
                f"{imp_content}\n"
                f"<!-- End of import from: {imp} -->\n"
            )
            # Substitui o @path pelo conteúdo
            result = result.replace(f"@{imp}", replacement)
        except (OSError, UnicodeDecodeError):
            pass  # Skip imports que não dá pra ler

    return result


def load_context(cwd: str | None = None) -> str:
    """
    Carrega todo o contexto hierárquico e processa imports.
    Retorna a string concatenada com marcadores.
    """
    working = Path(cwd) if cwd else Path.cwd()
    project_root = find_project_root(working)
    files = discover_context_files(cwd)

    if not files:
        return ""

    processed_parts: list[str] = []

    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8")
            # Processa imports
            content = process_imports(content, fpath.parent, project_root)
            # Wrapper com marcadores
            rel = fpath.relative_to(Path.cwd()) if str(fpath).startswith(str(Path.cwd())) else fpath
            processed_parts.append(
                f"--- Context from: {rel} ---\n{content}\n--- End of Context from: {rel} ---"
            )
        except (OSError, UnicodeDecodeError) as e:
            print(f"⚠️  Erro lendo {fpath}: {e}")

    return "\n\n".join(processed_parts)
