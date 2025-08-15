from __future__ import annotations
from pathlib import Path
from typing import Tuple
from ..config import span

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)

def write_docs(readme: str, changelog: str, root: str | None = None) -> Tuple[str, str]:
    root = root or "."
    readme_path = Path(root) / "README.md"
    changelog_path = Path(root) / "CHANGELOG.md"
    with span("tool.write_docs", files=f"{readme_path},{changelog_path}"):
        write_file(readme_path, readme)
        write_file(changelog_path, changelog)
    return str(readme_path), str(changelog_path)
