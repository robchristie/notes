from __future__ import annotations
import subprocess, tempfile, os, shutil
from ..contracts import PatchIn, PatchOut
from ..config import span

GIT_BIN = shutil.which("git")

async def code_apply_patch(inp: PatchIn, cwd: str | None = None) -> PatchOut:
    cwd = cwd or os.getcwd()
    if not GIT_BIN:
        return PatchOut(applied=False, conflicts="git not available; install git for 3-way apply")

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(inp.unified_diff)
        diff_path = f.name
    try:
        with span("tool.code_apply_patch"):
            proc = subprocess.run(
                [GIT_BIN, "apply", "--3way", diff_path],
                cwd=cwd, capture_output=True, text=True
            )
        if proc.returncode == 0:
            return PatchOut(applied=True)
        return PatchOut(applied=False, conflicts=proc.stderr or proc.stdout)
    finally:
        os.unlink(diff_path)
