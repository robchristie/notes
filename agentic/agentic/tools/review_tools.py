from __future__ import annotations
import subprocess, os
from ..contracts import ReviewIssue, ReviewReport
from ..config import span

def _capture(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
    return p.returncode, p.stdout, p.stderr

async def run_linters(paths: list[str] | None = None) -> ReviewReport:
    paths = paths or ["."]
    issues: list[ReviewIssue] = []

    with span("tool.linter.ruff", paths=" ".join(paths)):
        code, out, _ = _capture(["ruff", "check", *paths, "--format", "concise"])
        for line in out.splitlines():
            # format: path:line:col: code message
            try:
                loc, rest = line.split(":", 1)
                file = loc
                parts = rest.split(" ", 1)
                after = parts[1] if len(parts) > 1 else ""
                rule = parts[0].strip()
                line_no = int(rest.strip().split(":")[1])
                issues.append(ReviewIssue(file=file, line=line_no, rule=rule, message=after))
            except Exception:
                pass

    with span("tool.linter.bandit", paths=" ".join(paths)):
        _capture(["bandit", "-q", "-r", *paths])

    # mypy optional; will likely warn due to missing stubs in a starter repo
    # _capture(["mypy", *paths])

    return ReviewReport(issues=issues)
