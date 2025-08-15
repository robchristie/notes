from __future__ import annotations
import os, subprocess, tempfile
from ..contracts import TestReport, TestFailure
from ..config import span

async def run_pytests(targets: list[str] | None = None) -> TestReport:
    targets = targets or ["."]
    with span("tool.run_pytests", targets=" ".join(targets)):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            junit_path = f.name
        try:
            proc = subprocess.run(
                ["pytest", "-q", f"--junitxml={junit_path}", *targets],
                capture_output=True, text=True, env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
            )
            passed = failed = 0
            for line in (proc.stdout + "\n" + proc.stderr).splitlines():
                if line.strip().startswith("==") and "passed" in line:
                    parts = line.split(",")
                    for p in parts:
                        if "passed" in p:
                            passed = int(p.split()[0].strip("= "))
                        if "failed" in p:
                            failed = int(p.split()[0].strip())
            return TestReport(passed=passed, failed=failed, duration_s=0.0)
        finally:
            try: os.unlink(junit_path)
            except Exception: pass
