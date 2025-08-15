uv run python -m agentic.orchestration.worker

agentic run examples/sample_prp.json --watch --auto-approve

# later, when ready to finish:

agentic approve deploy_ok --id <id>
