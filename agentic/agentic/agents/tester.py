from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from ..contracts import TestReport
from ..config import settings, span

provider = OpenRouterProvider(api_key=settings.openrouter_api_key)

tester_agent = Agent(
    OpenAIModel(settings.openrouter_model_fast, provider=provider),
    output_type=TestReport,
    instructions=(
        "Convert raw testing output into a structured TestReport. If parsing fails, be conservative."
    ),
)


async def run_tester(raw_text: str) -> TestReport:
    with span("agent.tester"):
        if settings.dry_run:
            return TestReport(passed=0, failed=0, duration_s=0.0)
        out = await tester_agent.run(raw_text)
        return out.output
