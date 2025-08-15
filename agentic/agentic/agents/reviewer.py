from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from ..contracts import ReviewReport
from ..config import settings, span

provider = OpenRouterProvider(api_key=settings.openrouter_api_key)

reviewer_agent = Agent(
    OpenAIModel(settings.openrouter_model_reasoning, provider=provider),
    output_type=ReviewReport,
    instructions=(
        "Summarize lint outputs into actionable issues and prioritize them. Emit ReviewReport."
    ),
)


async def run_reviewer(raw_text: str) -> ReviewReport:
    with span("agent.reviewer"):
        if settings.dry_run:
            return ReviewReport(issues=[])
        out = await reviewer_agent.run(raw_text)
        return out.output
