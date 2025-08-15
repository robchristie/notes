from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic import BaseModel
from ..config import settings, span


class PatchSpec(BaseModel):
    unified_diff: str


provider = OpenRouterProvider(api_key=settings.openrouter_api_key)

coder_agent = Agent(
    OpenAIModel(settings.openrouter_model_fast, provider=provider),
    output_type=PatchSpec,
    instructions=(
        "Write unified diffs only. Use minimal, surgical edits.\n"
        "Never include prose outside the diff."
    ),
)


async def run_coder(context: str) -> PatchSpec:
    with span("agent.coder"):
        if settings.dry_run:
            return PatchSpec(unified_diff="")
        out = await coder_agent.run(context)
        return out.output
