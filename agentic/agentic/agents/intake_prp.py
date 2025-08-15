from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from ..contracts import PRP
from ..config import settings, span

provider = OpenRouterProvider(api_key=settings.openrouter_api_key)

intake_agent = Agent(
    OpenAIModel(settings.openrouter_model_reasoning, provider=provider),
    output_type=PRP,
    instructions=(
        "You are the Intake Agent. You MUST return ONLY a valid JSON object matching the PRP schema.\n"
        "No extra keys. Ask clarifying questions if needed, then emit final PRP."
    ),
)


async def run_intake(prompt: str) -> PRP:
    with span("agent.intake", prompt=prompt):
        if settings.dry_run:
            return PRP(feature="dry", context="dry", acceptance_tests=["dry"])
        out = await intake_agent.run(prompt)
        return out.output
