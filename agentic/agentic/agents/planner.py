from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from ..contracts import ExecutionPlan, PRP
from ..config import settings, span

provider = OpenRouterProvider(api_key=settings.openrouter_api_key)

planner_agent = Agent(
    OpenAIModel(settings.openrouter_model_reasoning, provider=provider),
    output_type=ExecutionPlan,
    instructions=(
        "Create a typed, acyclic plan. Each node must map to a known tool name, with explicit inputs/outputs schemas.\n"
        "Do NOT invent tools. Respect allowed tool catalog."
    ),
)


async def run_planner(prp: PRP) -> ExecutionPlan:
    with span("agent.planner", feature=prp.feature):
        if settings.dry_run:
            return ExecutionPlan(prp=prp, tasks=[])
        out = await planner_agent.run({"prp": prp.model_dump()})
        return out.output
