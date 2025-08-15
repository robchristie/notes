from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from ..contracts import DocsOut, PRP
from ..config import settings, span

provider = OpenRouterProvider(api_key=settings.openrouter_api_key)

doc_agent = Agent(
    OpenAIModel(settings.openrouter_model_fast, provider=provider),
    output_type=DocsOut,
    instructions=("Generate concise README and CHANGELOG text based on the PRP & Plan."),
)


async def run_doc(prp: PRP) -> DocsOut:
    with span("agent.docs"):
        if settings.dry_run:
            return DocsOut(readme="# README\n", changelog="## Changelog\n")
        out = await doc_agent.run({"prp": prp.model_dump()})
        return out.output
