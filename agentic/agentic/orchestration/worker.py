from __future__ import annotations
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker, UnsandboxedWorkflowRunner
from ..config import settings
from .workflows import (
    BuildWorkflow,
    apply_patch_activity,
    run_tests_activity,
    run_linters_activity,
    write_docs_activity,
    repo_query_activity,
    coder_activity,
    docs_generate_activity,
)
from ._converter import data_converter


async def main() -> None:
    client = await Client.connect(
        settings.temporal_address,
        namespace=settings.temporal_namespace,
        data_converter=data_converter,
    )
    worker = Worker(
        client,
        task_queue="agentic-build-q",
        workflows=[BuildWorkflow],
        activities=[
            apply_patch_activity,
            run_tests_activity,
            run_linters_activity,
            write_docs_activity,
            repo_query_activity,
            coder_activity,
            docs_generate_activity,
        ],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    print("Worker started on queue 'agentic-build-q'")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
