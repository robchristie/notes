from __future__ import annotations
import asyncio
import time
from typing import Optional
import typer
from temporalio.client import Client
from .workflows import BuildWorkflow
from ..config import settings
from ..contracts import PRP, ExecutionPlan, PlanTask
from ._converter import data_converter

app = typer.Typer(add_completion=False, help="Agentic build CLI")


def _mk_default_plan(prp: PRP) -> ExecutionPlan:
    tasks = [
        PlanTask(
            id="code-1",
            kind="code",
            tool="code_apply_patch",
            inputs_schema_ref="Write a hello endpoint",
            outputs_schema_ref="PatchOut",
            depends_on=[],
        ),
        PlanTask(
            id="test-1",
            kind="test",
            tool="run_pytests",
            inputs_schema_ref=".",
            outputs_schema_ref="TestReport",
            depends_on=["code-1"],
        ),
        PlanTask(
            id="review-1",
            kind="review",
            tool="run_linters",
            inputs_schema_ref=".",
            outputs_schema_ref="ReviewReport",
            depends_on=["code-1"],
        ),
        PlanTask(
            id="doc-1",
            kind="doc",
            tool="write_docs",
            inputs_schema_ref="README/CHANGELOG",
            outputs_schema_ref="paths",
            depends_on=["test-1", "review-1"],
        ),
        PlanTask(
            id="deploy-1",
            kind="deploy",
            tool="noop",
            inputs_schema_ref="",
            outputs_schema_ref="",
            depends_on=["doc-1"],
        ),
    ]
    return ExecutionPlan(prp=prp, tasks=tasks)


@app.command()
def run(
    prp_path: str,
    plan_path: Optional[str] = None,
    watch: bool = typer.Option(False, help="Poll workflow status until done"),
    auto_approve: bool = typer.Option(False, help="Auto-signal prp_ok immediately"),
):
    async def _run():
        client = await Client.connect(
            settings.temporal_address, namespace=settings.temporal_namespace
        )
        with open(prp_path) as f:
            prp = PRP.model_validate_json(f.read())
        if plan_path:
            with open(plan_path) as f:
                plan = ExecutionPlan.model_validate_json(f.read())
        else:
            plan = _mk_default_plan(prp)
            handle = await client.start_workflow(
                BuildWorkflow.run,
                args=[prp, plan],
                id=f"build-{prp.feature[:16]}",
                task_queue="agentic-build-q",
            )
        print("Started:", handle.id)
        status = await handle.query("status")
        print("Status:", status.model_dump() if hasattr(status, "model_dump") else status)
        if auto_approve and (hasattr(status, "prp_ok") and not status.prp_ok):
            await handle.signal("approve", "prp_ok")
            print("Auto-approved: prp_ok")
        if watch:
            # simple poll loop; ctrl-c to stop
            while True:
                status = await handle.query("status")
                print("Status:", status.model_dump() if hasattr(status, "model_dump") else status)
                if hasattr(status, "deploy_ok") and status.deploy_ok:
                    break
                await asyncio.sleep(2)

    asyncio.run(_run())


@app.command()
def approve(gate: str, id: str = typer.Option(..., "--id")):
    async def _approve():
        client = await Client.connect(
            settings.temporal_address,
            namespace=settings.temporal_namespace,
            data_converter=data_converter,
        )
        handle = client.get_workflow_handle(id)
        await handle.signal("approve", gate)
        print("OK ->", gate)

    asyncio.run(_approve())
