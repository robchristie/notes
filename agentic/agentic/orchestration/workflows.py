from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from ..contracts import (
    PRP,
    ExecutionPlan,
    PatchIn,
    TestReport,
    ReviewReport,
    DocsOut,
    PlanTask,
    BuildWorkflowOutput,
)
from ..tools import code_apply_patch, run_pytests, run_linters, write_docs, repo_query
from ..agents import run_coder, run_doc
from ..config import span


@activity.defn
async def coder_activity(context_text: str) -> str:
    # wraps agent call so we can retriable-ize it
    spec = await run_coder(context_text)
    return spec.unified_diff


@activity.defn
async def docs_generate_activity(prp_dict: dict) -> DocsOut:
    """Generate README/CHANGELOG content via Doc agent; writing files is separate."""
    return await run_doc(PRP(**prp_dict))


def topo_sort(tasks: List[PlanTask]) -> List[PlanTask]:
    # simple Kahn's algorithm
    by_id = {t.id: t for t in tasks}
    deps = {t.id: set(t.depends_on) for t in tasks}
    ready = [t for t in tasks if not deps[t.id]]
    sorted_out: List[PlanTask] = []
    while ready:
        n = ready.pop()
        sorted_out.append(n)
        for m in tasks:
            if n.id in deps[m.id]:
                deps[m.id].remove(n.id)
                if not deps[m.id]:
                    ready.append(m)
    if len(sorted_out) != len(tasks):
        return tasks  # fallback if cycle
    return sorted_out


@dataclass
class BuildState:
    prp_ok: bool = False
    deploy_ok: bool = False
    results: Dict[str, str] | None = None


@activity.defn
async def apply_patch_activity(diff_text: str) -> str:
    out = await code_apply_patch(PatchIn(unified_diff=diff_text))
    if not out.applied:
        return f"CONFLICTS: {out.conflicts}" if out.conflicts else "FAILED"
    return "APPLIED"


@activity.defn
async def run_tests_activity(targets: list[str] | None = None) -> TestReport:
    return await run_pytests(targets)


@activity.defn
async def run_linters_activity(paths: list[str] | None = None) -> ReviewReport:
    return await run_linters(paths)


@activity.defn
async def write_docs_activity(readme: str, changelog: str) -> str:
    p1, p2 = write_docs(readme, changelog)
    return f"{p1},{p2}"


@activity.defn
async def repo_query_activity(kind: str, query: str) -> str:
    out = await repo_query(type("Q", (), {"kind": kind, "query": query})())  # small shim
    return str(out.model_dump())


@workflow.defn
class BuildWorkflow:
    def __init__(self) -> None:
        self.state = BuildState(results={})

    @workflow.run
    async def run(self, prp: PRP, plan: ExecutionPlan) -> BuildWorkflowOutput:
        # Gate A: PRP approval
        await workflow.wait_condition(lambda: self.state.prp_ok)

        # topologically sort plan
        tasks = topo_sort(plan.tasks)

        for t in tasks:
            # span name hinting for workflow logs (not Langfuse—this is Temporal)
            if t.kind == "code":
                # Call Coder agent to generate a diff (context could include repo graph queries)
                with span("wf.code", task=t.id, tool=t.tool):
                    patch = await workflow.execute_activity(
                        coder_activity,
                        args=[t.inputs_schema_ref],  # context placeholder
                        schedule_to_close_timeout=timedelta(minutes=5),
                    )
                    res = await workflow.execute_activity(
                        apply_patch_activity,
                        args=[patch],
                        schedule_to_close_timeout=timedelta(minutes=5),
                        retry_policy=RetryPolicy(maximum_attempts=2),
                    )
                    self.state.results[t.id] = res

            elif t.kind == "test":
                with span("wf.test", task=t.id):
                    report = await workflow.execute_activity(
                        run_tests_activity,
                        args=[["."]],
                        schedule_to_close_timeout=timedelta(minutes=20),
                        retry_policy=RetryPolicy(
                            maximum_attempts=3,
                            initial_interval=timedelta(seconds=2),
                            maximum_interval=timedelta(seconds=30),
                            backoff_coefficient=2.0,
                        ),
                    )
                    self.state.results[t.id] = f"{report.passed} passed/{report.failed} failed"

            elif t.kind == "review":
                with span("wf.review", task=t.id):
                    review = await workflow.execute_activity(
                        run_linters_activity,
                        args=[["."]],
                        schedule_to_close_timeout=timedelta(minutes=10),
                    )
                    self.state.results[t.id] = f"{len(review.issues)} issues"

            elif t.kind == "doc":
                with span("wf.docs", task=t.id):
                    # 1) generate docs via agent
                    docs = await workflow.execute_activity(
                        docs_generate_activity,
                        args=[prp.model_dump()],
                        schedule_to_close_timeout=timedelta(minutes=5),
                    )
                    # 2) write them to disk
                    paths = await workflow.execute_activity(
                        write_docs_activity,
                        args=[docs.readme, docs.changelog],
                        schedule_to_close_timeout=timedelta(minutes=5),
                    )
                    self.state.results[t.id] = f"docs:{paths}"
            else:
                self.state.results[t.id] = "skipped"

        # Gate D: Deploy approval
        await workflow.wait_condition(lambda: self.state.deploy_ok)
        return BuildWorkflowOutput(
            prp_ok=self.state.prp_ok,
            deploy_ok=self.state.deploy_ok,
            results=self.state.results or {},
        )

    @workflow.signal
    async def approve(self, gate: str) -> None:
        if gate == "prp_ok":
            self.state.prp_ok = True
        if gate == "deploy_ok":
            self.state.deploy_ok = True

    @workflow.query
    def status(self) -> BuildWorkflowOutput:
        return BuildWorkflowOutput(
            prp_ok=self.state.prp_ok,
            deploy_ok=self.state.deploy_ok,
            results=self.state.results or {},
        )
