from __future__ import annotations
from typing import List, Literal, Dict, Optional, Tuple
from pydantic import BaseModel


class BuildWorkflowOutput(BaseModel):
    prp_ok: bool
    deploy_ok: bool
    results: Dict[str, str] = {}


class PRP(BaseModel):
    feature: str
    context: str
    non_goals: List[str] = []
    acceptance_tests: List[str]
    api_surfaces: List[str] = []
    constraints: Dict[str, str] = {}
    artifacts: List[str] = ["code", "tests", "docs"]


class PlanTask(BaseModel):
    id: str
    kind: Literal["code", "test", "doc", "review", "deploy"]
    tool: str
    inputs_schema_ref: str
    outputs_schema_ref: str
    depends_on: List[str] = []


class ExecutionPlan(BaseModel):
    prp: PRP
    tasks: List[PlanTask]


class TestFailure(BaseModel):
    nodeid: str
    message: str
    captured: Optional[str] = None


class TestReport(BaseModel):
    passed: int
    failed: int
    duration_s: float
    failures: List[TestFailure] = []


class PatchIn(BaseModel):
    unified_diff: str


class PatchOut(BaseModel):
    applied: bool
    conflicts: Optional[str] = None


class ReviewIssue(BaseModel):
    file: str
    line: int | None = None
    severity: Literal["info", "warn", "error"] = "warn"
    rule: str
    message: str


class ReviewReport(BaseModel):
    issues: List[ReviewIssue] = []


class DocsOut(BaseModel):
    readme: str
    changelog: str
