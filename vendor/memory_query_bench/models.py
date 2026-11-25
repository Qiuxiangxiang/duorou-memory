"""Shared, independently designed benchmark contracts (no target-engine code)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryCase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    query: str
    scope: str = "bench-demo"
    expected_ids: list[str] = Field(default_factory=list)
    forbidden_ids: list[str] = Field(default_factory=list)
    expect_empty: bool = False

    @model_validator(mode="after")
    def check_gold(self):
        if not self.id or not self.scope:
            raise ValueError("id and scope must be nonempty")
        if not self.expected_ids and not self.expect_empty:
            raise ValueError("provide expected_ids or explicitly set expect_empty=true")
        if self.expect_empty and self.expected_ids:
            raise ValueError("expect_empty conflicts with expected_ids")
        if set(self.expected_ids) & set(self.forbidden_ids):
            raise ValueError("gold and forbidden IDs overlap")
        return self


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["mock", "http", "mcp_stdio", "mcp_http"]
    description: str = ""
    revision: str = "unspecified"
    synthetic_fixture: bool = False
    url: str | None = None
    token_env: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    query_tool: str = "memory_search"
    query_path: str = "/query"
    query_template: dict[str, Any] = Field(default_factory=lambda: {
        "query": "$query", "top_k": "$top_k", "scope": "$scope"
    })
    results_path: str = "results"
    id_field: str = "id"
    write_enabled: bool = False
    write_tool: str = "memory_upsert"
    delete_tool: str = "memory_delete"
    write_path: str = "/upsert"
    delete_path: str = "/delete"
    write_template: dict[str, Any] = Field(default_factory=lambda: {
        "record_id": "$record_id", "text": "$text", "scope": "$scope"
    })
    delete_template: dict[str, Any] = Field(default_factory=lambda: {
        "record_id": "$record_id", "scope": "$scope"
    })
    max_response_bytes: int = Field(default=2_000_000, ge=1024, le=2_000_000)
    mock_latency_ms: float = Field(default=5, ge=0, le=10000)
    mock_jitter_ms: float = Field(default=5, ge=0, le=10000)
    mock_tail_probability: float = Field(default=0.02, ge=0, le=1)
    mock_tail_ms: float = Field(default=40, ge=0, le=10000)
    mock_error_probability: float = Field(default=0, ge=0, le=1)
    mock_empty_probability: float = Field(default=0, ge=0, le=1)
    mock_visibility_delay_ms: float = Field(default=100, ge=0, le=60000)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    target_id: str
    dataset_id: str = "synthetic"
    mode: Literal["closed_loop", "open_loop"] = "closed_loop"
    requests: int = Field(default=500, ge=1, le=20000)
    concurrency: int = Field(default=4, ge=1, le=64)
    rate_per_second: float = Field(default=50, gt=0, le=2000)
    queue_capacity: int = Field(default=64, ge=1, le=20000)
    warmup: int = Field(default=20, ge=0, le=1000)
    timeout_ms: int = Field(default=2000, ge=1, le=60000)
    top_k: int = Field(default=10, ge=1, le=1000)
    seed: int = 42
    slo_ms: float = Field(default=200, gt=0)
    min_recall: float = Field(default=1, ge=0, le=1)
    max_error_rate: float = Field(default=0.01, ge=0, le=1)
    max_run_seconds: int = Field(default=300, ge=1, le=3600)


@dataclass
class QueryResult:
    ids: list[str]
    response_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample:
    index: int
    query_id: str
    status: str
    scheduled_ms: float
    dispatch_ms: float | None = None
    finish_ms: float | None = None
    latency_ms: float | None = None
    sojourn_ms: float | None = None
    queue_ms: float | None = None
    recall: float | None = None
    any_hit: bool | None = None
    all_hit: bool | None = None
    quality_ok: bool = False
    forbidden_count: int = 0
    result_count: int = 0
    response_bytes: int = 0
    error_type: str | None = None
    censored: bool = False

