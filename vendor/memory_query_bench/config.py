"""Operator-controlled targets and labelled replay data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import QueryCase, TargetConfig


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def synthetic_cases() -> list[QueryCase]:
    return [QueryCase(id=f"q-{i:04d}", query=f"Find fixture item {i}",
                      expected_ids=[f"doc-{i:04d}"],
                      forbidden_ids=[f"private-{i:04d}"]) for i in range(128)] + [
        QueryCase(id=f"negative-{i}", query=f"Missing fixture item {i}",
                  expect_empty=True) for i in range(16)
    ]


class Registry:
    def __init__(self, config_path: str | Path | None = None):
        self.targets: dict[str, TargetConfig] = {
            "mock": TargetConfig(kind="mock", description="Synthetic harness validation only",
                                 write_enabled=True),
            "mock-lossy": TargetConfig(kind="mock", description="Deliberately fast but incomplete",
                                       mock_latency_ms=1, mock_jitter_ms=0,
                                       mock_tail_probability=0, mock_empty_probability=1),
        }
        self.datasets: dict[str, list[QueryCase]] = {"synthetic": synthetic_cases()}
        if config_path is not None:
            path = Path(config_path).expanduser().resolve()
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) - {"targets", "datasets"}:
                raise ValueError("config accepts only targets and datasets")
            for name, value in payload.get("targets", {}).items():
                cfg = TargetConfig.model_validate(value)
                if cfg.cwd:
                    cfg.cwd = str((path.parent / cfg.cwd).resolve())
                self.targets[name] = cfg
            for name, filename in payload.get("datasets", {}).items():
                if name == "synthetic":
                    raise ValueError("synthetic is a reserved built-in dataset name")
                source = (path.parent / filename).resolve()
                if source.stat().st_size > 32_000_000:
                    raise ValueError("dataset exceeds 32 MB; use a bounded replay sample")
                cases = [QueryCase.model_validate_json(line) for line in
                         source.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not cases or len(cases) > 50000:
                    raise ValueError("dataset must contain 1..50000 cases")
                if len({case.id for case in cases}) != len(cases):
                    raise ValueError("duplicate query IDs")
                self.datasets[name] = cases

    def target(self, name: str) -> TargetConfig:
        if name not in self.targets:
            raise ValueError("unknown configured target_id")
        return self.targets[name]

    def cases(self, name: str) -> list[QueryCase]:
        if name not in self.datasets:
            raise ValueError("unknown configured dataset_id")
        return self.datasets[name]

    def describe(self) -> dict:
        return {
            "targets": [{"target_id": name, "kind": cfg.kind,
                         "description": cfg.description, "writes_enabled": cfg.write_enabled,
                         "simulated": cfg.kind == "mock" or cfg.synthetic_fixture}
                        for name, cfg in self.targets.items()],
            "datasets": [{"dataset_id": name, "cases": len(cases),
                          "sha256": digest([case.model_dump() for case in cases])}
                         for name, cases in self.datasets.items()],
            "limits": {"active_runs": 1, "requests": 20000, "concurrency": 64,
                       "rate_per_second": 2000, "max_run_seconds": 3600},
        }
