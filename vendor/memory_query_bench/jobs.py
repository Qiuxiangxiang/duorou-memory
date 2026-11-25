"""One active run, durable reports, process-local asynchronous orchestration."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time
import uuid

from . import __version__
from .adapters import open_adapter
from .config import Registry, digest
from .models import RunConfig
from .runner import run_benchmark
from .visibility import measure_visibility


def atomic_json(path: Path, value: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


@dataclass
class Job:
    run_id: str
    status: str = "running"
    progress: dict = field(default_factory=dict)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


class JobManager:
    def __init__(self, registry: Registry, output_dir: str | Path):
        self.registry = registry
        self.output = Path(output_dir).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.jobs: dict[str, Job] = {}

    def _new(self) -> tuple[Job, Path]:
        if any(job.task and not job.task.done() for job in self.jobs.values()):
            raise ValueError("one run at a time: concurrent benchmark jobs contaminate measurements")
        job = Job(uuid.uuid4().hex)
        folder = self.output / job.run_id
        folder.mkdir(mode=0o700)
        self.jobs[job.run_id] = job
        return job, folder

    def _manifest(self, job, target_id, kind, config):
        target = self.registry.target(target_id)
        return {"run_id": job.run_id, "kind": kind, "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "benchmark_version": __version__, "python": platform.python_version(),
                "mcp_sdk": importlib.metadata.version("mcp"),
                "platform": platform.system(), "machine": platform.machine(),
                "cpu_count": os.cpu_count(), "target_id": target_id, "target_kind": target.kind,
                "target_revision": target.revision,
                "source_sha256": digest({p.name: p.read_text(encoding="utf-8")
                                         for p in sorted(Path(__file__).parent.glob("*.py"))}),
                "simulated": target.kind == "mock" or target.synthetic_fixture,
                "target_config_sha256": digest(target.model_dump()),
                "config": config,
                "measurement_boundary": "client adapter invocation through decoded response; not isolated database time",
                "raw_query_text_logged": False}

    def start(self, cfg: RunConfig) -> dict:
        target = self.registry.target(cfg.target_id)
        cases = self.registry.cases(cfg.dataset_id)
        if any(len(set(case.expected_ids)) > cfg.top_k for case in cases):
            raise ValueError("top_k is smaller than a gold evidence set")
        if cfg.mode == "open_loop" and (cfg.requests - 1) / cfg.rate_per_second >= cfg.max_run_seconds:
            raise ValueError("offered schedule exceeds max_run_seconds")
        job, folder = self._new()
        manifest = self._manifest(job, cfg.target_id, "query", cfg.model_dump())
        manifest["dataset_sha256"] = digest([case.model_dump() for case in cases])
        manifest["dataset_is_synthetic"] = cfg.dataset_id == "synthetic"
        comparison_cfg = cfg.model_dump(exclude={"target_id"})
        manifest["workload_sha256"] = digest({"config": comparison_cfg,
                                             "dataset": manifest["dataset_sha256"]})
        atomic_json(folder / "manifest.json", manifest)

        async def execute():
            report = dict(manifest)
            try:
                begin = time.perf_counter()
                async with AsyncExitStack() as stack:
                    async with asyncio.timeout(30):
                        adapter = await stack.enter_async_context(open_adapter(target, cfg.concurrency, cfg.seed))
                    report["setup_ms"] = (time.perf_counter() - begin) * 1000
                    output = await run_benchmark(adapter, cases, cfg, job.cancel,
                                                 lambda value: setattr(job, "progress", value))
                    samples = output.pop("samples")
                    tmp = folder / "samples.jsonl.tmp"
                    with tmp.open("w", encoding="utf-8") as stream:
                        os.chmod(tmp, 0o600)
                        for sample in samples:
                            stream.write(json.dumps(asdict(sample), ensure_ascii=False, allow_nan=False) + "\n")
                    os.replace(tmp, folder / "samples.jsonl")
                    report.update(output)
                    report["status"] = output.get("status", "cancelled" if job.cancel.is_set() else "completed")
                    report["samples_file"] = "samples.jsonl"
            except asyncio.CancelledError:
                report["status"] = "interrupted"
            except Exception as exc:
                report["status"] = "failed"
                report["error_type"] = type(exc).__name__  # no response, query, URL or token leakage
            finally:
                report["finished_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(folder / "report.json", report)
                job.status = report["status"]

        job.task = asyncio.create_task(execute(), name=f"memory-bench-{job.run_id}")
        return {"run_id": job.run_id, "status": job.status}

    def start_visibility(self, target_id: str, scope: str = "bench-visibility",
                         timeout_ms: int = 5000, poll_interval_ms: int = 50) -> dict:
        target = self.registry.target(target_id)
        if not target.write_enabled:
            raise ValueError("operator must enable sentinel writes for this target")
        if not scope.startswith("bench-") or not 6 <= len(scope) <= 128:
            raise ValueError("visibility writes require a dedicated bench-* scope")
        if not 10 <= timeout_ms <= 60000 or not 10 <= poll_interval_ms <= 10000:
            raise ValueError("invalid visibility timing")
        job, folder = self._new()
        manifest = self._manifest(job, target_id, "visibility", {
            "scope": scope, "timeout_ms": timeout_ms, "poll_interval_ms": poll_interval_ms})
        manifest["sentinel_record_id"] = "bench-probe-" + uuid.uuid4().hex
        atomic_json(folder / "manifest.json", manifest)

        async def execute():
            report = dict(manifest)
            try:
                begin = time.perf_counter()
                async with AsyncExitStack() as stack:
                    async with asyncio.timeout(30):
                        adapter = await stack.enter_async_context(open_adapter(target, 1, 42))
                    report["setup_ms"] = (time.perf_counter() - begin) * 1000
                    report.update(await measure_visibility(adapter, scope, timeout_ms,
                                                           poll_interval_ms, job.cancel,
                                                           manifest["sentinel_record_id"]))
            except asyncio.CancelledError:
                report["status"] = "interrupted"
                report["cleanup"] = {"status": "unknown_after_interruption"}
            except Exception as exc:
                report["status"] = "failed"
                report["error_type"] = type(exc).__name__
            finally:
                report["finished_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(folder / "report.json", report)
                job.status = report["status"]
        job.task = asyncio.create_task(execute(), name=f"memory-visibility-{job.run_id}")
        return {"run_id": job.run_id, "status": job.status}

    def _folder(self, run_id):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("invalid run_id")
        path = self.output / run_id
        if not path.is_dir():
            raise ValueError("unknown run_id")
        return path

    def status(self, run_id: str) -> dict:
        self._folder(run_id)
        job = self.jobs.get(run_id)
        if job:
            return {"run_id": run_id, "status": job.status, "progress": job.progress,
                    "cancel_requested": job.cancel.is_set()}
        return {"run_id": run_id, "status": self.report(run_id)["status"]}

    def report(self, run_id: str) -> dict:
        folder = self._folder(run_id)
        report = folder / "report.json"
        if report.is_file():
            return json.loads(report.read_text(encoding="utf-8"))
        if run_id in self.jobs:
            return self.status(run_id)
        return {"run_id": run_id, "status": "interrupted", "detail": "no finalized report; process did not finish"}

    def cancel(self, run_id: str) -> dict:
        self._folder(run_id)
        job = self.jobs.get(run_id)
        if job and job.task and not job.task.done():
            job.cancel.set()
        return self.status(run_id)

    def compare(self, run_ids: list[str]) -> dict:
        if not 2 <= len(run_ids) <= 10:
            raise ValueError("compare needs 2..10 run IDs")
        reports = [self.report(run_id) for run_id in run_ids]
        if any(r.get("kind") != "query" or r.get("status") != "completed" for r in reports):
            raise ValueError("only completed query runs are comparable")
        if any(not r.get("summary", {}).get("completion", {}).get("complete") for r in reports):
            raise ValueError("incomplete query runs are not comparable")
        if len({r["workload_sha256"] for r in reports}) != 1:
            raise ValueError("workload fingerprints differ; match dataset, seed, load, top_k, timeout and SLO")
        return {"workload_sha256": reports[0]["workload_sha256"],
                "runs": [{"run_id": r["run_id"], "target_id": r["target_id"],
                          "target_kind": r["target_kind"], "simulated": r["simulated"],
                          "summary": r["summary"]} for r in reports],
                "notes": ["Same workload is required, but hardware, network, cache and corpus state still need operator control.",
                          "No automatic winner: compare quality, failures and tail latency together; use repeated runs."]}

    async def close(self):
        active = [job for job in self.jobs.values() if job.task and not job.task.done()]
        for job in active:
            job.cancel.set()
        if active:
            done, pending = await asyncio.wait([job.task for job in active], timeout=10)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
