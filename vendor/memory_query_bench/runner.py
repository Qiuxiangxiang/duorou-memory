"""Independent, bounded, no-retry load generation for query adapters.

Open-loop arrivals follow a precomputed monotonic schedule.  The producer never
waits for a free queue slot: overload is counted as dropped offered work.
Closed-loop workers take their next request only after the preceding one ends.
The hard run budget covers warmup, measured load, and drain.  Adapters must
cooperate with cancellation to stop external work; the runner will not wait
indefinitely for an adapter that suppresses cancellation.
"""
from __future__ import annotations

import asyncio
import inspect
import random
from collections import Counter
from time import perf_counter
from typing import Callable, Protocol

from .metrics import ProtocolError, TERMINAL_STATUSES, score_quality, summarize_samples
from .models import QueryCase, QueryResult, RunConfig, Sample


class QueryAdapter(Protocol):
    async def query(self, case: QueryCase, top_k: int) -> QueryResult: ...


def _consume_task(task: asyncio.Future) -> None:
    """Retrieve detached-task exceptions without logging query contents."""
    if not task.cancelled():
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass


class _Phase:
    def __init__(self, adapter: QueryAdapter, plan: list[QueryCase],
                 config: RunConfig, name: str, deadline: float,
                 cancel_event: asyncio.Event | None,
                 progress: Callable | None):
        self.adapter = adapter
        self.plan = plan
        self.config = config
        self.name = name
        self.deadline = deadline
        self.cancel_event = cancel_event
        self.progress = progress
        self.started = perf_counter()
        self.samples: dict[int, Sample] = {}
        self.next_index = 0
        self.stop_reason: str | None = None
        self.stop_at: float | None = None
        self.rpc_tasks: set[asyncio.Task] = set()
        self.owned_tasks: list[asyncio.Task] = []
        self.last_progress = 0.0
        self.progress_errors = 0

    def stop(self, reason: str, when: float | None = None) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            self.stop_at = perf_counter() if when is None else when

    def stopped(self) -> bool:
        if self.stop_reason is not None:
            return True
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.stop("cancelled")
        elif perf_counter() >= self.deadline:
            self.stop("max_run_seconds", self.deadline)
        return self.stop_reason is not None

    def offer(self, index: int, scheduled: float) -> Sample:
        sample = Sample(index=index, query_id=self.plan[index].id, status="pending",
                        scheduled_ms=max(0.0, (scheduled - self.started) * 1000))
        self.samples[index] = sample
        return sample

    def finish(self, sample: Sample, status: str, when: float,
               error_type: str | None = None, censored: bool = False) -> None:
        if sample.status in TERMINAL_STATUSES:
            return
        sample.status = status
        sample.finish_ms = max(sample.scheduled_ms, (when - self.started) * 1000)
        sample.sojourn_ms = max(0.0, sample.finish_ms - sample.scheduled_ms)
        if sample.dispatch_ms is not None:
            sample.latency_ms = max(0.0, sample.finish_ms - sample.dispatch_ms)
        sample.error_type = error_type
        sample.censored = censored
        self.report()

    def report(self, force: bool = False) -> None:
        if self.progress is None:
            return
        now = perf_counter()
        if not force and now - self.last_progress < 0.1:
            return
        self.last_progress = now
        counts = Counter(sample.status for sample in self.samples.values())
        completed = sum(counts[status] for status in TERMINAL_STATUSES)
        event = {
            "phase": self.name,
            "offered": len(self.samples),
            "attempted": sum(sample.dispatch_ms is not None for sample in self.samples.values()),
            "completed": completed,
            "total": len(self.plan),
            "counts": {status: counts[status] for status in TERMINAL_STATUSES},
            "elapsed_seconds": max(0.0, now - self.started),
            "stop_reason": self.stop_reason,
        }
        try:
            result = self.progress(event)
            if inspect.isawaitable(result):
                task = asyncio.ensure_future(result)
                task.add_done_callback(_consume_task)
        except Exception:
            # UI/telemetry callbacks must not change a benchmark response into
            # an RPC error, and may not cause automatic request retries.
            self.progress_errors += 1

    async def attempt(self, sample: Sample) -> None:
        if self.stopped():
            self.finish(sample, "cancelled", self.stop_at or perf_counter(),
                        self.stop_reason)
            return
        dispatched = perf_counter()
        sample.dispatch_ms = max(sample.scheduled_ms, (dispatched - self.started) * 1000)
        sample.queue_ms = max(0.0, sample.dispatch_ms - sample.scheduled_ms)
        sample.status = "running"
        case = self.plan[sample.index]
        rpc: asyncio.Task | None = None
        try:
            rpc = asyncio.create_task(self.adapter.query(case, self.config.top_k))
            self.rpc_tasks.add(rpc)
            rpc.add_done_callback(self.rpc_tasks.discard)
            rpc.add_done_callback(_consume_task)
            rpc_deadline = dispatched + self.config.timeout_ms / 1000
            limit = min(rpc_deadline, self.deadline)
            done, _ = await asyncio.wait({rpc}, timeout=max(0.0, limit - perf_counter()))
            now = perf_counter()
            if not done:
                rpc.cancel()
                if self.deadline <= rpc_deadline and now >= self.deadline:
                    self.stop("max_run_seconds", self.deadline)
                    self.finish(sample, "cancelled", now, "max_run_seconds", censored=True)
                else:
                    self.finish(sample, "timeout", now, "rpc_timeout", censored=True)
                return
            if self.stopped():
                self.finish(sample, "cancelled", now, self.stop_reason, censored=True)
                return
            if rpc.cancelled():
                self.finish(sample, "cancelled", now, "adapter_cancelled", censored=True)
                return
            result = rpc.result()
            # Preserve counts and forbidden-ID evidence even when the envelope
            # is invalid (e.g. a duplicate or oversized top-k response).
            if isinstance(result, QueryResult) and isinstance(result.ids, list):
                sample.result_count = len(result.ids)
                valid_ids = {value for value in result.ids if isinstance(value, str)}
                sample.forbidden_count = len(valid_ids & set(case.forbidden_ids))
            quality = score_quality(case, result, self.config.top_k, self.config.min_recall)
            for key, value in quality.items():
                setattr(sample, key, value)
            self.finish(sample, "success", now)
        except ProtocolError as exc:
            self.finish(sample, "error", perf_counter(), exc.code)
        except (asyncio.TimeoutError, TimeoutError):
            self.finish(sample, "timeout", perf_counter(), "adapter_timeout", censored=True)
        except asyncio.CancelledError:
            if rpc is not None and not rpc.done():
                rpc.cancel()
            self.finish(sample, "cancelled", perf_counter(),
                        self.stop_reason or "cancelled", censored=sample.dispatch_ms is not None)
            raise
        except Exception as exc:
            self.finish(sample, "error", perf_counter(), type(exc).__name__)

    async def closed_worker(self) -> None:
        while not self.stopped() and self.next_index < len(self.plan):
            index = self.next_index
            self.next_index += 1
            sample = self.offer(index, perf_counter())
            await self.attempt(sample)

    async def queue_worker(self, queue: asyncio.Queue) -> None:
        while True:
            sample = await queue.get()
            try:
                await self.attempt(sample)
            finally:
                queue.task_done()

    async def produce(self, queue: asyncio.Queue) -> None:
        interval = 1.0 / self.config.rate_per_second
        for index in range(len(self.plan)):
            scheduled = self.started + index * interval
            if self.stopped():
                break
            await asyncio.sleep(max(0.0, scheduled - perf_counter()))
            if self.stopped():
                break
            sample = self.offer(index, scheduled)
            self.next_index = index + 1
            try:
                queue.put_nowait(sample)
            except asyncio.QueueFull:
                self.finish(sample, "dropped", perf_counter(), "queue_full")

    async def work(self, mode: str) -> None:
        worker_count = min(self.config.concurrency, len(self.plan))
        if mode == "closed_loop":
            self.owned_tasks = [asyncio.create_task(self.closed_worker())
                                for _ in range(worker_count)]
            await asyncio.gather(*self.owned_tasks)
            return
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.queue_capacity)
        workers = [asyncio.create_task(self.queue_worker(queue)) for _ in range(worker_count)]
        producer = asyncio.create_task(self.produce(queue))
        self.owned_tasks = [producer, *workers]
        await producer
        await queue.join()  # Draining is part of measured time and goodput.
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def cancel_outstanding(self, mode: str) -> None:
        when = self.stop_at if self.stop_at is not None else perf_counter()
        if mode == "open_loop":
            # Preserve arrivals that were due before cancellation even if the
            # event loop failed to run the producer in time.  Otherwise client
            # scheduling stalls would hide offered load (coordinated omission).
            interval = 1.0 / self.config.rate_per_second
            for index in range(self.next_index, len(self.plan)):
                scheduled = self.started + index * interval
                if scheduled > when or scheduled >= self.deadline:
                    break
                if index not in self.samples:
                    self.offer(index, scheduled)
        for sample in self.samples.values():
            if sample.status not in TERMINAL_STATUSES:
                self.finish(sample, "cancelled", when, self.stop_reason or "cancelled",
                            censored=sample.dispatch_ms is not None)
        for task in [*self.owned_tasks, *self.rpc_tasks]:
            if not task.done():
                task.cancel()
            task.add_done_callback(_consume_task)


async def _run_phase(adapter: QueryAdapter, plan: list[QueryCase], config: RunConfig,
                     name: str, mode: str, deadline: float,
                     cancel_event: asyncio.Event | None,
                     progress: Callable | None) -> dict:
    phase = _Phase(adapter, plan, config, name, deadline, cancel_event, progress)
    if not plan:
        return {"samples": [], "duration_seconds": 0.0, "termination": None,
                "progress_errors": 0}
    phase.report(force=True)
    if phase.stopped():
        phase.report(force=True)
        return {"samples": [], "duration_seconds": 0.0, "termination": phase.stop_reason,
                "progress_errors": phase.progress_errors}
    work = asyncio.create_task(phase.work(mode))
    cancelled = asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
    waiting = {work}
    if cancelled is not None:
        waiting.add(cancelled)
    try:
        done, _ = await asyncio.wait(waiting, timeout=max(0.0, deadline - perf_counter()),
                                     return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            # Surface programming failures rather than reporting fabricated
            # measurements from a partially broken harness.
            work.result()
        elif cancelled is not None and cancelled in done:
            phase.stop("cancelled")
        else:
            phase.stop("max_run_seconds", deadline)
    except asyncio.CancelledError:
        # A caller cancelling the run task still receives terminal samples.
        phase.stop("cancelled")
    except BaseException:
        phase.stop("runner_error")
        phase.cancel_outstanding(mode)
        work.cancel()
        work.add_done_callback(_consume_task)
        raise
    finally:
        if cancelled is not None:
            cancelled.cancel()
            cancelled.add_done_callback(_consume_task)
    if phase.stop_reason is not None:
        phase.cancel_outstanding(mode)
        work.cancel()
        work.add_done_callback(_consume_task)
        # Deliver cancellation without awaiting possibly uncooperative external
        # RPCs.  All samples have already been finalized exactly once.
        await asyncio.sleep(0)
    finished = perf_counter()
    phase.report(force=True)
    return {
        "samples": [phase.samples[index] for index in sorted(phase.samples)],
        "duration_seconds": max(0.0, finished - phase.started),
        "termination": phase.stop_reason,
        "progress_errors": phase.progress_errors,
    }


async def run_benchmark(adapter: QueryAdapter, cases: list[QueryCase], config: RunConfig,
                        cancel_event: asyncio.Event | None = None,
                        progress: Callable | None = None) -> dict:
    """Run a warmup and a measured workload using an already-connected adapter.

    The callback receives compact progress dictionaries and may be synchronous
    or return an awaitable.  It must be nonblocking.  Warmup is closed-loop and
    has a separate seeded plan so changing warmup does not change the measured
    query sequence.  No request is ever retried.
    """
    if not cases:
        raise ValueError("at least one query case is required")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("query case IDs must be unique")
    for case in cases:
        if len(set(case.expected_ids)) > config.top_k:
            raise ValueError(
                f"query {case.id!r} has {len(set(case.expected_ids))} gold IDs "
                f"but top_k={config.top_k}; all-hit would be unreachable"
            )
    rng = random.Random(config.seed)
    plan = [cases[rng.randrange(len(cases))] for _ in range(config.requests)]
    warm_rng = random.Random(config.seed ^ 0x5741524D)
    warm_plan = [cases[warm_rng.randrange(len(cases))] for _ in range(config.warmup)]
    started = perf_counter()
    deadline = started + config.max_run_seconds
    warm = await _run_phase(adapter, warm_plan, config, "warmup", "closed_loop",
                            deadline, cancel_event, progress)
    if warm["termination"] is None:
        measured = await _run_phase(adapter, plan, config, "measurement", config.mode,
                                    deadline, cancel_event, progress)
    else:
        measured = {"samples": [], "duration_seconds": 0.0,
                    "termination": warm["termination"], "progress_errors": 0}
    termination = measured["termination"]
    samples = measured["samples"]
    summary = summarize_samples(samples, cases, config, measured["duration_seconds"],
                                termination=termination)
    summary["total_elapsed_seconds"] = max(0.0, perf_counter() - started)
    summary["max_run_seconds_scope"] = "warmup + measurement + drain"
    summary["completion"] = {
        "complete": termination is None and len(samples) == config.requests,
        "stop_reason": termination,
    }
    callback_errors = warm["progress_errors"] + measured["progress_errors"]
    if callback_errors:
        summary["warnings"].append(f"{callback_errors} progress callback invocation(s) failed.")
    warm_samples = warm["samples"]
    warm_counts = Counter(sample.status for sample in warm_samples)
    warm_summary = {
        "planned": config.warmup,
        "offered": len(warm_samples),
        "attempted": sum(sample.dispatch_ms is not None for sample in warm_samples),
        "counts": {status: warm_counts[status] for status in TERMINAL_STATUSES},
        "duration_seconds": warm["duration_seconds"],
        "termination": warm["termination"],
        "mode": "closed_loop",
        "included_in_measurements": False,
    }
    status = ("deadline_exceeded" if termination == "max_run_seconds" else
              "cancelled" if termination is not None else "completed")
    return {"status": status, "summary": summary, "samples": samples, "warmup": warm_summary}
