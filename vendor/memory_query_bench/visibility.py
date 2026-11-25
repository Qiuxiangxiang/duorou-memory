"""Optional sentinel visibility measurement; never touches existing records."""
from __future__ import annotations

import asyncio
import time
import uuid

from .models import QueryCase


async def measure_visibility(adapter, scope: str, timeout_ms: int, poll_interval_ms: int,
                             cancel_event: asyncio.Event, record_id: str | None = None) -> dict:
    if not scope.startswith("bench-") or not 6 <= len(scope) <= 128:
        raise ValueError("visibility writes require a dedicated bench-* scope")
    if not 10 <= poll_interval_ms <= 10000 or not 10 <= timeout_ms <= 60000:
        raise ValueError("invalid polling interval or timeout")
    record_id = record_id or "bench-probe-" + uuid.uuid4().hex
    text = "Visibility probe " + record_id
    case = QueryCase(id=record_id, query=text, scope=scope, expected_ids=[record_id])
    result = {"record_id": record_id, "scope": scope, "status": "running",
              "poll_interval_ms": poll_interval_ms, "cleanup": {"status": "not_attempted"},
              "notes": ["Visibility is observed through this one client/session, not every replica.",
                        "Polling brackets visibility; it does not reveal the exact commit instant.",
                        "Deletion visibility is not proof that backups or all storage copies are erased."]}
    write_started = False

    async def poll(expect_present: bool):
        started = time.perf_counter()
        previous_probe_start_ms = None
        probes = 0
        while (time.perf_counter() - started) * 1000 < timeout_ms:
            if cancel_event.is_set() and expect_present:
                return {"status": "cancelled", "probes": probes}
            probe_start = time.perf_counter()
            remaining = timeout_ms / 1000 - (probe_start - started)
            try:
                response = await asyncio.wait_for(adapter.query(case, 10), max(0.001, remaining))
            except TimeoutError:
                return {"status": "timeout", "probes": probes + 1}
            probes += 1
            present = record_id in response.ids
            if present == expect_present:
                return {"status": "observed", "probes": probes,
                        "last_opposite_probe_start_ms": previous_probe_start_ms,
                        "first_matching_probe_end_ms": (time.perf_counter() - started) * 1000}
            previous_probe_start_ms = (probe_start - started) * 1000
            await asyncio.sleep(min(poll_interval_ms / 1000,
                                    max(0, timeout_ms / 1000 - (time.perf_counter() - started))))
        return {"status": "timeout", "probes": probes}

    try:
        begin = time.perf_counter()
        write_started = True  # cleanup even if an ACK is lost after a successful write
        await asyncio.wait_for(adapter.upsert(record_id, text, scope), timeout_ms / 1000)
        result["write_ack_ms"] = (time.perf_counter() - begin) * 1000
        result["visibility"] = await poll(True)
        result["status"] = "completed" if result["visibility"]["status"] == "observed" else result["visibility"]["status"]
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
    finally:
        if write_started:
            try:
                begin = time.perf_counter()
                await asyncio.wait_for(adapter.delete(record_id, scope), timeout_ms / 1000)
                result["cleanup"] = {"status": "delete_acknowledged",
                                     "delete_ack_ms": (time.perf_counter() - begin) * 1000,
                                     "absence": await poll(False)}
                if result["cleanup"]["absence"]["status"] == "observed":
                    result["cleanup"]["status"] = "absence_observed"
                else:
                    result["status"] = "cleanup_failed"
            except Exception as exc:
                result["cleanup"] = {"status": "failed", "error_type": type(exc).__name__}
                result["status"] = "cleanup_failed"
    return result
