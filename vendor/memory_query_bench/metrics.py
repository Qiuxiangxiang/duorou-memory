"""Quality and timing metrics for an external memory service.

This module does not know the implementation of the service under test.  A
timeout is a censored observation, not a successful request taking exactly the
timeout duration.  Durations to terminal states are therefore reported apart
from successful response latency.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .models import QueryCase, QueryResult, RunConfig, Sample


TERMINAL_STATUSES = ("success", "error", "timeout", "cancelled", "dropped")
MIN_P99_SAMPLES = 1000


class ProtocolError(ValueError):
    """The response cannot be evaluated as a valid top-k result."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def nearest_rank(values: Iterable[float], percentile: float) -> float | None:
    """Return the nearest-rank percentile, with no interpolation.

    ``percentile`` is on the 0..100 scale.  Empty samples have no percentile;
    reporting zero would make a service returning no responses look fast.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("durations must be finite and nonnegative")
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def distribution(values: Iterable[float]) -> dict:
    """Describe a duration sample.  Percentiles are nearest rank."""
    observed = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in observed):
        raise ValueError("durations must be finite and nonnegative")
    ordered = sorted(observed)
    size = len(ordered)

    def at(percentile: float) -> float | None:
        return ordered[max(0, math.ceil(percentile * size) - 1)] if size else None

    return {
        "count": size,
        "min": ordered[0] if size else None,
        "mean": sum(ordered) / size if size else None,
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1] if size else None,
        "method": "nearest_rank",
    }


def score_quality(case: QueryCase, result: QueryResult, top_k: int,
                  min_recall: float = 1.0) -> dict:
    """Validate the response and evaluate independently supplied gold IDs.

    A positive-gold query must retrieve at least one gold item, including when
    the configured recall floor is zero.  Expected-empty cases are evaluated
    separately and are never assigned an artificial recall of 1.
    """
    if not isinstance(result, QueryResult) or not isinstance(result.ids, list):
        raise ProtocolError("protocol_result_type")
    ids = result.ids
    if any(not isinstance(item, str) or not item for item in ids):
        raise ProtocolError("protocol_invalid_id")
    if len(ids) != len(set(ids)):
        raise ProtocolError("protocol_duplicate_ids")
    if len(ids) > top_k:
        raise ProtocolError("protocol_over_top_k")
    if (not isinstance(result.response_bytes, int)
            or isinstance(result.response_bytes, bool)
            or result.response_bytes < 0):
        raise ProtocolError("protocol_response_bytes")

    returned = set(ids)
    forbidden_count = len(returned & set(case.forbidden_ids))
    gold = set(case.expected_ids)
    if case.expect_empty:
        return {
            "recall": None,
            "any_hit": None,
            "all_hit": None,
            "quality_ok": not returned and forbidden_count == 0,
            "forbidden_count": forbidden_count,
            "result_count": len(ids),
            "response_bytes": result.response_bytes,
        }
    if not gold:
        raise ValueError("positive query must have gold IDs")
    if len(gold) > top_k:
        raise ValueError(f"query {case.id!r} has more gold IDs than top_k")
    intersection = returned & gold
    recall = len(intersection) / len(gold)
    return {
        "recall": recall,
        "any_hit": bool(intersection),
        "all_hit": gold <= returned,
        "quality_ok": bool(intersection) and recall >= min_recall and forbidden_count == 0,
        "forbidden_count": forbidden_count,
        "result_count": len(ids),
        "response_bytes": result.response_bytes,
    }


def summarize_samples(samples: list[Sample], cases: list[QueryCase],
                      config: RunConfig, duration_seconds: float,
                      *, termination: str | None = None) -> dict:
    """Summarize *offered* requests; queue drops remain in rate denominators."""
    case_by_id = {case.id: case for case in cases}
    counts = {status: 0 for status in TERMINAL_STATUSES}
    for sample in samples:
        if sample.status not in counts:
            raise ValueError(f"non-terminal sample: {sample.status!r}")
        if sample.query_id not in case_by_id:
            raise ValueError(f"unknown query id: {sample.query_id!r}")
        counts[sample.status] += 1
    offered = len(samples)
    attempted = [sample for sample in samples if sample.dispatch_ms is not None]
    successful = [sample for sample in samples if sample.status == "success"]
    quality_pass = [sample for sample in successful if sample.quality_ok]
    quality_failures = [sample for sample in successful if not sample.quality_ok]
    positive = [sample for sample in samples if not case_by_id[sample.query_id].expect_empty]
    positive_successes = [sample for sample in positive if sample.status == "success"]
    expected_empty = [sample for sample in samples if case_by_id[sample.query_id].expect_empty]
    empty_successes = [sample for sample in expected_empty if sample.status == "success"]
    forbidden_samples = [sample for sample in samples if sample.forbidden_count > 0]
    protocol_errors = sum(
        sample.status == "error" and
        ((sample.error_type or "").startswith("protocol_") or
         sample.error_type in {"ResponseSchemaError", "ResponseTooLarge"})
        for sample in samples
    )
    duration_seconds = max(0.0, float(duration_seconds))

    def ratio(numerator: int | float, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def rate(numerator: int) -> float:
        return numerator / duration_seconds if duration_seconds > 0 else 0.0

    def times(rows: list[Sample], attribute: str) -> dict:
        return distribution(getattr(row, attribute) for row in rows
                            if getattr(row, attribute) is not None)

    success_latency = times(successful, "latency_ms")
    success_sojourn = times(successful, "sojourn_ms")
    good = [sample for sample in quality_pass
            if sample.sojourn_ms is not None and sample.sojourn_ms <= config.slo_ms]
    offered_failures = offered - len(successful)
    error_rate = ratio(offered_failures, offered)
    good_fraction = ratio(len(good), offered)
    # Error and quality gates are deliberately independent.  A transport error
    # is not mislabeled a successful response that failed the recall threshold.
    error_gate_ok = offered > 0 and error_rate <= config.max_error_rate
    quality_gate_ok = bool(successful) and not quality_failures and not forbidden_samples
    latency_gate_ok = (success_sojourn["p99"] is not None and
                       success_sojourn["p99"] <= config.slo_ms)
    sufficient = len(successful) >= MIN_P99_SAMPLES
    complete = termination is None and offered == config.requests
    failures = []
    if offered and not error_gate_ok:
        failures.append("offered_failure_rate")
    if quality_failures or forbidden_samples:
        failures.append("quality")
    if success_sojourn["p99"] is not None and not latency_gate_ok:
        failures.append("p99_sojourn")
    if failures:
        verdict = "fail"
    elif not complete:
        verdict = "incomplete"
    elif not sufficient:
        verdict = "insufficient_samples"
    else:
        verdict = "pass_observed"
    warnings = []
    if not sufficient:
        warnings.append(
            f"p99 has {len(successful)} successful observations; fewer than "
            f"{MIN_P99_SAMPLES} does not meet this tool's observation threshold. "
            "Meeting this threshold does not certify an SLO or SLA."
        )
    censored = sum(sample.censored for sample in attempted)
    if censored:
        warnings.append(
            "Timeout and interrupted RPC durations are censored lower bounds, "
            "not observed successful response latencies."
        )
    if counts["dropped"]:
        warnings.append("Queue-full drops remain in offered-request denominators.")
    if not complete:
        warnings.append("The run did not complete the planned workload.")
    if not successful:
        warnings.append("No valid response was observed; latency and quality cannot pass.")

    quality = {
        "evaluated_successes": len(successful),
        "pass_count": len(quality_pass),
        "failure_count": len(quality_failures),
        "failure_rate_on_successes": ratio(len(quality_failures), len(successful)),
        "pass_rate_offered": ratio(len(quality_pass), offered),
        "positive_offered": len(positive),
        "positive_successes": len(positive_successes),
        "mean_recall": ratio(sum(sample.recall or 0.0 for sample in positive), len(positive)),
        "mean_recall_on_successes": ratio(sum(sample.recall or 0.0 for sample in positive_successes),
                                          len(positive_successes)),
        "any_hit_rate": ratio(sum(sample.any_hit is True for sample in positive), len(positive)),
        "all_hit_rate": ratio(sum(sample.all_hit is True for sample in positive), len(positive)),
        "expected_empty": {
            "offered": len(expected_empty),
            "successful_responses": len(empty_successes),
            "correct": sum(sample.quality_ok for sample in empty_successes),
            "pass_rate_offered": ratio(sum(sample.quality_ok for sample in empty_successes),
                                       len(expected_empty)),
        },
        "forbidden_response_count": len(forbidden_samples),
        "forbidden_id_count": sum(sample.forbidden_count for sample in samples),
        "protocol_error_count": protocol_errors,
        "denominator_note": "Recall/any-hit/all-hit use positive-gold offered requests; failures and drops count as zero. Expected-empty cases are separate.",
    }
    return {
        "mode": config.mode,
        "seed": config.seed,
        "planned": config.requests,
        "offered": offered,
        "attempted": len(attempted),
        "not_offered": config.requests - offered,
        "completed": offered,
        "counts": counts,
        "duration_seconds": duration_seconds,
        "duration_ms": duration_seconds * 1000,
        "termination": termination,
        "throughput_rps": rate(len(successful)),
        "attempted_rps": rate(len(attempted)),
        "offered_rps": rate(offered),
        "goodput_rps": rate(len(good)),
        "good_count": len(good),
        "good_fraction_offered": good_fraction,
        "error_rate": error_rate,
        "rpc_error_rate": ratio(sum(sample.status != "success" for sample in attempted), len(attempted)),
        "error_types": dict(Counter(sample.error_type for sample in samples if sample.error_type)),
        "censored_count": censored,
        "latency_ms": {
            "success_only": success_latency,
            "quality_pass_only": times(quality_pass, "latency_ms"),
            "all_attempted_terminal": times(attempted, "latency_ms"),
        },
        "sojourn_ms": {
            "success_only": success_sojourn,
            "all_attempted_terminal": times(attempted, "sojourn_ms"),
            "all_offered_terminal": times(samples, "sojourn_ms"),
        },
        "queue_ms": times(attempted, "queue_ms"),
        "quality": quality,
        "slo": {
            "threshold_ms": config.slo_ms,
            "basis": "successful-response p99 sojourn plus offered-error and response-quality gates",
            "verdict": verdict,
            "certified": False,
            "observed_pass": verdict == "pass_observed",
            "sufficient_samples": sufficient,
            "minimum_success_samples": MIN_P99_SAMPLES,
            "error_gate_ok": error_gate_ok,
            "quality_gate_ok": quality_gate_ok,
            "latency_gate_ok": latency_gate_ok,
            "complete_workload": complete,
            "failed_gates": failures,
            "scope": "This observed workload only; not a statistical or production guarantee.",
        },
        "warnings": warnings,
        "timing_note": "perf_counter monotonic clock; dispatch-to-finish RPC latency; scheduled-to-finish sojourn includes queue/scheduler delay; duration includes final drain. All-attempted terminal times include censored failures.",
    }
