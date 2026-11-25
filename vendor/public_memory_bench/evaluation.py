"""Exhaustive, macro-averaged retrieval quality; distinct from load sampling."""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path


KS = (1, 5, 10, 20)


def score_ranking(expected_ids: list[str], ranked_ids: list[str], k: int) -> dict:
    gold = set(expected_ids)
    if not gold:
        raise ValueError("Retrieval quality requires a nonempty complete gold set")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Duplicate result IDs")
    top = ranked_ids[:k]
    found = gold.intersection(top)
    first = next((rank for rank, doc_id in enumerate(top, 1) if doc_id in gold), None)
    return {"recall": len(found) / len(gold), "any_hit": float(bool(found)),
            "all_hit": float(found == gold), "mrr": 1 / first if first else 0.0}


def aggregate(rows: list[dict]) -> dict:
    output = {"queries": len(rows), "errors": sum(row["status"] != "success" for row in rows)}
    output["at_k"] = {
        str(k): {metric: sum(row["at_k"][str(k)][metric] for row in rows) / len(rows)
                 if rows else None for metric in ("recall", "any_hit", "all_hit", "mrr")}
        for k in KS
    }
    return output


async def evaluate_once(adapter, cases, metadata: dict, doc_scopes: dict[str, str],
                        destination: Path, *, concurrency: int = 8) -> dict:
    """Every eligible question exactly once. Failures remain in every denominator."""
    order = list(range(len(cases)))
    random.Random(20261123).shuffle(order)
    rows: list[dict | None] = [None] * len(cases)
    queue: asyncio.Queue = asyncio.Queue()
    for index in order:
        queue.put_nowait(index)

    async def worker():
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            case = cases[index]
            start = time.perf_counter()
            ids = []
            status, error = "success", None
            size = 0
            try:
                result = await asyncio.wait_for(adapter.query(case, max(KS)), timeout=10)
                ids, size = result.ids, result.response_bytes
                if len(ids) > max(KS) or len(ids) != len(set(ids)):
                    raise ValueError("Invalid returned ranking")
                if any(doc_scopes.get(doc_id) != case.scope for doc_id in ids):
                    raise ValueError("Returned ID outside requested corpus")
            except Exception as exc:
                status, error, ids = "error", type(exc).__name__, []
            elapsed_ms = (time.perf_counter() - start) * 1000
            row = {"query_id": case.id, "scope": case.scope,
                   "category": metadata[case.id].get("category_name", metadata[case.id]["category"]),
                   "status": status, "error_type": error,
                   "gold_ids": case.expected_ids, "ranked_ids": ids,
                   "latency_ms": elapsed_ms, "response_bytes": size,
                   "at_k": {str(k): score_ranking(case.expected_ids, ids, k) for k in KS}}
            rows[index] = row
            queue.task_done()

    began = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - began
    assert all(row is not None for row in rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    categories, scopes = defaultdict(list), defaultdict(list)
    for row in rows:
        categories[str(row["category"])].append(row)
        scopes[row["scope"]].append(row)
    return {"protocol": "Streamable HTTP MCP", "top_k_requested": max(KS),
            "quality_pass": "exhaustive once, not sampled with replacement",
            "wall_seconds": elapsed, "concurrency": concurrency,
            "overall": aggregate(rows),
            "categories": {key: aggregate(value) for key, value in sorted(categories.items())},
            "scopes": {key: aggregate(value) for key, value in sorted(scopes.items())},
            "predictions_file": str(destination.name),
            "note": "Evidence retrieval only, not generated-answer accuracy; k<20 metrics derive from the top-20 ranking."}
