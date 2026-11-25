"""Reproducible LoCoMo evidence retrieval evaluation of the duorou prototype.

Run from the release root: python -m benchmarks.run --locomo /path/locomo10.json
All quality requests and all load samples cross the actual loopback MCP endpoint.
No extraction/answer/judge model is called. Existing output is never overwritten.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

import psutil

from examples.sqlite_demo import (LOCOMO_SHA256, TABLES, connect_database,
    import_locomo_records, initialize_schema, search_sources)
from public_memory_bench.evaluation import evaluate_once
from public_memory_bench.locomo import convert_locomo, DATA_URL, REVISION
from public_memory_bench.run import free_port, performance_rollup, ready, source_hash
from memory_query_bench.adapters import open_adapter
from memory_query_bench.config import Registry
from memory_query_bench.jobs import JobManager
from memory_query_bench.models import QueryCase, RunConfig, TargetConfig


METHODS = ("bm25_raw", "fts_raw", "bm25_enriched", "fts_enriched", "fts_enriched_neighbors")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def code_hashes() -> dict:
    return {
        "benchmark_core": source_hash(ROOT / "vendor/memory_query_bench"),
        "public_benchmark": source_hash(ROOT / "vendor/public_memory_bench"),
        "duorou_experiment": source_hash(ROOT / "benchmarks"),
        "schema_sql": digest(ROOT / "schema/schema.sql"),
        "existing_sqlite_demo": digest(ROOT / "examples/sqlite_demo.py"),
    }


def prepare(raw: Path, output: Path) -> dict:
    from benchmarks.retriever import build_enriched_documents

    if digest(raw) != LOCOMO_SHA256:
        raise ValueError("LoCoMo SHA256 does not match the pinned public dataset")
    records = json.loads(raw.read_text(encoding="utf-8"))
    converted = convert_locomo(records, include_captions=True)
    db_path = output / "inputs/duorou.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_database(db_path)
    began = time.perf_counter()
    recorded_at_us = time.time_ns() // 1000
    initialize_schema(conn)
    with conn:
        scopes = import_locomo_records(conn, records, recorded_at_us=recorded_at_us)
        conn.execute("INSERT INTO principals VALUES ('benchmark-reader', 'service')")
        conn.execute("INSERT INTO principals VALUES ('benchmark-denied', 'service')")
        conn.executemany("INSERT INTO scope_grants VALUES (?, 'benchmark-reader', 'read')", [(s,) for s in scopes])
    raw_documents = [{"id": row["scope_id"] + ":" + row["source_id"],
                      "scope": row["scope_id"], "text": row["content"]}
                     for row in conn.execute("SELECT scope_id, source_id, content FROM sources ORDER BY source_rowid")]
    enriched_documents = build_enriched_documents(conn)
    by_id = lambda docs: {row["id"]: row for row in docs}
    if by_id(enriched_documents) != by_id(converted["documents"]):
        raise AssertionError("Enriched projection is not identical to the historic baseline content")
    source_counts = {s: conn.execute("SELECT count(*) FROM sources WHERE scope_id=?", (s,)).fetchone()[0] for s in scopes}
    if len(raw_documents) != 5882 or len(scopes) != 10 or len(converted["queries"]) != 1527:
        raise AssertionError("Unexpected public corpus size")
    if conn.execute("PRAGMA foreign_key_check").fetchall():
        raise AssertionError("Foreign key violation")
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise AssertionError(integrity)
    conn.execute("INSERT INTO source_fts(source_fts, rank) VALUES ('integrity-check', 1)")
    conn.commit()
    denied = 0
    for scope in scopes:
        try:
            search_sources(conn, authenticated_principal="benchmark-denied", scope_id=scope, query="support", limit=20)
        except PermissionError:
            denied += 1
        else:
            raise AssertionError("Denied principal returned data")
    tables = {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in TABLES}
    conn.close()
    files = {
        "raw_documents": output / "inputs/raw.documents.jsonl",
        "enriched_documents": output / "inputs/enriched.documents.jsonl",
        "queries": output / "inputs/queries.jsonl",
        "metadata": output / "inputs/query_metadata.json",
        "exclusions": output / "inputs/exclusions.json",
    }
    save_jsonl(files["raw_documents"], raw_documents)
    save_jsonl(files["enriched_documents"], converted["documents"])
    save_jsonl(files["queries"], converted["queries"])
    save(files["metadata"], converted["metadata"])
    save(files["exclusions"], converted["exclusions"])
    manifest = {
        "dataset": "LoCoMo-10", "revision": REVISION, "url": DATA_URL,
        "raw_sha256": digest(raw), "raw_bytes": raw.stat().st_size,
        "license": "CC-BY-NC-4.0", "summary": converted["summary"],
        "db": str(db_path.relative_to(output)), "db_sha256": digest(db_path),
        "db_bytes": db_path.stat().st_size, "recorded_at_us": recorded_at_us,
        "import_and_checks_seconds": time.perf_counter() - began,
        "table_counts": tables, "documents_per_scope": source_counts,
        "enriched_projection_all_documents_match_original_baseline": True,
        "denied_principal_scope_checks": denied, "sqlite_integrity": integrity,
        "foreign_key_violations": 0, "fts_integrity": "passed",
        "files": {key: {"path": str(path.relative_to(output)), "sha256": digest(path), "bytes": path.stat().st_size}
                  for key, path in files.items()},
        "target_input_allowlist": ["source raw text", "source scope/IDs", "speaker/date/caption metadata", "source-session memberships", "local scope grants"],
        "never_indexed": ["question", "answer", "gold evidence", "QA category", "summaries", "observations"],
    }
    save(output / "dataset.json", manifest)
    return {"manifest": manifest, "db": db_path, "files": files,
            "doc_scopes": {row["id"]: row["scope"] for row in raw_documents},
            "metadata": converted["metadata"]}


async def suite(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    order = list(METHODS)
    random.Random(20261124).shuffle(order)
    protocol = {
        "release": "duorou 0.2.2", "schema_version": "0.2",
        "written_before_quality_execution": True, "methods": order,
        "quality": "all 1527 eligible questions once per method, failures retained in denominator",
        "ks": [1, 5, 10, 20], "requested_top_k": 20, "leaf_evidence_only": True,
        "raw": "source.content only; FTS arm calls original search_sources including snippet computation",
        "enriched": "identical Session timestamp + Speaker + Text + optional Image caption strings for both engines",
        "neighbors": {"introduced_this_release": True, "seed_count": 5, "preserve_first": 10,
            "candidate_limit": 100, "order": "first10 lexical, predecessor then successor of each first5 seed within same source_session, then remaining lexical; dedupe; final prefix20",
            "semantic_event_extraction": False, "parameter_tuning": False},
        "ranking_confounds": "SQLite unicode61/global FTS statistics versus sklearn English stopword filtering/per-scope BM25 statistics; engine comparisons are not schema ablations",
        "budget": "same maximum20 returned source IDs, NOT equal LLM token count; context characters/word counts reported diagnostically",
        "transport": "MCP Streamable HTTP over loopback", "response": "IDs and scores only",
        "server": "single target process; synchronous search in async MCP handler; trusted local service principal; no production login",
        "query_result_cache": False, "quality_concurrency": 8,
        "performance": {"load": "closed_loop", "concurrencies": args.concurrency,
            "repeats": args.repeats, "requests_per_phase": args.requests, "warmup": 30,
            "seed": 42, "goodput": "all gold IDs returned AND sojourn<=200ms", "slo_ms": 200,
            "execution": "all target/load phases sequential; not a saturation or calibrated capacity test"},
        "model_calls": 0, "official_qa_score": False,
        "evaluation_status": "development evaluation on previously inspected public data, not an unseen holdout",
    }
    save(output / "protocol.json", protocol)
    hashes = code_hashes()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "source_hashes_before": hashes,
        "environment": {"python": sys.version, "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version, "cpu_count": os.cpu_count(), "memory_bytes": psutil.virtual_memory().total,
            "packages": {name: importlib.metadata.version(name) for name in ("mcp", "httpx", "pydantic", "numpy", "scipy", "scikit-learn", "psutil")},
            "cgroup": {name: Path('/sys/fs/cgroup', name).read_text().strip() for name in ("cpu.max", "memory.max") if Path('/sys/fs/cgroup', name).exists()}},
        "quality": [], "performance": [], "index_reports": []}
    prepared = prepare(args.locomo.resolve(), output)
    report["dataset"] = prepared["manifest"]
    cases = [QueryCase.model_validate_json(line) for line in prepared["files"]["queries"].read_text().splitlines()]
    if len({c.id for c in cases}) != len(cases) or max(len(c.expected_ids) for c in cases) > 20:
        raise AssertionError("Invalid query/gold count")
    if not all(prepared["doc_scopes"].get(doc) == case.scope for case in cases for doc in case.expected_ids):
        raise AssertionError("Gold references outside their declared scope")
    for method in order:
        print(f"BUILD {method}", flush=True)
        port = free_port()
        index_path = output / "indexes" / f"{method}.json"
        documents = prepared["files"]["raw_documents" if method.endswith("_raw") else "enriched_documents"]
        env = os.environ.copy()
        env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
        process = None
        manager = None
        startup = time.perf_counter()
        try:
            with (output / f"{method}.target.log").open("w") as target_log:
                command = [sys.executable, "-m", "benchmarks.target", "--db", str(prepared["db"]),
                    "--method", method, "--port", str(port), "--index-report", str(index_path)]
                if method.startswith("bm25"):
                    command.extend(["--documents", str(documents)])
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=target_log, stderr=target_log)
                await ready(process, port)
                target = TargetConfig(kind="mcp_http", url=f"http://127.0.0.1:{port}/mcp",
                    revision=f"duorou-0.2.2-{hashes['duorou_experiment'][:16]}-{method}",
                    description="Duorou public-data evidence retrieval test target")
                index_info = json.loads(index_path.read_text())
                index_info.update(dataset="locomo", method=method,
                    startup_ready_seconds=time.perf_counter()-startup,
                    rss_after_start_bytes=psutil.Process(process.pid).memory_info().rss)
                report["index_reports"].append(index_info)
                save(index_path, index_info)
                print(f"QUALITY {method}: {len(cases)} questions", flush=True)
                async with open_adapter(target, concurrency=8) as adapter:
                    quality = await evaluate_once(adapter, cases, prepared["metadata"], prepared["doc_scopes"],
                        output / "quality" / f"{method}.predictions.jsonl", concurrency=8)
                quality.update(dataset="locomo", method=method)
                report["quality"].append(quality)
                save(output / "quality" / f"{method}.summary.json", quality)
                print(f"QUALITY_DONE {method}: R20={quality['overall']['at_k']['20']['recall']:.6f} AllHit20={quality['overall']['at_k']['20']['all_hit']:.6f} errors={quality['overall']['errors']}", flush=True)
                config_path = output / f"{method}.registry.json"
                save(config_path, {"targets": {method: target.model_dump()}, "datasets": {"locomo": str(prepared["files"]["queries"])}})
                manager = JobManager(Registry(config_path), output / "performance")
                for repeat in range(args.repeats):
                    levels = args.concurrency if repeat % 2 == 0 else list(reversed(args.concurrency))
                    for concurrency in levels:
                        cfg = RunConfig(target_id=method, dataset_id="locomo", requests=args.requests,
                            concurrency=concurrency, top_k=20, warmup=30, seed=42,
                            slo_ms=200, min_recall=1.0, max_run_seconds=180)
                        started = manager.start(cfg)
                        await manager.jobs[started["run_id"]].task
                        result = manager.report(started["run_id"])
                        if result["status"] != "completed":
                            raise RuntimeError(f"Incomplete performance phase: {started['run_id']}: {result['status']}")
                        report["performance"].append({"dataset": "locomo", "method": method,
                            "repeat": repeat+1, "concurrency": concurrency, "run_id": started["run_id"], "summary": result["summary"]})
                        print(f"PERF {method} C={concurrency} repeat={repeat+1} QPS={result['summary']['throughput_rps']:.1f}", flush=True)
                        save(output / "checkpoint.json", report)
        finally:
            if manager is not None:
                await manager.close()
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    await asyncio.to_thread(process.wait, timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait, timeout=5)
    report["source_hashes_after"] = code_hashes()
    if hashes != report["source_hashes_after"]:
        raise RuntimeError("Experiment source changed during execution")
    report["db_hash_after"] = digest(prepared["db"])
    if report["db_hash_after"] != report["dataset"]["db_sha256"]:
        raise RuntimeError("Read-only target modified the source database")
    report["performance_rollup"] = performance_rollup(report["performance"])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(output / "suite.json", report)
    print(f"COMPLETE {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/locomo-20261124")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.requests <= 20000 or any(not 1 <= c <= 64 for c in args.concurrency):
        parser.error("Invalid load settings")
    asyncio.run(suite(args))


if __name__ == "__main__":
    main()
