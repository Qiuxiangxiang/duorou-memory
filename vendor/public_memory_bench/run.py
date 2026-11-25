"""Run public-dataset retrieval quality and the unchanged v0.1 load engine."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .evaluation import evaluate_once


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def ready(process: subprocess.Popen, port: int, timeout: float = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Target exited before readiness: {process.returncode}")
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(.1)
    raise TimeoutError("Target index/startup exceeded readiness budget")


def performance_rollup(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["concurrency"])].append(row)
    output = []
    for (dataset, method, concurrency), group in sorted(grouped.items()):
        summaries = [row["summary"] for row in group]
        def values(key):
            return [summary[key] for summary in summaries]
        p95 = [s["sojourn_ms"]["success_only"]["p95"] for s in summaries]
        p99 = [s["sojourn_ms"]["success_only"]["p99"] for s in summaries]
        output.append({"dataset": dataset, "method": method, "concurrency": concurrency,
                       "repeats": len(group), "run_ids": [r["run_id"] for r in group],
                       "qps_median": statistics.median(values("throughput_rps")),
                       "qps_min": min(values("throughput_rps")),
                       "qps_max": max(values("throughput_rps")),
                       "goodput_median": statistics.median(values("goodput_rps")),
                       "good_fraction_median": statistics.median(values("good_fraction_offered")),
                       "p95_ms_median_of_runs": statistics.median(p95),
                       "p99_ms_range": [min(p99), max(p99)],
                       "errors": sum(s["counts"]["error"] + s["counts"]["timeout"] for s in summaries),
                       "drops": sum(s["counts"]["dropped"] for s in summaries),
                       "note": "Medians/ranges across independent load phases; quantiles are not pooled or averaged."})
    return output


def render_report(report: dict, destination: Path) -> None:
    lines = ["# 公开数据集记忆检索 Benchmark", "",
             "本报告为真实执行结果。数据来自固定版本的LoCoMo 10及LongMemEval-S cleaned；目标为本项目的BM25、TF-IDF检索基线。没有生成回答或调用LLM judge，分数不是官方问答准确率。", "",
             "## 数据与口径", "",
             "| 数据集 | 检索文档 | 作用域 | 纳入问题 | 排除问题 | 检索单位 |",
             "|---|---:|---:|---:|---:|---|"]
    for name, data in report["datasets"].items():
        lines.append(f"| {name} | {data['documents']} | {data['scopes']} | {data['eligible_queries']} | {data['excluded_queries']} | {'对话回合（含上游非空caption）' if name == 'locomo' else '会话'} |")
    lines += ["", "每个纳入问题在质量评测中查询一次。top_k=20，R@1/5/10来自同一排名前缀；完整gold不因k缩小而被截断。Recall为每题证据覆盖比例的宏平均，All-Hit为找齐全部证据的问题比例，MRR为首条证据的倒数排名。未标注片段不等同于不相关。", "",
              "## 全量检索质量", "",
              "| 数据集 | 基线 | 题数 | Recall@5 | Recall@10 | Recall@20 | Any-Hit@20 | All-Hit@20 | MRR@20 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in report["quality"]:
        s, k = item["overall"], item["overall"]["at_k"]
        lines.append(f"| {item['dataset']} | {item['method']} | {s['queries']} | {k['5']['recall']:.2%} | {k['10']['recall']:.2%} | {k['20']['recall']:.2%} | {k['20']['any_hit']:.2%} | {k['20']['all_hit']:.2%} | {k['20']['mrr']:.4f} |")
    lines += ["", "## 分类型检索结果", "",
              "| 数据集 | 基线 | 类型 | 题数 | Recall@10 | All-Hit@20 |",
              "|---|---|---|---:|---:|---:|"]
    for item in report["quality"]:
        for name, row in item["categories"].items():
            lines.append(f"| {item['dataset']} | {item['method']} | {name} | {row['queries']} | {row['at_k']['10']['recall']:.2%} | {row['at_k']['20']['all_hit']:.2%} |")
    lines += ["", "## 当前v0.1引擎的工程测试", "",
              "下表是top20、ID+score返回、同机loopback Streamable HTTP MCP、预热后固定并发的结果；每档重复3次（若复现参数另设，以JSON记录为准）。返回QPS与要求找齐gold且200ms内返回的goodput分别列出。p95为各次p95的中位数，p99给出各次范围。", "",
              "| 数据集 | 基线 | 并发 | 返回QPS中位 | QPS范围 | 达标goodput中位 | p95中位(ms) | p99范围(ms) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["performance_rollup"]:
        lo, hi = row["p99_ms_range"]
        lines.append(f"| {row['dataset']} | {row['method']} | {row['concurrency']} | {row['qps_median']:.1f} | {row['qps_min']:.1f}–{row['qps_max']:.1f} | {row['goodput_median']:.1f} | {row['p95_ms_median_of_runs']:.2f} | {lo:.2f}–{hi:.2f} |")
    lines += ["", "## 边界", "",
              "- 两个数据集的检索单位、每题候选规模不同，不跨数据集比较分数高低。LongMemEval每题只检索自身会话库；全局文档总数不等于单题候选数。",
              "- LoCoMo的446条adversarial及无效/缺失gold单列排除；LongMemEval的abstention单列排除，不把不可回答错误定义为检索必须为空。详见data/converted中的manifest/exclusions。",
              "- 质量评测为全量一次；压力测试由原版引擎固定seed有放回抽样，不能把压力样本当成全量质量分母。",
              "- 相同查询可被多次调用，但目标没有结果缓存；索引已在内存。没有强制清系统缓存，不报告冷启动查询性能。",
              "- 使用相同的固定词法参数，无神经embedding、重排、记忆提取或答案生成。没有根据评测分数调参。",
              "- 同一目标进程内顺序进行重复负载阶段；不是独立机器、独立进程重启的统计复现。阶段顺序写入suite.json。",
              "- 当前单进程目标与发压器在同一宿主运行；未完成发压器自校准，不能宣称最大可持续容量或生产SLA。",
              "- MCP响应只含ID和score，未测完整上下文返回、LLM token成本或任务成功率。",
              "- 查询与基线代码、环境版本、原始数据SHA、全部逐题排名和逐请求样本均有记录。", "",
              "## 重现", "",
              "参见README.md。suite.json为总索引，quality/*.predictions.jsonl为逐题排名，performance/<run_id>/包含原版引擎的manifest/report/samples。"]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def suite(args) -> None:
    root, engine = args.root.resolve(), args.engine_root.resolve()
    sys.path.insert(0, str(engine))
    from memory_query_bench.adapters import open_adapter
    from memory_query_bench.config import Registry
    from memory_query_bench.jobs import JobManager
    from memory_query_bench.models import QueryCase, RunConfig, TargetConfig

    output = root / "reports" / args.output_name
    if (output / "suite.json").exists():
        raise FileExistsError("Choose a new --output-name; existing results are not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    before_hash = source_hash(engine / "memory_query_bench")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "core_source_sha256": before_hash,
              "experiment_source_sha256": source_hash(root / "public_memory_bench"),
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "cpu_count": os.cpu_count(), "memory_bytes": psutil.virtual_memory().total,
                              "packages": {name: importlib.metadata.version(name) for name in
                                           ("mcp", "httpx", "pydantic", "numpy", "scipy", "scikit-learn", "psutil")}},
              "protocol": {"top_k": 20, "repeats": args.repeats, "requests_per_phase": args.requests,
                           "concurrencies": args.concurrency, "seed": 42, "quality_seed": 20261123,
                           "slo_ms": 200, "min_recall": 1.0, "quality_k": [1, 5, 10, 20],
                           "core_unchanged": True, "query_result_cache": False,
                           "target_execution": "single process, synchronous sparse retrieval in event loop",
                           "load_type": "closed_loop", "transport": "MCP Streamable HTTP, loopback",
                           "response": "IDs and scores only", "tuning": "none; fixed parameters before evaluation"},
              "datasets": {}, "quality": [], "performance": [], "index_reports": []}

    for dataset in args.datasets:
        converted = root / "data" / "converted"
        query_path = converted / f"{dataset}.queries.jsonl"
        document_path = converted / f"{dataset}.documents.jsonl"
        cases = [QueryCase.model_validate_json(line) for line in query_path.read_text().splitlines() if line.strip()]
        if not cases or max(len(case.expected_ids) for case in cases) > 20:
            raise ValueError("All nonempty gold sets must fit top20 for the unchanged v0.1 engine")
        metadata = json.loads((converted / f"{dataset}.query_metadata.json").read_text())
        doc_scopes = {}
        with document_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if set(row) != {"id", "scope", "text"} or row["id"] in doc_scopes:
                    raise ValueError("Invalid document fields or duplicate document ID")
                doc_scopes[row["id"]] = row["scope"]
        assert len({case.id for case in cases}) == len(cases)
        assert all(doc_scopes.get(doc_id) == case.scope for case in cases for doc_id in case.expected_ids)
        raw_count = 1986 if dataset == "locomo" else 500
        manifest = json.loads((converted / f"{dataset}.manifest.json").read_text())
        report["datasets"][dataset] = {"documents": len(doc_scopes), "scopes": len(set(doc_scopes.values())),
                                       "eligible_queries": len(cases), "excluded_queries": raw_count - len(cases),
                                       "manifest": manifest}
        for method in args.methods:
            label = f"{dataset}-{method}"
            print(f"BUILD {label}", flush=True)
            port = free_port()
            index_path = output / "indexes" / f"{label}.json"
            log_path = output / f"{label}.target.log"
            env = os.environ.copy()
            env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
            startup = time.perf_counter()
            process = None
            manager = None
            try:
                with log_path.open("w", encoding="utf-8") as target_log:
                    process = subprocess.Popen([sys.executable, "-m", "public_memory_bench.target",
                                                "--documents", str(document_path), "--method", method,
                                                "--port", str(port), "--index-report", str(index_path)],
                                               cwd=root, env=env, stdout=target_log, stderr=target_log)
                    await ready(process, port)
                    target = TargetConfig(kind="mcp_http", url=f"http://127.0.0.1:{port}/mcp",
                                          revision=f"{method}-{report['experiment_source_sha256'][:16]}",
                                          description="Public-dataset sparse retrieval baseline")
                    index_info = json.loads(index_path.read_text())
                    index_info.update(dataset=dataset, startup_ready_seconds=time.perf_counter() - startup,
                                      rss_after_start_bytes=psutil.Process(process.pid).memory_info().rss)
                    report["index_reports"].append(index_info)
                    save(index_path, index_info)
                    print(f"QUALITY {label}: {len(cases)} questions", flush=True)
                    async with open_adapter(target, concurrency=8) as adapter:
                        quality = await evaluate_once(adapter, cases, metadata, doc_scopes,
                                                      output / "quality" / f"{label}.predictions.jsonl")
                    quality.update(dataset=dataset, method=method)
                    report["quality"].append(quality)
                    save(output / "quality" / f"{label}.summary.json", quality)
                    print(f"QUALITY_DONE {label}: R20={quality['overall']['at_k']['20']['recall']:.4f}, errors={quality['overall']['errors']}", flush=True)
                    config_path = output / f"{label}.registry.json"
                    save(config_path, {"targets": {label: target.model_dump()},
                                       "datasets": {dataset: str(query_path)}})
                    registry = Registry(config_path)
                    manager = JobManager(registry, output / "performance")
                    # Alternate concurrency order across repeats to reduce a simple monotonic order effect.
                    for repeat in range(args.repeats):
                        levels = args.concurrency if repeat % 2 == 0 else list(reversed(args.concurrency))
                        for concurrency in levels:
                            cfg = RunConfig(target_id=label, dataset_id=dataset, requests=args.requests,
                                            concurrency=concurrency, top_k=20, warmup=30, seed=42,
                                            slo_ms=200, min_recall=1, max_run_seconds=180)
                            started = manager.start(cfg)
                            await manager.jobs[started["run_id"]].task
                            result = manager.report(started["run_id"])
                            if result["status"] != "completed":
                                raise RuntimeError(f"Incomplete performance run {started['run_id']}: {result['status']}")
                            row = {"dataset": dataset, "method": method, "repeat": repeat + 1,
                                   "concurrency": concurrency, "run_id": started["run_id"],
                                   "summary": result["summary"]}
                            report["performance"].append(row)
                            print(f"PERF {label} C={concurrency} repeat={repeat+1} QPS={result['summary']['throughput_rps']:.1f}", flush=True)
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
            # Temporary URLs/ports are intentionally retained in the local registry for audit,
            # but rerunning the suite allocates fresh targets and does not reuse these endpoints.
    after_hash = source_hash(engine / "memory_query_bench")
    if after_hash != before_hash:
        raise RuntimeError("Original benchmark engine changed during experiment")
    report["core_hash_verified_unchanged"] = True
    report["performance_rollup"] = performance_rollup(report["performance"])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(output / "suite.json", report)
    render_report(report, output / "REPORT.md")
    print(f"COMPLETE {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--engine-root", type=Path, default=Path(__file__).resolve().parents[1] / "engine")
    p.add_argument("--output-name", default="public-benchmark-v1")
    p.add_argument("--datasets", nargs="+", default=["locomo", "longmemeval"], choices=["locomo", "longmemeval"])
    p.add_argument("--methods", nargs="+", default=["bm25", "tfidf"], choices=["bm25", "tfidf"])
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    args = p.parse_args()
    if not 1 <= args.repeats <= 10:
        p.error("repeats must be 1..10")
    asyncio.run(suite(args))


if __name__ == "__main__":
    main()
