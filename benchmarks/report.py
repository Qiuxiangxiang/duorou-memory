"""Validate completed runs and render an evidence-retrieval report.

This command never queries a model or runs a load generator. All results are
recomputed from predictions and checked against the completed suite metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


METHODS = ("bm25_raw", "fts_raw", "bm25_enriched", "fts_enriched", "fts_enriched_neighbors")
LABELS = {
    "bm25_raw": "BM25／纯正文控制",
    "fts_raw": "duorou 当前 FTS／纯正文",
    "bm25_enriched": "BM25／日期＋speaker＋正文＋caption",
    "fts_enriched": "实验 FTS／相同增强字段",
    "fts_enriched_neighbors": "实验 FTS／增强字段＋会话内邻居",
}
COMPARISONS = (
    ("fts_raw", "bm25_raw", "同内容的检索实现比较；分词与统计范围不同"),
    ("fts_enriched", "fts_raw", "字段投影及索引存储实现同时改变，非纯字段消融"),
    ("bm25_enriched", "bm25_raw", "相同 BM25 引擎的联合字段消融"),
    ("fts_enriched", "bm25_enriched", "相同增强内容的检索实现比较"),
    ("fts_enriched_neighbors", "fts_enriched", "同源会话内固定邻居扩展；最多 20 个叶子 ID"),
)
KS = (1, 5, 10, 20)
METRICS = ("recall", "any_hit", "all_hit", "mrr")
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique(rows: list[dict], key: str, description: str) -> dict:
    require(all(isinstance(row.get(key), str) and row[key] for row in rows), f"invalid {description} ID")
    output = {row[key]: row for row in rows}
    require(len(output) == len(rows), f"duplicate {description} IDs")
    return output


def score(gold: list[str], ranked: list[str], k: int) -> dict:
    expected, returned = set(gold), ranked[:k]
    found = expected.intersection(returned)
    first = next((i for i, item in enumerate(returned, 1) if item in expected), None)
    return {"recall": len(found) / len(expected), "any_hit": float(bool(found)),
            "all_hit": float(found == expected), "mrr": 1 / first if first else 0.0}


def aggregate(rows: list[dict]) -> dict:
    require(bool(rows), "cannot aggregate an empty group")
    return {"queries": len(rows), "errors": sum(row["status"] != "success" for row in rows),
            "at_k": {str(k): {metric: statistics.mean(row["at_k"][str(k)][metric] for row in rows)
                              for metric in METRICS} for k in KS}}


def close(actual: float, expected: float, context: str) -> None:
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f"metric mismatch: {context}")


def check_aggregate(actual: dict, expected: dict, context: str) -> None:
    require(actual["queries"] == expected["queries"] and actual["errors"] == expected["errors"],
            f"aggregate counts mismatch: {context}")
    for k in KS:
        for metric in METRICS:
            close(actual["at_k"][str(k)][metric], expected["at_k"][str(k)][metric], f"{context}/{k}/{metric}")


def distribution(values: list[int | float]) -> dict:
    require(bool(values), "empty distribution")
    ordered = sorted(values)
    def percentile(q: float) -> float:
        point = (len(ordered) - 1) * q
        lower = int(point)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (point - lower)
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": percentile(.95), "min": min(values), "max": max(values)}


def paired(left: dict, right: dict) -> dict:
    require(left.keys() == right.keys(), "paired query IDs differ")
    deltas = [left[q]["at_k"]["20"]["recall"] - right[q]["at_k"]["20"]["recall"] for q in left]
    gains = [q for q in left if left[q]["at_k"]["20"]["all_hit"] > right[q]["at_k"]["20"]["all_hit"]]
    losses = [q for q in left if left[q]["at_k"]["20"]["all_hit"] < right[q]["at_k"]["20"]["all_hit"]]
    return {"queries": len(left), "recall20_delta": statistics.mean(deltas),
            "recall20_delta_percentage_points": 100 * statistics.mean(deltas),
            "recall20_wins": sum(d > 1e-12 for d in deltas),
            "recall20_ties": sum(abs(d) <= 1e-12 for d in deltas),
            "recall20_losses": sum(d < -1e-12 for d in deltas),
            "all_hit20_gains": len(gains), "all_hit20_losses": len(losses),
            "all_hit20_delta_percentage_points": 100 * (len(gains) - len(losses)) / len(left),
            "all_hit20_gained_query_ids": gains, "all_hit20_lost_query_ids": losses}


def performance_summary(suite: dict) -> list[dict]:
    config = suite["protocol"]["performance"]
    expected = {(method, c, r) for method in METHODS for c in config["concurrencies"]
                for r in range(1, config["repeats"] + 1)}
    rows = suite["performance"]
    keys = [(row["method"], row["concurrency"], row["repeat"]) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "performance phases missing or duplicated")
    groups = defaultdict(list)
    for row in rows:
        summary = row["summary"]
        require(summary["planned"] == config["requests_per_phase"] and
                summary["offered"] == summary["planned"] and summary.get("termination") is None,
                "performance phase incomplete")
        require(sum(summary["counts"].values()) == summary["offered"], "performance status counts mismatch")
        groups[(row["method"], row["concurrency"])].append(summary)
    output = []
    for (method, concurrency), group in sorted(groups.items()):
        p95s = [s["sojourn_ms"]["success_only"]["p95"] for s in group]
        p99s = [s["sojourn_ms"]["success_only"]["p99"] for s in group]
        require(all(x is not None for x in p95s + p99s), "latency quantiles missing")
        output.append({"method": method, "concurrency": concurrency, "repeats": len(group),
            "offered": sum(s["offered"] for s in group),
            "qps_median": statistics.median(s["throughput_rps"] for s in group),
            "goodput_median": statistics.median(s["goodput_rps"] for s in group),
            "good_fraction_median": statistics.median(s["good_fraction_offered"] for s in group),
            "p95_ms_median_of_runs": statistics.median(p95s),
            "p99_ms_median_of_runs": statistics.median(p99s), "p99_ms_range": [min(p99s), max(p99s)],
            "errors": sum(s["counts"]["error"] for s in group),
            "timeouts": sum(s["counts"]["timeout"] for s in group),
            "drops": sum(s["counts"]["dropped"] for s in group)})
    return output


def analyze(run_dir: Path, historical: Path | None = None) -> dict:
    suite = read_json(run_dir / "suite.json")
    require(bool(suite.get("finished_utc")), "suite has not completed")
    require(suite["source_hashes_before"] == suite["source_hashes_after"], "source code changed during run")
    require(suite["db_hash_after"] == suite["dataset"]["db_sha256"], "source database changed during run")
    dataset = suite["dataset"]
    files = dataset["files"]
    for name, entry in files.items():
        path = run_dir / entry["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"input hash mismatch: {name}")
    raw = unique(read_jsonl(run_dir / files["raw_documents"]["path"]), "id", "raw document")
    enriched = unique(read_jsonl(run_dir / files["enriched_documents"]["path"]), "id", "enriched document")
    queries = unique(read_jsonl(run_dir / files["queries"]["path"]), "id", "query")
    metadata = read_json(run_dir / files["metadata"]["path"])
    require(raw.keys() == enriched.keys() and queries.keys() == metadata.keys(), "projection or metadata IDs differ")
    require(len(raw) == dataset["summary"]["documents"] and len(queries) == dataset["summary"]["eligible_questions"], "dataset counts mismatch")
    for doc_id, row in raw.items():
        require(set(row) == {"id", "scope", "text"} and set(enriched[doc_id]) == set(row), "unexpected index fields")
        require(row["scope"] == enriched[doc_id]["scope"], "document scope differs across projections")
    for q in queries.values():
        gold = q["expected_ids"]
        require(0 < len(gold) <= 20 and len(gold) == len(set(gold)), "invalid gold list")
        require(all(item in raw and raw[item]["scope"] == q["scope"] for item in gold), "gold outside scope")
    summaries = unique(suite["quality"], "method", "quality method")
    require(set(summaries) == set(METHODS), "missing or unexpected quality method")
    lengths = {projection: {key: (len(doc["text"]), len(WORD_RE.findall(doc["text"])))
                            for key, doc in documents.items()}
               for projection, documents in (("raw", raw), ("enriched", enriched))}
    results, predictions, contexts = {}, {}, {}
    for method in METHODS:
        rows = read_jsonl(run_dir / "quality" / f"{method}.predictions.jsonl")
        lookup = unique(rows, "query_id", method + " prediction")
        require(lookup.keys() == queries.keys(), f"not every eligible query exactly once: {method}")
        groups, scopes = defaultdict(list), defaultdict(list)
        for qid, row in lookup.items():
            q = queries[qid]
            require(row["scope"] == q["scope"] and row["gold_ids"] == q["expected_ids"], f"query/gold mismatch: {method}/{qid}")
            category = metadata[qid].get("category_name", metadata[qid]["category"])
            require(str(row["category"]) == str(category), f"category mismatch: {method}/{qid}")
            ids = row["ranked_ids"]
            require(isinstance(ids, list) and all(isinstance(item, str) for item in ids), "invalid ranking type")
            require(len(ids) <= 20 and len(ids) == len(set(ids)), "duplicate or more than 20 returned IDs")
            require(all(item in raw and raw[item]["scope"] == q["scope"] for item in ids), "returned ID outside scope")
            require(row["status"] in ("success", "error"), "unexpected quality status")
            require(row["status"] == "success" or not ids, "failed query has nonempty ranking")
            for k in KS:
                recomputed = score(q["expected_ids"], ids, k)
                for metric in METRICS:
                    close(row["at_k"][str(k)][metric], recomputed[metric], f"{method}/{qid}/{k}/{metric}")
            groups[str(category)].append(row)
            scopes[row["scope"]].append(row)
        result = {"overall": aggregate(rows), "categories": {k: aggregate(v) for k, v in sorted(groups.items())},
                  "scopes": {k: aggregate(v) for k, v in sorted(scopes.items())}}
        check_aggregate(summaries[method]["overall"], result["overall"], method)
        for group in ("categories", "scopes"):
            require(set(summaries[method][group]) == set(result[group]), f"{group} keys mismatch")
            for key in result[group]:
                check_aggregate(summaries[method][group][key], result[group][key], f"{method}/{group}/{key}")
        results[method], predictions[method] = result, lookup
        contexts[method] = {"returned_ids": distribution([len(row["ranked_ids"]) for row in rows])}
        for projection in ("raw", "enriched"):
            contexts[method][projection] = {
                "characters": distribution([sum(lengths[projection][i][0] for i in row["ranked_ids"]) for row in rows]),
                "regex_words": distribution([sum(lengths[projection][i][1] for i in row["ranked_ids"]) for row in rows])}
    historical_result = {"requested": historical is not None}
    if historical is not None:
        old = unique(read_jsonl(historical), "query_id", "historical prediction")
        require(old.keys() == queries.keys(), "historical query set differs")
        mismatches = [qid for qid in queries if old[qid]["ranked_ids"] != predictions["bm25_enriched"][qid]["ranked_ids"]]
        require(not mismatches, f"historical enriched BM25 rankings differ on {len(mismatches)} queries")
        require(all(old[q]["gold_ids"] == queries[q]["expected_ids"] and old[q]["scope"] == queries[q]["scope"]
                    for q in queries), "historical gold/scope differs")
        historical_result.update({"file_name": historical.name, "sha256": hashlib.sha256(historical.read_bytes()).hexdigest(),
            "queries": len(queries), "exact_ranked_id_matches": len(queries), "all_rankings_identical": True})
    comparisons = []
    for left, right, meaning in COMPARISONS:
        item = paired(predictions[left], predictions[right])
        item.update(left=left, right=right, interpretation=meaning)
        comparisons.append(item)
    return {"release": suite["protocol"]["release"], "started_utc": suite["started_utc"], "finished_utc": suite["finished_utc"],
        "protocol": suite["protocol"], "dataset": dataset, "environment": suite["environment"],
        "validation": {"passed": True, "all_methods_once_per_query": True, "scope_valid": True,
            "unique_source_ids_max20": True, "source_hashes_unchanged": True, "source_database_unchanged": True,
            "predictions_and_summary_metrics_agree": True, "input_file_hashes_verified": True},
        "quality": results, "paired_comparisons": comparisons, "context_diagnostics": contexts,
        "context_definition": "Sum Unicode codepoints or regex \\b\\w+\\b matches in each actually returned source projection; excludes separators and IDs. These are NOT model tokens and were NOT sent on the ID-only MCP wire.",
        "performance": performance_summary(suite), "index_reports": suite["index_reports"],
        "historical_replication": historical_result,
        "source_hashes": suite["source_hashes_before"],
        "evidence_files": {method: f"quality/{method}.predictions.jsonl" for method in METHODS}}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def metric_row(name: str, group: dict) -> str:
    at = group["at_k"]
    return f"| {name} | {group['queries']} | {pct(at['5']['recall'])} | {pct(at['10']['recall'])} | {pct(at['20']['recall'])} | {pct(at['20']['all_hit'])} | {at['20']['mrr']:.4f} | {group['errors']} |"


def render(a: dict) -> str:
    d, p = a["dataset"], a["protocol"]
    summary, perf = d["summary"], a["performance"]
    lines = ["# duorou：公开 LoCoMo 数据实际检索测试", "",
        f"版本：{a['release']}；完成时间：{a['finished_utc']}（UTC）。", "",
        f"本次实际执行 {len(METHODS)} 个方法 × {summary['eligible_questions']:,} 个合格问题，共 {len(METHODS)*summary['eligible_questions']:,} 次全量质量查询；另有 {len(perf)} 组方法/并发组合、{sum(x['repeats'] for x in perf)} 个负载阶段、{sum(x['offered'] for x in perf):,} 个正式负载请求。所有请求均跨越本机 MCP Streamable HTTP 接口。", "",
        "这是证据检索评测。没有调用提取模型、答案生成模型或裁判模型，模型调用数为 0；这些分数不是官方端到端问答准确率。新 MCP 测试适配器与增强字段、邻居扩展是本次新增工作，不能归入旧版本已经具备的能力。", "",
        "## 1. 数据、计分与实验边界", "",
        f"固定数据：[LoCoMo 原始 JSON]({d['url']})；revision `{d['revision']}`；SHA256 `{d['raw_sha256']}`；许可 `{d['license']}`。", "",
        f"共 {summary['conversations']} 段对话、{summary['sessions']} 个源会话、{summary['documents']:,} 个回合。原始 {summary['source_questions']:,} 个问题，纳入 {summary['eligible_questions']:,} 个，排除 {summary['excluded_questions']} 个：" + "；".join(f"`{k}` {v}" for k, v in summary['exclusion_counts'].items()) + "。排除规则沿用旧评测，未修补或部分丢弃无效 gold。", "",
        "每题请求最多 20 个同 scope、无重复的回合 ID；k=5/10 指标取该排序前缀。Recall@k 是每题 gold 证据召回比例后宏平均；AllHit@20 是全部 gold 均召回的问题比例；MRR@20 是首个 gold 排名倒数的宏平均。失败计为 0，保留在分母。这里只纳入可评分的非空 gold 问题，因此没有测拒答能力。", "",
        "最大 20 个回合不等于固定上下文 token 预算。下文同时统计真实返回 ID 对应文本的字符数和正则词数；线上响应仅有 ID 和分数。已查看过这些公开数据，因此这是开发评测，不是未见数据盲测。", "",
        "| 方法 | 含义 |", "|---|---|"]
    lines.extend(f"| `{m}` | {LABELS[m]} |" for m in METHODS)
    lines += ["", "增强字段两组输入逐文档相同，包括原数据提供的日期、speaker、正文与非空 caption；没有生成新的 caption。纯正文两组只索引 `source.content`。", "",
        "SQLite FTS 使用 unicode61、OR 查询、全语料统计；BM25 控制使用 sklearn 英语停用词及每 scope 独立统计。两引擎差异不能归因于 schema 或认知机制。原始 FTS 调用旧 `search_sources()`，含 snippet 计算；增强 FTS 使用内存派生索引。因此增强 FTS 对原始 FTS 的速度变化还包含存储及查询实现变化。", "",
        "邻居实验预先固定：保留词法排序前 10；按前 5 个 seed 顺序追加同一 source_session 中的前驱、后继；去重后从词法候选补足，最后截取 20。没有使用 gold 或 QA 类别选邻居，也没有把命中会话直接计为命中其全部子证据。", "",
        "## 2. 全量质量结果", "",
        "| 方法 | 问题数 | Recall@5 | Recall@10 | Recall@20 | AllHit@20 | MRR@20 | 错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.extend(metric_row(f"`{m}`", a["quality"][m]["overall"]) for m in METHODS)
    lines += ["", "## 3. 同题配对比较", "",
        "差值为左侧方法减右侧方法；pp 为百分点。胜/平/负按每题 Recall@20 比较，AllHit 新增/丢失按是否找到全部 gold 比较。", "",
        "| 左侧 − 右侧 | Δ Recall@20（pp） | R20 胜 / 平 / 负 | AllHit 新增 / 丢失 | Δ AllHit（pp） |",
        "|---|---:|---:|---:|---:|"]
    for x in a["paired_comparisons"]:
        lines.append(f"| `{x['left']}` − `{x['right']}` | {x['recall20_delta_percentage_points']:+.2f} | {x['recall20_wins']} / {x['recall20_ties']} / {x['recall20_losses']} | {x['all_hit20_gains']} / {x['all_hit20_losses']} | {x['all_hit20_delta_percentage_points']:+.2f} |")
    lines += ["", "`bm25_enriched − bm25_raw` 是相同检索实现的联合字段消融，但未分别拆开日期、speaker 和 caption；不能把全部增益归给其中一个字段。`fts_enriched − fts_raw` 同时改变投影与索引存储实现。邻居策略只代表这条固定启发式的观测结果，不证明最优，也不证明更复杂的语义事件机制有效。", "",
        "## 4. 按问题类型", "",
        "| 类别／方法 | 问题数 | Recall@5 | Recall@10 | Recall@20 | AllHit@20 | MRR@20 | 错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for category in a["quality"][METHODS[0]]["categories"]:
        lines.extend(metric_row(f"{category} / `{m}`", a["quality"][m]["categories"][category]) for m in METHODS)
    lines += ["", "## 5. 按对话分组", "",
        "每个单元格为 Recall@20 / AllHit@20。问题共享 10 段对话，不能将所有问题视作来自同等数量的独立用户。", "",
        "| 对话 | 问题数 | BM25 raw | FTS raw | BM25 enriched | FTS enriched | FTS neighbors |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for scope, first in a["quality"][METHODS[0]]["scopes"].items():
        cells = []
        for m in METHODS:
            metric = a["quality"][m]["scopes"][scope]["at_k"]["20"]
            cells.append(f"{pct(metric['recall'])} / {pct(metric['all_hit'])}")
        lines.append(f"| {scope} | {first['queries']} | " + " | ".join(cells) + " |")
    lines += ["", "## 6. MCP 负载结果", "",
        f"每阶段 {p['performance']['requests_per_phase']:,} 个正式请求、{p['performance']['warmup']} 个预热请求，每组合 {p['performance']['repeats']} 次重复；闭环负载，全部目标及负载阶段顺序运行。表格取各阶段统计量的中位数，未合并或平均分位数。", "",
        "Goodput 定义为每秒同时满足“全部 gold 命中，且 sojourn ≤200ms”的请求数；负载 query 有放回采样，不能替代上面的全量质量分数。P95/P99 仅针对成功请求，错误、超时和丢弃另外列出。", "",
        "| 方法 | 并发 | QPS 中位数 | Goodput 中位数 | 成功 P95 ms 中位数 | 成功 P99 ms 中位数 | P99 范围 ms | 错误 / 超时 / 丢弃 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in perf:
        lines.append(f"| `{x['method']}` | {x['concurrency']} | {x['qps_median']:.2f} | {x['goodput_median']:.2f} | {x['p95_ms_median_of_runs']:.2f} | {x['p99_ms_median_of_runs']:.2f} | {x['p99_ms_range'][0]:.2f}–{x['p99_ms_range'][1]:.2f} | {x['errors']} / {x['timeouts']} / {x['drops']} |")
    env = a["environment"]
    lines += ["", f"运行环境：Python `{env['python'].split()[0]}`，SQLite `{env['sqlite']}`，平台 `{env['platform']}`；报告的宿主逻辑 CPU 数 {env['cpu_count']}，宿主内存 {env['memory_bytes']/1024**3:.2f} GiB。容器限制原值：`{json.dumps(env.get('cgroup', {}), ensure_ascii=False)}`。宿主硬件读数不等于可独占资源。", "",
        "这些是同一次运行环境下的本机 IPC/HTTP 与索引组合结果；没有网络跨机、饱和点、负载发生器能力校准或生产认证。不能据此声称性能已经压榨完毕或生产容量已确定。", "",
        "## 7. 实际返回证据的文本规模", "",
        "以下按真实返回的 ID 回查两种正文投影，对 Unicode 字符数及 `\\b\\w+\\b` 正则词数求和；不包含 ID、包装符和分隔符。这些是诊断量，**不是模型 tokens，也不是 MCP 返回的全文负载**。每种方法均同时报告 raw 与 enriched，避免字段差异被掩盖。", "",
        "| 方法 | 回查投影 | 字符数均值 | 字符数中位数 | 字符数 P95 | 正则词数均值 | 正则词数中位数 | 正则词数 P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        for projection in ("raw", "enriched"):
            x = a["context_diagnostics"][m][projection]
            c, w = x["characters"], x["regex_words"]
            lines.append(f"| `{m}` | {projection} | {c['mean']:.1f} | {c['median']:.1f} | {c['p95']:.1f} | {w['mean']:.1f} | {w['median']:.1f} | {w['p95']:.1f} |")
    historical = a["historical_replication"]
    lines += ["", "## 8. 完整性、历史复现与未测能力", "",
        f"输入文件哈希、代码运行前后哈希、数据库运行前后哈希、每题每方法恰好一次、同 scope、无重复且最多 20 个结果，以及预测与汇总指标一致性检查均已通过。数据库 integrity 为 `{d['sqlite_integrity']}`，外键问题 {d['foreign_key_violations']}，FTS integrity `{d['fts_integrity']}`；无权限 principal 对 {d['denied_principal_scope_checks']} 个 scope 的读取均被拒绝。", ""]
    if historical["requested"]:
        lines.append(f"历史增强 BM25 逐题复现：{historical['exact_ranked_id_matches']}/{historical['queries']} 个问题的完整返回 ID 排序与旧预测完全相同。旧预测文件 SHA256 `{historical['sha256']}`。这只证明该基线在相同数据内容下可复现，不能把历史负载数字当成此次性能结果。")
    else:
        lines.append("此次没有传入历史预测文件，未执行逐题历史排序复现检查。")
    lines += ["", "| 数据表 | 实际行数 |", "|---|---:|"]
    lines.extend(f"| `{table}` | {count:,} |" for table, count in d["table_counts"].items())
    lines += ["", "claims、entity_mentions、entity_aliases、intentions、working_sets 的实际行数见表。没有在这些表上形成可评价的语义记忆内容，因而本次不能验证自动事实提取、身份消歧、冲突修订、工作记忆、长期遗忘或任务规划。source_session 来自公开数据本身的会话边界，speaker 是数据源元信息绑定。", "",
        "FTS 全局语料统计由所有 scope 共享；返回行的权限隔离不等于统计或时间侧信道隔离。服务 principal 由本地部署绑定，没有面向远程用户的登录认证。增强索引为本次静态测试构建，实时增删后的派生索引同步未测试。当前英语数据与 unicode61 结果也不能证明中文检索效果。", "",
        "本轮没有运行其他记忆产品，因此不构成与 Mem0、Zep 或其他系统的统一条件排名。下一轮需要固定证据文本预算及同一 reader/judge 后，才能评估完整问答；若要验证语义提取，应先实现提取、证据回链和版本更新，再对这些新增模块另行设置数据与指标。", "",
        "## 9. 复核材料", "",
        "原始预测与 gold 记录在 `quality/*.predictions.jsonl`；完整指标、配对变化问题 ID、文本规模分布在 [analysis.json](analysis.json)；执行配置与环境在 [suite.json](suite.json) 和 [protocol.json](protocol.json)；输入哈希和排除清单指针在 [dataset.json](dataset.json)。", "",
        "所有结果均由本次执行文件重新计算，报告生成器不会执行模型调用或负载测试。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--historical", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir.resolve(), args.historical)
    (args.run_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.run_dir / "REPORT.zh.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"report": str(args.run_dir / "REPORT.zh.md"), "validation": result["validation"],
                      "historical": result["historical_replication"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
