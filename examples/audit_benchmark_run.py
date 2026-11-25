#!/usr/bin/env python3
"""Recount recorded load samples and expose quality/latency SLO failures."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    root = p.parse_args().run_dir
    suite = json.loads((root / "suite.json").read_text())
    phases, statuses, slow_by_method = [], Counter(), Counter()
    for phase in suite["performance"]:
        rows = [json.loads(line) for line in
                (root / "performance" / phase["run_id"] / "samples.jsonl").read_text().splitlines()]
        summary = phase["summary"]
        counts = Counter(row["status"] for row in rows)
        if len(rows) != summary["offered"] or any(counts[k] != v for k, v in summary["counts"].items()):
            raise ValueError("raw sample count mismatch")
        limit = summary["slo"]["threshold_ms"]
        success = [row for row in rows if row["status"] == "success"]
        good = sum(row["quality_ok"] and row["sojourn_ms"] <= limit for row in success)
        slow = sum(row["sojourn_ms"] > limit for row in success)
        if good != summary["good_count"]:
            raise ValueError("goodput sample count mismatch")
        if any(not 0 <= row["result_count"] <= 20 for row in rows):
            raise ValueError("result count exceeds budget")
        statuses.update(counts)
        slow_by_method[phase["method"]] += slow
        phases.append({"method": phase["method"], "concurrency": phase["concurrency"],
                       "repeat": phase["repeat"], "run_id": phase["run_id"],
                       "formal_samples": len(rows), "good_count": good,
                       "successful_over_200ms": slow, "slo": summary["slo"]})
    result = {"formal_samples": sum(statuses.values()), "status_counts": dict(statuses),
              "phases": len(phases), "slo_pass_phases": sum(x["slo"]["observed_pass"] for x in phases),
              "quality_gate_failed_phases": sum(not x["slo"]["quality_gate_ok"] for x in phases),
              "latency_gate_failed_phases": sum(not x["slo"]["latency_gate_ok"] for x in phases),
              "successful_over_200ms": sum(slow_by_method.values()),
              "successful_over_200ms_by_method": dict(slow_by_method),
              "quality_gate_definition": "Every successful request must have quality_ok=true; min_recall=1 requires all annotated gold, and no forbidden IDs.",
              "latency_gate_definition": "Successful response p99 sojourn <=200ms; not every single request <=200ms.",
              "audit_limit": "Load samples retain counts/quality flags, not full returned rankings. Full scope/ranking audit belongs to saved exhaustive quality predictions.",
              "phase_details": phases}
    (root / "postrun_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    text = f"""# 逐请求复核与 SLO 结果

正式负载共 **{result['formal_samples']:,}** 条、**{result['phases']}** 个阶段。
原始样本的状态计数为 `{json.dumps(result['status_counts'])}`；逐阶段样本数与 Goodput 计数均与报告相符。

**请求成功不等于通过 SLO。** 本轮有 **{result['quality_gate_failed_phases']}/{result['phases']}** 个阶段未通过配置的质量门槛：原引擎要求每个成功请求的 `quality_ok=true`，而配置的 `min_recall=1` 要求找齐该题全部标注证据。

另有 **{result['latency_gate_failed_phases']}** 个阶段未通过成功请求 P99≤200ms 门槛。全部正式请求中，成功但超过 200ms 的有 **{result['successful_over_200ms']}** 条；各方法计数如下。

| 方法 | 成功但超过 200ms 的请求 |
|---|---:|
"""
    text += "\n".join(f"| `{k}` | {v} |" for k, v in sorted(slow_by_method.items()))
    text += "\n\n质量门槛采用本次严格配置，未通过不代表所有返回都无用；各方法的全量证据覆盖和有效吞吐分别见主报告。时延门槛按 P99 判断，因此少量请求超过 200ms 也可能通过该分位数门槛。\n\n负载日志没有保存完整返回 ID 列表，不能据此重新验证全部负载响应的逐条作用域与排序。保存了完整排名的 7,635 次质量查询已由报告生成器进行这项验证。\n\n详细逐阶段记录见 [postrun_audit.json](postrun_audit.json)。\n"
    (root / "POSTRUN_AUDIT.zh.md").write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k != "phase_details"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
