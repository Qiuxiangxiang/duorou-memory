"""ATM-Bench P0 experiments and full paired evaluation."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile

from cloud import JUDGE, QUERIES, REWRITE, TERRA, run_batches

BUDGET = 32768
P0 = ("baseline", "subquery", "coverage", "local30", "terra30")
ANSWER = (
    "Answer each question independently using its own evidence. Return one concise answer per ID. "
    "For numbers include the value and requested unit. For lists of source records return exact IDs "
    "comma-separated. For a list of named values return the values comma-separated. Do not invent "
    "missing records. If no answer is supported, write Insufficient evidence.")
RERANK = (
    "Rank source IDs for each question independently. Prefer evidence that would directly support "
    "the answer. For questions needing multiple facts, retain complementary evidence. Return up to "
    "30 distinct supplied IDs in relevance order, using only IDs listed for that question.")
STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING}


def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def usage(batches):
    result = defaultdict(int)
    for batch in batches:
        for field, value in batch.items():
            if isinstance(value, int):
                result[field] += value
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return dict(result)


def batch_usage(responses):
    return [batch["usage"] for batch in responses]


def make_inputs(memory, questions, rewrites):
    from duorou.retrieval import rrf
    result = {}
    for q in questions:
        phrases = [q["question"], *(s.strip() for s in rewrites[q["id"]][:2])]
        phrases = list(dict.fromkeys(s for s in phrases if s and len(s) <= 4096))
        pools = [
            [hit["id"] for hit in memory.search("atm", phrase, top_k=50, budget=2_000_000)]
            for phrase in phrases
        ]
        scores = rrf(pools)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))
        result[q["id"]] = (phrases, pools, ordered)
    return result


def ordered_by_model(encoder, question, ids, views):
    scores = encoder.score(question, [views[key] for key in ids])
    return [key for _, key in sorted(zip(scores, ids), key=lambda row: (-row[0], row[1]))]


def fuse(lists):
    from duorou.retrieval import rrf
    scores = rrf(lists)
    return sorted(scores, key=lambda key: (-scores[key], key))


def select(order, records):
    selected, used = [], 0
    for key in dict.fromkeys(order):
        size = len(records[key]["text"].encode("utf-8"))
        if used + size <= BUDGET:
            selected.append(key)
            used += size
        if len(selected) == 10:
            break
    return selected


def make_rankings(cache, split, experiments, questions, inputs, encoder, records, views):
    target = cache.parent / f"rankings-{split}-{'-'.join(experiments)}.json"
    if target.exists():
        saved = json.loads(target.read_text())
        return saved["rankings"], saved["terra_rerank_usage"]
    rankings = {arm: {} for arm in experiments}
    usage_rows = []
    remote = {}
    if "terra30" in experiments:
        rows = [{"id": q["id"], "question": q["question"], "candidates": [
                 {"id": key, "text": views[key][:920]} for key in inputs[q["id"]][2][:30]]}
                for q in questions]
        remote, batch_reports = run_batches(cache, f"rerank-{split}", TERRA, RERANK, rows,
                                            {"ranked": STRINGS}, 6, 160000)
        usage_rows = batch_usage(batch_reports)
    cached_base = {}
    if split == "Main":
        journal = cache.parent / "main-original-order.jsonl"
        if journal.exists():
            for line in journal.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cached_base[item["id"]] = item

    def rank_one(q):
        qid = q["id"]
        phrases, pools, all_ids = inputs[qid]
        orders = {}
        if set(experiments) & {"baseline", "subquery", "coverage"}:
            digest = hashlib.sha256(json.dumps(all_ids).encode()).hexdigest()
            cached = cached_base.get(qid)
            if cached and cached["candidate_sha256"] == digest:
                order = cached["ranked"]
            else:
                order = ordered_by_model(encoder, q["question"], all_ids, views)
            orders["baseline"] = fuse([all_ids, order])
            if set(experiments) & {"subquery", "coverage"}:
                branches = [ordered_by_model(encoder, phrase, items, views)
                            for phrase, items in zip(phrases[1:], pools[1:], strict=True)]
                orders["subquery"] = fuse([all_ids, order, *branches])
                initial = orders["subquery"][:2]
                for ranking in branches:
                    first_extra = next((key for key in ranking if key not in initial), None)
                    if first_extra:
                        initial.append(first_extra)
                orders["coverage"] = initial + orders["subquery"]
        shortlist = all_ids[:30]
        if "local30" in experiments:
            local = ordered_by_model(encoder, q["question"], shortlist, views)
            orders["local30"] = fuse([shortlist, local]) + all_ids[30:]
        if "terra30" in experiments:
            validated = [key for key in dict.fromkeys(remote[qid]["ranked"]) if key in shortlist]
            orders["terra30"] = fuse([shortlist, validated]) + all_ids[30:]
        source_rank = None
        if split == "Main" and "baseline" in orders:
            source_rank = dict(id=qid, candidate_sha256=digest, ranked=order)
        return qid, {arm: select(orders[arm], records) for arm in experiments}, source_rank

    journal = target.with_suffix(".jsonl")
    complete = {}
    if journal.exists():
        with journal.open() as stream:
            for line in stream:
                try:
                    part = json.loads(line)
                except json.JSONDecodeError:
                    continue
                complete[part["id"]] = part["rankings"]
    pending = [q for q in questions if q["id"] not in complete]
    with journal.open("a") as stream, ThreadPoolExecutor(max_workers=3 if split=="Main" else 1) as workers:
        for index, (qid, result, source_rank) in enumerate(workers.map(rank_one, pending), 1):
            complete[qid] = result
            if source_rank and qid not in cached_base:
                with (cache.parent/"main-original-order.jsonl").open("a") as source_stream:
                    source_stream.write(json.dumps(source_rank)+"\n")
            stream.write(json.dumps(dict(id=qid, rankings=result))+"\n")
            stream.flush()
            if index % 25 == 0 or index == len(pending):
                print("local ranking", split, f"{len(complete)}/{len(questions)}", flush=True)
    assert len(complete) == len(questions)
    for q in questions:
        qid = q["id"]
        for arm in experiments:
            rankings[arm][qid] = complete[qid][arm]
    target.write_text(json.dumps({"rankings": rankings, "terra_rerank_usage": usage_rows}))
    return rankings, usage_rows


def answer_and_score(cache, split, arm, qs, records, ranks, qa, retrieval, rewrite_use, rerank_use, output,
                     evidence_view=None):
    if (output / f"{split}-{arm}.json").exists():
        return json.loads((output / f"{split}-{arm}.json").read_text())
    rows = [
        {"id": q["id"], "question": q["question"], "format": q["qtype"], "evidence": [
         {"id": key, "text": evidence_view(q, records[key]["text"]) if evidence_view else records[key]["text"]}
         for key in ranks[q["id"]]]} for q in qs
    ]
    answers, answer_batches = run_batches(cache, f"answer-{split}-{arm}", TERRA, ANSWER, rows,
                                          {"answer": STRING}, 10, 230000)
    evaluations, judge_inputs = [], []
    for q in qs:
        key, kind = q["id"], q["qtype"]
        prediction = answers[key]["answer"]
        row = dict(id=key, qtype=kind, prediction=prediction, ranked=ranks[key])
        if kind == "number":
            row["accuracy"] = qa.deterministic_accuracy(str(q["answer"]), prediction, q["question"])
        elif kind == "list_recall":
            row["accuracy"] = qa.list_jaccard_score(str(q["answer"]), prediction)
        else:
            judge_inputs.append(dict(id=key, instructions=qa.build_judge_prompt(
                q["question"], str(q["answer"]), prediction)))
        evaluations.append(row)
    judge, judge_batches = run_batches(
        cache, f"judge-{split}-{arm}", JUDGE,
        "Apply each item's supplied grading instructions independently. Return one boolean accuracy per id.",
        judge_inputs, {"accuracy": {"type": "boolean"}}, 24, 87000)
    for row in evaluations:
        if row["id"] in judge:
            row["accuracy"] = judge[row["id"]]["accuracy"]
    metrics = qa.summarize_atm(evaluations)
    recall = retrieval.compute_metrics([dict(gt_evidence_ids=q["evidence_ids"], retrieval_ids=ranks[q["id"]])
                                        for q in qs], [10])["metrics"]["recall@10"]
    stages = {"rewrite": usage(rewrite_use), "llm_rerank": usage(rerank_use),
              "answer": usage(batch_usage(answer_batches)), "judge": usage(batch_usage(judge_batches))}
    report = dict(split=split, arm=arm, n=len(qs), qs=metrics["accuracy"], recall_at_10=recall,
                  total_tokens=sum(stages[s]["total_tokens"] for s in ("rewrite", "llm_rerank", "answer")),
                  by_qtype=metrics["by_qtype"], stages=stages, judge=JUDGE, answer_model=TERRA)
    (output / f"{split}-{arm}.json").write_text(json.dumps(report, indent=2)+"\n")
    with (output / f"{split}-{arm}.jsonl").open("w") as handle:
        for row in evaluations:
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(split, arm, json.dumps(report), flush=True)
    return report


def write_table(output, rows):
    with (output/"metrics.csv").open("w", newline="") as handle:
        fields = ("split", "arm", "n", "qs", "recall_at_10", "total_tokens")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    base = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(base), str(args.release.resolve()), str(args.official.resolve())]
    from run import load_inputs
    from duorou import Memory
    from memqa.utils.evaluator import evaluate_qa as qa
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval as retrieval

    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    scorer = read_module("local_rerank", base/"rerank"/"run.py")
    encoder = scorer.CrossEncoder(args.model)
    records, _, splits, _ = load_inputs(args.data)
    views = {key: scorer.ranking_view(row["text"]) for key, row in records.items()}
    report_rows = []

    def run_split(memory, split, arms, rewrite_rows, rewrite_use):
        qs = splits[split]
        rewrites = {key: row["queries"] for key, row in rewrite_rows.items()}
        inputs = make_inputs(memory, qs, rewrites)
        ranks, rerank_use = make_rankings(args.cache, split, arms, qs, inputs, encoder, records, views)
        for arm in arms:
            report_rows.append(answer_and_score(args.cache, split, arm, qs, records, ranks[arm], qa,
                                                retrieval, rewrite_use, rerank_use if arm=="terra30" else [],
                                                args.output))
            if not args.main_baseline_only:
                write_table(args.output, report_rows)

    with tempfile.TemporaryDirectory() as scratch:
        engine = Memory(Path(scratch)/"bench.sqlite", "bench")
        try:
            engine.provision("atm")
            engine.ingest_many("atm", records.values())
            if args.main_baseline_only:
                main_qs = [{"id": q["id"], "question": q["question"]} for q in splits["Main"]]
                rewrites, used = run_batches(args.cache, "rewrite-main", TERRA, REWRITE, main_qs,
                                             QUERIES, 48, 32000)
                run_split(engine, "Main", ("baseline",), rewrites, batch_usage(used))
                return
            hard_qs = [{"id": q["id"], "question": q["question"]} for q in splits["Hard"]]
            hard_rewrites, hard_usage = run_batches(args.cache, "rewrite-Hard", TERRA, REWRITE, hard_qs,
                                                    QUERIES, 48, 32000)
            run_split(engine, "Hard", P0, hard_rewrites, batch_usage(hard_usage))
            candidate_arms = P0[1:3]+P0[4:]
            best = max((row for row in report_rows if row["arm"] in candidate_arms),
                       key=lambda row:(row["qs"], row["recall_at_10"], -row["total_tokens"]))
            (args.output/"selection.json").write_text(json.dumps(dict(
                selected=best["arm"], rule="Highest Hard QS among the two P0 directions; break ties by recall then tokens",
                selected_metrics=best), indent=2)+"\n")
            if not args.hard_only:
                main_qs = [{"id": q["id"], "question": q["question"]} for q in splits["Main"]]
                main_rewrites, main_usage = run_batches(args.cache, "rewrite-main", TERRA, REWRITE, main_qs,
                                                        QUERIES, 48, 32000)
                finalists = ("baseline", best["arm"])
                run_split(engine, "Main", finalists, main_rewrites, batch_usage(main_usage))
                full = []
                for arm in finalists:
                    arm_rows = [row for row in report_rows if row["arm"]==arm]
                    n = sum(row["n"] for row in arm_rows)
                    full.append(dict(split="Main+Hard", arm=arm, n=n,
                                     qs=sum(r["qs"]*r["n"] for r in arm_rows)/n,
                                     recall_at_10=sum(r["recall_at_10"]*r["n"] for r in arm_rows)/n,
                                     total_tokens=sum(r["total_tokens"] for r in arm_rows)))
                write_table(args.output, report_rows+full)
        finally:
            engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "official", "model", "cache", "output"):
        parser.add_argument(name, type=Path)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--hard-only", action="store_true")
    phase.add_argument("--main-baseline-only", action="store_true")
    main(parser.parse_args())
