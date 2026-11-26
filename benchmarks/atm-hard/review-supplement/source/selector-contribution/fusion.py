"""Combine a precise early rank with the previous coverage shortlist."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs


def merge_order(before, model):
    positions = {key: n for n, key in enumerate(before)}
    alternate = {key: n for n, key in enumerate(model)}
    return sorted(before, key=lambda key: (0.57 * positions[key] + 0.43 * alternate.get(key, 11), positions[key]))


def combine(before, model, portfolio, outside, facts, evidence):
    score = defaultdict(float)
    runs = (before, model, portfolio, outside)
    for ranking, weight in zip(runs, (5, 2.5, 1.5, 1), strict=True):
        for i, key in enumerate(ranking, 1):
            score[key] += weight * 16 / (16 + i)
    old = set(before)
    candidate = old | (set(model) & (set(portfolio) | set(outside)) & facts.keys() & evidence.keys())
    best = sorted(candidate, key=lambda key: (-score[key], key))[:10]
    return merge_order(best, model)


def main(args):
    sys.path[:0] = [str(args.official.resolve())]
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval

    helper = read_module("fusion_helpers", ROOT / "selector-panels/run.py")
    metrics = read_module("fusion_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("fusion_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    original = helper.keyed(ROOT / "selector-witness/review-rankings.jsonl")
    ledger = helper.keyed(args.input / "rankings.jsonl")
    portfolio = helper.keyed(args.input / "portfolio-rankings.jsonl")
    packets = helper.keyed(args.input / "reader-packets.jsonl")
    scope = helper.keyed(ROOT / "selector-scope/challenge-rankings.jsonl")
    initial = helper.keyed(ROOT / "selector-witness/rankings.jsonl")
    saved = helper.keyed(ROOT / "selector-witness/review-reader-packets.jsonl")
    qs = splits["Hard"]
    traces = []
    evidence = []
    alterations = Counter()
    for q in qs:
        key, query = q["id"], q["question"]
        old = original[key]["review"]
        model = ledger[key]["model"]
        selected = portfolio[key]["preserved"]
        independent = scope[key]["independent"]
        assert old == ledger[key]["previous"] == portfolio[key]["previous"]
        proof = {**initial[key]["claims"], **original[key]["newly_quoted"]}
        cards = {item["id"]: item["text"]
                 for field in ("model", "contextual", "guarded")
                 for item in packets[key]["evidence"][field]}
        # The original quote-checked packet supplies old IDs even if another ranker did not select them.
        previous_cards = saved[key]["evidence"]
        cards.update({doc["id"]: doc["text"] for doc in previous_cards})
        ordering = merge_order(old, model)
        fused = combine(old, model, selected, independent, proof, cards)
        trace = dict(id=key, previous=old, ordering=ordering, agreement=fused)
        arms = ("ordering", "agreement")
        snippets = {}
        for arm in arms:
            rank = trace[arm]
            assert len(rank) == len(set(rank)) == 10 and set(rank) <= records.keys()
            docs = [dict(id=doc, text=cards[doc]) for doc in rank]
            assert sum(len(doc["text"].encode()) for doc in docs) <= BUDGET
            snippets[arm] = docs
            alterations[arm] += set(rank) != set(old)
        traces.append(trace)
        evidence.append(dict(id=key, question=query, evidence=snippets))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fusion-rankings", traces), ("fusion-reader-packets", evidence)):
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    report = dict(n=len(qs), data_sha256=hashes, evidence_budget_bytes=BUDGET, model_api_tokens=0,
                  changes=dict(alterations),
                  metrics=metrics.measure(qs, traces, ("previous", *arms), comprehensive_eval))
    (args.output / "fusion-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "official", "input", "output"):
        parser.add_argument(arg, type=Path)
    main(parser.parse_args())
