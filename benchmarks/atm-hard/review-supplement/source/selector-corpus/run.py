"""Rank the complete ATM-Bench corpus by independent question-source relevance."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0"), str(ROOT / "p0-full")]
from encoder import MiniLM
from evaluate import BUDGET, read_module
from run import load_inputs

K = 10
BYTES_PER_SOURCE = 3050
RERANK_LIMIT = 100


def utf8_prefix(text, size):
    return text.encode("utf-8")[:size].decode("utf-8", errors="ignore")


def relevance_view(text, reranker):
    lines = reranker.ranking_view(text).splitlines()
    field_limits = {
        "short_summary": 420,
        "short_caption": 290,
        "ocr_text": 480,
        "caption": 450,
        "detail": 740,
    }
    return "\n".join(utf8_prefix(line, field_limits.get(line.partition(":")[0], 160))
                     for line in lines[:12])


def normalize(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def descending(scores):
    # IDs were sorted first. Equal scores therefore use source ID as a stable tie-breaker.
    return np.argsort(-scores, kind="stable")


def reciprocal(order):
    scores = np.empty(len(order))
    scores[order] = 1 / (60 + np.arange(1, len(order) + 1))
    return scores


def lexical_scores(index, question, positions, terms):
    scores = np.zeros(len(positions))
    for term in dict.fromkeys(terms(question)):
        for key, weight in index.postings.get(term, {}).items():
            scores[positions[key]] += weight
    return scores


def measure(questions, traces, fields, official):
    results = {}
    for field in fields:
        metrics_input = []
        recalls = defaultdict(list)
        reciprocal_ranks = []
        gold_hits = 0
        for question, trace in zip(questions, traces, strict=True):
            assert question["id"] == trace["id"]
            gold = set(question["evidence_ids"])
            order = trace[field]
            metrics_input.append(dict(gt_evidence_ids=question["evidence_ids"], retrieval_ids=order))
            hits = len(gold.intersection(order))
            recall = hits / len(gold)
            for group in ("overall", question["qtype"]):
                recalls[group].append(recall)
            gold_hits += hits
            reciprocal_ranks.append(next((1 / rank for rank, key in enumerate(order, 1) if key in gold), 0))
        recall_at_10 = official.compute_metrics(metrics_input, [K])["metrics"]["recall@10"]
        assert abs(recall_at_10 - mean(recalls["overall"])) < 1e-12
        results[field] = dict(recall_at_10=recall_at_10, gold_hits=gold_hits,
                              all_hit=recalls["overall"].count(1), zero_hit=recalls["overall"].count(0),
                              mrr_at_10=mean(reciprocal_ranks),
                              by_qtype={kind: mean(values) for kind, values in recalls.items()
                                        if kind != "overall"})
    return results


def run(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()),
                    str(ROOT / "memory-design")]
    from duorou.retrieval import BM25, terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    records, kinds, splits, hashes = load_inputs(args.data)
    pinned = read_module("public_dataset", ROOT / "iterative-search/search.py")
    if hashes != pinned.PUBLIC_SHA256:
        raise ValueError("ATM-Bench files do not match the pinned dataset")
    reranker = read_module("corpus_reranker", ROOT / "rerank/run.py")
    previous = {row["id"]: row["session_20"]
                for row in map(json.loads, args.previous.open())}
    source_loader = read_module("corpus_sources", ROOT / "event-links/links.py")
    sources = source_loader.load_sources(args.data)
    with (ROOT / "event-hierarchy/source-tags.jsonl").open() as stream:
        tags = {row["id"]: row for row in map(json.loads, stream)}
    questions = splits["Hard"]
    ids = sorted(records)
    positions = {key: i for i, key in enumerate(ids)}
    views = {key: relevance_view(records[key]["text"], reranker) for key in ids}
    raw_index = BM25({key: records[key]["text"] for key in ids})
    compact_index = BM25(views)
    dense = MiniLM(args.dense_model)
    # Score queries and source descriptions in the same normalized vector space.
    vectors = normalize(dense.encode([views[key] for key in ids]))
    query_vectors = normalize(dense.encode([question["question"] for question in questions]))
    cross = reranker.CrossEncoder(args.cross_model)
    print(f"Indexed {len(ids)} sources in {perf_counter() - started:.1f}s", flush=True)
    indexing_seconds = perf_counter() - started
    frames = {}
    traces = []
    seconds = []
    actual_rank_fields = ("bm25", "dense", "relevance", "relevance_cross")
    with (gzip.open(args.output / "corpus-ranks.jsonl.gz", "wt", encoding="utf-8") as scores,
          (args.output / "reader-packets.jsonl").open("w") as packets):
        for number, (question, query_vector) in enumerate(zip(questions, query_vectors, strict=True), 1):
            start = perf_counter()
            qid, query = question["id"], question["question"]
            raw_scores = lexical_scores(raw_index, query, positions, terms)
            compact_scores = lexical_scores(compact_index, query, positions, terms)
            dense_scores = vectors @ query_vector
            raw_order = descending(raw_scores)
            compact_order = descending(compact_scores)
            dense_order = descending(dense_scores)
            fused = 0.25 * reciprocal(raw_order) + 0.25 * reciprocal(compact_order) + 0.5 * reciprocal(dense_order)
            fused_order = descending(fused)
            shortlist = fused_order[:RERANK_LIMIT]
            cross_scores = cross.score(query, [views[ids[i]] for i in shortlist])
            cross_by_pos = dict(zip(map(int, shortlist), cross_scores, strict=True))
            cross_rank = sorted(cross_by_pos, key=lambda i: (-cross_by_pos[i], ids[i]))
            cross_reciprocal = {i: 1 / (60 + rank) for rank, i in enumerate(cross_rank, 1)}
            reranked = sorted(cross_rank, key=lambda i: (-(0.5 * fused[i] + 0.5 * cross_reciprocal[i]), ids[i]))
            orders = (raw_order, dense_order, fused_order, reranked)
            trace = {"id": qid, "previous": previous[qid], "sources_scored": len(ids), "bytes": {}}
            evidence_by_arm = {}
            # This path sees only the query text and public source fields. Labels are used after ranking.
            required = plan({"question": query, "qtype": ""})
            tokens = set(terms(query)) - DIRECT_STOP
            for arm, order in zip(actual_rank_fields, orders, strict=True):
                selected = [ids[i] for i in order[:K]]
                evidence = []
                for key in selected:
                    if key in sources and key in tags:
                        if key not in frames:
                            frames[key] = frame(sources[key])
                        body, spans = project(sources[key], tags[key], frames[key], required, tokens, BYTES_PER_SOURCE)
                    else:
                        body, spans = utf8_prefix(records[key]["text"], BYTES_PER_SOURCE), []
                    evidence.append({"id": key, "text": body, "source_spans": spans})
                byte_count = sum(len(row["text"].encode("utf-8")) for row in evidence)
                assert len(selected) == len(set(selected)) == K and byte_count <= BUDGET
                trace[arm] = selected
                trace["bytes"][arm] = byte_count
                if arm in ("relevance", "relevance_cross"):
                    evidence_by_arm[arm] = evidence
            for rank, pos in enumerate(fused_order, 1):
                scores.write(json.dumps([qid, rank, ids[pos], float(fused[pos]), float(raw_scores[pos]),
                                         float(compact_scores[pos]), float(dense_scores[pos]),
                                         cross_by_pos.get(int(pos))], separators=(",", ":")) + "\n")
            packets.write(json.dumps({"id": qid, "question": query, "evidence": evidence_by_arm},
                                     ensure_ascii=False) + "\n")
            traces.append(trace)
            seconds.append(perf_counter() - start)
            print(f"Hard {number}/{len(questions)}", flush=True)
    with (args.output / "rankings.jsonl").open("w") as stream:
        for row in traces:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = dict(n=len(questions), sources=dict(Counter(kinds.values())),
                   question_source_pairs=len(questions) * len(ids), data_sha256=hashes,
                   dense_sha256=dense.hashes, cross_encoder_sha256=cross.hashes,
                   evidence_budget_bytes=BUDGET, cross_encoder_pairs=RERANK_LIMIT * len(questions),
                   corpus_ranks_columns=["question_id", "rank", "source_id", "relevance", "bm25",
                                         "compact_bm25", "dense_cosine", "cross_encoder_logit"],
                   metrics=measure(questions, traces, ("previous", *actual_rank_fields), comprehensive_eval),
                   indexing_seconds=indexing_seconds, mean_seconds_per_question_with_diagnostics=mean(seconds),
                   model_api_tokens=0)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "official", "dense_model", "cross_model", "previous", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
