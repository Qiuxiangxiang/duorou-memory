"""Rank every original source independently and evaluate the first ten."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

K = 10
PER_SOURCE = 3050
ARMS = ("scope", "retrieval", "verified")
RATING_FILES = {
    "scope": ("selector-scope/challenge-rankings.jsonl", "ratings"),
    "partial": ("selector-recheck/partial/rankings.jsonl", "graded"),
    "witness": ("selector-witness/rankings.jsonl", "claims"),
    "review": ("selector-witness/review-rankings.jsonl", "newly_quoted"),
}


def read_rows(name):
    with (ROOT / name).open() as file:
        return {row["id"]: row for row in map(json.loads, file)}


def load_grades(records):
    def validate(source, quote):
        if source not in records or len(quote.strip()) < 8:
            return False
        # Evidence cards can normalize whitespace or omit field prefixes.
        return " ".join(quote.split()) in " ".join(records[source]["text"].split())

    grades = defaultdict(lambda: defaultdict(dict))
    audit = defaultdict(int)
    for origin, (path, key) in RATING_FILES.items():
        for qid, row in read_rows(path).items():
            if origin in ("scope", "partial"):
                id_field = "source_id" if origin == "scope" else "id"
                scale = 4 if origin == "scope" else 3
                for grade in row[key]:
                    source, value, quote = grade[id_field], grade["score"], grade["quote"]
                    if value >= 2 and not validate(source, quote):
                        audit[origin + "_unmatched_quotes"] += 1
                        continue
                    grades[qid][source][origin] = value / scale
            else:
                for source, claims in row[key].items():
                    values = [(claim["strength"] - 1) / 3 for claim in claims
                              if validate(source, claim["quote"])]
                    if values:
                        existing = grades[qid][source].get("witness", 0)
                        grades[qid][source]["witness"] = max(existing, max(values))
                    else:
                        audit[origin + "_unmatched_quotes"] += 1
    return grades, dict(audit)


def rank_signal(rank):
    return 20 / (20 + rank)


def score_question(corpus, model, facts, grades):
    """Each result depends on its own signals, with no cross-source selection step."""
    cross_ranks = {
        source: rank for rank, source in enumerate(sorted(
            (key for key in corpus if corpus[key][3] is not None),
            key=lambda key: (-corpus[key][3], key)), 1)
    }
    model_ranks = {row["id"]: rank for rank, row in enumerate(
        sorted(model, key=lambda row: (-row["mixedbread"], row["id"])), 1)}
    fact_ranks = {row["id"]: rank for rank, row in enumerate(
        sorted(facts, key=lambda row: (-row["inverse"], row["id"])), 1)}
    ranked = {}
    for source, (position, topic, scope, _) in corpus.items():
        # Optional scorers contribute only where evaluated. Missing grades are unknown.
        cross = rank_signal(cross_ranks[source]) if source in cross_ranks else 0
        semantic = rank_signal(model_ranks[source]) if source in model_ranks else 0
        fact = rank_signal(fact_ranks[source]) if source in fact_ranks else 0
        retrieval = 0.50 * rank_signal(position) + 0.23 * cross + 0.17 * semantic + 0.10 * fact
        ratings = grades.get(source, {})
        # Count the two witness passes as one grader. Direct evidence needs a literal quote.
        # The retrieval score supplies a prior for missing/sparse judgments and breaks ties.
        verified = (1.5 * retrieval + sum(ratings.values())) / (1.5 + len(ratings))
        ranked[source] = (scope, retrieval, verified)
    return ranked


def main(args):
    started = perf_counter()
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()),
                    str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    reference = read_rows("selector-answer-trace/agreement-rankings.jsonl")
    pretrained = read_rows("selector-pretrained/rankings.jsonl")
    facts = read_rows("selector-inverse/rankings.jsonl")
    metrics = read_module("source_ranking_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("source_ranking_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    sources = read_module("source_ranking_links", ROOT / "event-links/links.py").load_sources(args.data)
    tags = read_rows("event-hierarchy/source-tags.jsonl")
    grades, grade_audit = load_grades(records)
    questions = splits["Hard"]
    corpus = defaultdict(dict)
    with gzip.open(ROOT / "selector-scope/all-scores.jsonl.gz", "rt") as stream:
        for line in stream:
            qid, rank, source, topic, scope, cross = json.loads(line)
            assert source not in corpus[qid]
            corpus[qid][source] = (rank, topic, scope, cross)

    args.output.mkdir(parents=True, exist_ok=True)
    traces, frames = [], {}
    changes = {arm: defaultdict(int) for arm in ARMS}
    with (gzip.open(args.output / "all-scores.jsonl.gz", "wt") as score_file,
          (args.output / "rankings.jsonl").open("w") as rankings,
          (args.output / "reader-packets.jsonl").open("w") as packets):
        for question in questions:
            qid, query = question["id"], question["question"]
            assert corpus[qid].keys() == records.keys()
            per_source = score_question(corpus[qid], pretrained[qid]["scores"],
                                        facts[qid]["scores"], grades[qid])
            order = {arm: sorted(per_source, key=lambda key: (-per_source[key][i], key))
                     for i, arm in enumerate(ARMS)}
            row = dict(id=qid, reference=reference[qid]["protected_agreement"])
            row.update({arm: keys[:K] for arm, keys in order.items()})
            rank_positions = {arm: {key: i for i, key in enumerate(keys, 1)}
                              for arm, keys in order.items()}
            for key in order["verified"]:
                score_row = [qid, key, *per_source[key],
                             *(rank_positions[arm][key] for arm in ARMS), grades[qid].get(key, {})]
                score_file.write(json.dumps(score_row, separators=(",", ":")) + "\n")

            requirements = plan({"question": query, "qtype": ""})
            tokens = set(terms(query)) - DIRECT_STOP
            evidence = {}
            for arm in ("reference", *ARMS):
                ids = row[arm]
                assert len(ids) == len(set(ids)) == K and set(ids) <= records.keys()
                documents = []
                for key in ids:
                    if key in sources and key in tags:
                        if key not in frames:
                            frames[key] = frame(sources[key])
                        content, spans = project(sources[key], tags[key], frames[key],
                                                 requirements, tokens, PER_SOURCE)
                    else:
                        content, spans = metrics.utf8_prefix(records[key]["text"], PER_SOURCE), []
                    documents.append(dict(id=key, text=content, source_spans=spans))
                assert sum(len(item["text"].encode()) for item in documents) <= BUDGET
                evidence[arm] = documents
            rankings.write(json.dumps(row, ensure_ascii=False) + "\n")
            packets.write(json.dumps(dict(id=qid, question=query, evidence=evidence),
                                     ensure_ascii=False) + "\n")
            traces.append(row)

            gold = set(question["evidence_ids"])
            before = set(row["reference"])
            for arm in ARMS:
                chosen = set(row[arm])
                changes[arm]["added_gold"] += len((chosen - before) & gold)
                changes[arm]["lost_gold"] += len((before - chosen) & gold)
                changes[arm]["changed_questions"] += before != chosen

    result = dict(n=len(questions), sources=len(records), pairs=len(records) * len(questions),
                  data_sha256=hashes, evidence_budget_bytes=BUDGET, new_model_api_tokens=0,
                  scored_with_model_judgments=sum(map(len, grades.values())),
                  grade_quote_audit=grade_audit,
                  all_scores_columns=["qid", "source", "scope_score", "retrieval_score",
                                      "verified_score", "scope_rank", "retrieval_rank",
                                      "verified_rank", "grades"],
                  elapsed_seconds=perf_counter() - started, changes=changes,
                  metrics=metrics.measure(questions, traces, ("reference", *ARMS), comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "official", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
