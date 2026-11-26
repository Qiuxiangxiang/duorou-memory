"""Compare source-level model relevance over existing selections and new corpus results."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from cloud import TERRA
from evaluate import BUDGET, read_module
from run import load_inputs

PROMPT = """Rate EVERY source independently against the same question. A source that directly
supports even one requested answer item deserves full credit. Use the requested subject, time,
place, action and kind of proof. Score 4 for explicit requested evidence, 3 for grounded
partial evidence, 2 for uncertain indirect support, 1 for general topic, 0 for unrelated or
contradicted scope. For collections of media, rate each matching observation independently.
A capture with reliable city and date can support a visit. Match regions and date range.
An email date, an itinerary travel date, and a capture date have different meanings. When
the question asks what was paid, distinguish transaction proof and booking details. Do not
guess details missing from the source. Return the exact source ID, score and a short verbatim
quote that justifies the score. For scores 0 or 1, the quote may be empty. No answers."""
FIELDS = {"ratings": {"type": "array", "items": {"type": "object", "additionalProperties": False,
    "required": ["source_id", "score", "quote"], "properties": {
        "source_id": {"type": "string"}, "score": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
        "quote": {"type": "string"}}}}}


def sample(items, limit):
    if len(items) <= limit:
        return items
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame

    panels = read_module("scope_cards", ROOT / "selector-panels/run.py")
    grader = read_module("scope_model", ROOT / "event-graph-full/benchmark.py")
    metrics = read_module("scope_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    pins = read_module("scope_public_pins", ROOT / "iterative-search/search.py")
    if hashes != pins.PUBLIC_SHA256:
        raise ValueError("Only pinned public ATM-Bench evidence can be sent")
    source_adapter = read_module("scope_source_adapter", ROOT / "event-links/links.py")
    sources = source_adapter.load_sources(args.data)
    tags = panels.keyed(ROOT / "event-hierarchy/source-tags.jsonl")
    scope = panels.keyed(args.scope / "rankings.jsonl")
    old = panels.keyed(ROOT / "selector-panels/nominations.jsonl")
    geo = panels.keyed(ROOT / "selector-geography/pools.jsonl")
    finals = defaultdict(list)
    with gzip.open(args.scope / "all-scores.jsonl.gz", "rt") as stream:
        for line in stream:
            qid, position, key, *_ = json.loads(line)
            if position <= 160:
                finals[qid].append(key)
    setup = {}
    requests = []
    response_index = {}
    frames = {}
    qs = splits["Hard"]
    for question in qs:
        qid, query = question["id"], question["question"]
        ranked = scope[qid]
        before = ranked["previous"]
        old_set = set(geo[qid]["before"])
        fresh = [src for src in finals[qid] if src not in old_set]
        fresh_geo = geo[qid]["extra"]
        nominations = sorted(old[qid]["nominations"], key=lambda row: -row["priority"])
        prior_shortlist = [row["id"] for row in nominations if row["id"] not in before][:16]
        all_sources = panels.union(before, ranked["scope_cross"], fresh[:25], prior_shortlist,
                                   sample(fresh_geo, 28))
        assert set(all_sources) <= records.keys()
        weights = {word: 1 for word in set(terms(query)) - DIRECT_STOP}
        shown = {}
        for key in all_sources:
            if key in sources and key in tags:
                if key not in frames:
                    frames[key] = frame(sources[key])
                shown[key] = panels.make_card(sources[key], tags[key], frames[key], weights, 1250)
            else:
                shown[key] = panels.clip(records[key]["text"], 1250)
        setup[qid] = (before, all_sources, shown, set(fresh) | set(fresh_geo))
        for start in range(0, len(all_sources), 20):
            group = all_sources[start:start + 20]
            request_id = f"{qid}:{start // 20 + 1}"
            response_index[request_id] = (qid, set(group))
            requests.append(dict(id=request_id, question=query,
                candidates=[dict(id=key, source_type=tags.get(key, {}).get("source_kind", ""),
                                 text=shown[key]) for key in group]))
    print("Prepared", len(requests), "panels,", sum(len(row["candidates"]) for row in requests),
          "question-source pairs", flush=True)
    if args.prepare_only:
        return
    answers, usage = grader.gather(args.cache, "hard-scope-grades-v1", TERRA, PROMPT, requests,
                                  FIELDS, 3, 146000, args.workers)
    assessments = defaultdict(dict)
    for request_id, result in answers.items():
        qid, allowed = response_index[request_id]
        shown = setup[qid][2]
        for row in result["ratings"]:
            key, score, quote = (row[field] for field in ("source_id", "score", "quote"))
            if key in allowed and key not in assessments[qid] and (
                    score < 2 or (len(quote.strip()) >= 8 and quote in shown[key])):
                assessments[qid][key] = row
    scores = []
    packets = []
    changes = Counter()
    for question in qs:
        qid = question["id"]
        before, candidates, shown, new_ids = setup[qid]
        rating = {key: row["score"] for key, row in assessments[qid].items()}
        positions = {key: pos for pos, key in enumerate(candidates)}
        after = sorted(candidates, key=lambda key: (-rating.get(key, 0), positions[key]))[:10]
        # Retain a prior source unless a newly scored source clearly outranks it.
        challenger = sorted((key for key in candidates if key not in before and rating.get(key) == 4),
                            key=positions.get)
        removable = sorted((key for key in before if rating.get(key, 4) <= 2),
                           key=lambda key: (rating[key], -positions[key]))
        replacements = list(zip(removable[:3], challenger[:3]))
        mapping = dict(replacements)
        blended = [mapping.get(key, key) for key in before]
        for rank in (after, blended):
            assert len(rank) == len(set(rank)) == 10 and sum(len(shown[key].encode()) for key in rank) <= BUDGET
        scores.append(dict(id=qid, previous=before, independent=after, blended=blended,
                           replacement_pairs=replacements, ratings=list(assessments[qid].values()),
                           candidates=candidates, corpus_additions=sorted(new_ids.intersection(candidates))))
        packets.append(dict(id=qid, question=question["question"],
                            evidence=[dict(id=key, text=shown[key]) for key in blended]))
        gold = set(question["evidence_ids"])
        changes["added"] += len((set(blended) - set(before)) & gold)
        changes["lost"] += len((set(before) - set(blended)) & gold)
        changes["new_corpus_gold"] += len(set(blended) & new_ids & gold)
        changes["changed_questions"] += before != blended
    args.output.mkdir(parents=True, exist_ok=True)
    grader.write_rows(args.output / "challenge-rankings.jsonl", scores)
    grader.write_rows(args.output / "challenge-reader-packets.jsonl", packets)
    result = dict(n=len(qs), data_sha256=hashes, usage=usage, changes=dict(changes),
                  requested_ratings=sum(len(row["candidates"]) for row in requests),
                  valid_ratings=sum(map(len, assessments.values())),
                  metrics=metrics.measure(qs, scores, ("previous", "independent", "blended"), comprehensive_eval))
    (args.output / "challenge-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("release", "data", "official", "cache", "scope", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--prepare-only", action="store_true")
    main(parser.parse_args())
