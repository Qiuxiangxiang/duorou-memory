"""Recheck unnominated sources with longer source text and per-source decisions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

PROMPT = """Assess each supplied original separately. Some sources have short descriptions,
but may establish evidence via the date, captured location, exact title, OCR, or raw excerpt.
For each source return its ID and up to two claims that it can actually support. Return an
empty claims list if the source is only a topic match or outside the question's scope. Match
each claim to a numbered requirement and quote a literal contiguous part of the text. Name
the concrete item that is established, using a canonical lowercase key. Do not guess from
an ambiguous photo. For travel, retain specific visit or itinerary observations even when
they support only one part of the visit/route. For a question asking for the original media,
the original matching source is the item. Use strength 4 for direct, 3 for specific partial,
2 for useful corroboration. Do not answer the question or create proof not in the source."""


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from cloud import TERRA
    from duorou.retrieval import BM25, rrf, terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    helpers = read_module("review_cards", ROOT / "selector-panels/run.py")
    witness = read_module("witness_selection", ROOT / "selector-witness/run.py")
    grader = read_module("review_api", ROOT / "event-graph-full/benchmark.py")
    metrics = read_module("review_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    if hashes != read_module("review_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected pinned public data")
    sources = read_module("review_sources", ROOT / "event-links/links.py").load_sources(args.data)
    tags = helpers.keyed(ROOT / "event-hierarchy/source-tags.jsonl")
    prior = helpers.keyed(args.witness / "rankings.jsonl")
    grades = helpers.keyed(ROOT / "selector-scope/challenge-rankings.jsonl")
    panels = helpers.keyed(ROOT / "selector-panels/rankings.jsonl")
    first = helpers.keyed(ROOT / "selector-set/rankings.jsonl")
    scope = helpers.keyed(ROOT / "selector-scope/rankings.jsonl")
    prepared = {}
    history = {}
    calls = []
    frames = {}
    for question in splits["Hard"]:
        qid, query = question["id"], question["question"]
        old = prior[qid]
        before, candidate_ids, approved = (old[field] for field in ("previous", "candidate_ids", "claims"))
        missing = [key for key in candidate_ids if key not in approved and key not in before]
        index = BM25({key: records[key]["text"] for key in missing})
        ranks = [
            index.search(term, 18)
            for term in [query, *old["requirements"]] if term.strip()
        ]
        rank_scores = rrf(ranks)
        local = sorted(rank_scores, key=lambda key: (-rank_scores[key], key))
        assessment = sorted(grades[qid]["ratings"], key=lambda item: -item["score"])
        candidate_set = set(missing)
        # Prioritize sources surfaced independently and those matching the wording of a requirement.
        signals = [
            [x["source_id"] for x in assessment if x["score"] >= 3],
            scope[qid]["scope_cross"], local, missing,
        ]
        review_ids = witness.union_round_robin(*(list(key for key in row if key in candidate_set)
                                                for row in signals))[:28]
        body_limit = 2350
        evidence_types = plan({"question": query, "qtype": ""})
        selected_terms = set(terms(query)) - DIRECT_STOP
        cards = {}
        for key in candidate_ids:
            if key not in sources or key not in tags:
                cards[key] = helpers.clip(records[key]["text"], 1330 if key in approved else body_limit)
                continue
            if key not in frames:
                frames[key] = frame(sources[key])
            packet, _ = project(sources[key], tags[key], frames[key], evidence_types,
                                selected_terms, body_limit - 24)
            if key in approved:
                packet = helpers.make_card(sources[key], tags[key], frames[key],
                                           {word: 1 for word in selected_terms}, 1330)
            cards[key] = "Source type: " + tags[key]["source_kind"] + "\n" + packet
        votes = Counter()
        nominations = sorted(panels[qid]["nominations"], key=lambda item: -item["priority"])
        for order in (before, panels[qid]["after"], first[qid]["after"], grades[qid]["independent"],
                      scope[qid]["scope_cross"], [item["id"] for item in nominations[:20]]):
            votes.update(order)
        prepared[qid] = cards, candidate_ids, votes, review_ids
        for offset in range(0, len(review_ids), 12):
            keys = review_ids[offset:offset+12]
            ident = f"{qid}:{offset//12+1}"
            history[ident] = qid, {key: cards[key] for key in keys}
            calls.append(dict(id=ident, question=query, requirements=old["requirements"],
                              sources=[dict(id=key, text=cards[key]) for key in keys]))
    print("Hard second look", len(calls), "panels", sum(len(row["sources"]) for row in calls), "sources", flush=True)
    if args.prepare_only:
        return

    raw, usage = grader.gather(args.cache, "hard-witness-unnominated-v1", TERRA, PROMPT,
                               calls, witness.FIELDS, 3, 151000, args.workers)
    questions = {q["id"]: q for q in splits["Hard"]}
    assessed = defaultdict(dict)
    for key, result in raw.items():
        qid, shown = history[key]
        q = questions[qid]
        accepted = witness.valid_claims(result, shown, len(prior[qid]["requirements"]),
                                        witness.media_records(q["question"], q["qtype"]))
        assessed[qid].update(accepted)

    traces = []
    packets = []
    pool_hits = []
    changes = Counter()
    for qid, q in questions.items():
        before = prior[qid]["previous"]
        shown, candidates, votes, reviewed = prepared[qid]
        all_proof = {**prior[qid]["claims"], **assessed[qid]}
        use_ids = witness.media_records(q["question"], q["qtype"])
        ordered, decisions = witness.choose(before, candidates, all_proof, votes, "guarded", use_ids)
        eligible = set(before) | all_proof.keys()
        gold = set(q["evidence_ids"])
        pool_hits.append((len(gold.intersection(eligible)), len(gold)))
        changes["new_quoted_gold"] += len(gold.intersection(assessed[qid]))
        changes["final_added_gold"] += len(gold.intersection(set(ordered) - set(before)))
        changes["final_lost_gold"] += len(gold.intersection(set(before) - set(ordered)))
        changes["changed_questions"] += set(before) != set(ordered)
        assert len(ordered) == len(set(ordered)) == 10 and set(ordered) <= eligible
        assert sum(len(shown[key].encode()) for key in ordered) <= BUDGET
        assert all(proof["quote"] in shown[key] for key in ordered for proof in all_proof.get(key, []))
        traces.append(dict(id=qid, previous=before, first_round=prior[qid]["guarded"],
                           review=ordered, reviewed_ids=reviewed,
                           newly_quoted=assessed[qid], decisions=decisions))
        packets.append(dict(id=qid, question=q["question"],
                            evidence=[dict(id=key, text=shown[key]) for key in ordered]))
    args.output.mkdir(parents=True, exist_ok=True)
    grader.write_rows(args.output / "review-rankings.jsonl", traces)
    grader.write_rows(args.output / "review-reader-packets.jsonl", packets)
    result = dict(n=len(questions), data_sha256=hashes, evidence_budget_bytes=BUDGET, usage=usage,
                  review_count=sum(len(row["sources"]) for row in calls),
                  quoted_new_sources=sum(map(len, assessed.values())),
                  changes=dict(changes),
                  eligible_gold=sum(n for n, _ in pool_hits),
                  eligible_ceiling_at_10=mean(min(n, 10)/d for n, d in pool_hits),
                  metrics=metrics.measure(list(questions.values()), traces,
                                          ("previous", "first_round", "review"), comprehensive_eval))
    (args.output / "review-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("release", "data", "official", "cache", "witness", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--prepare-only", action="store_true")
    main(parser.parse_args())
