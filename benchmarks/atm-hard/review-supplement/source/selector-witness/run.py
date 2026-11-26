"""Select original evidence by quoted answer-item coverage on ATM-Bench Hard."""
import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
from math import ceil, log1p
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

PLAN = """Work from the question alone. Identify the answer-bearing requirements that can each
be proved or partially proved by original sources. Use up to 5. For a question that asks to
retrieve source IDs, include the concrete condition each original record must satisfy, without
inventing which IDs. For a collection of visits, titles, hotels, payments or occurrences,
describe what distinguishes separate items. For a duration include independently supported
endpoints when needed. Do not guess any answers or introduce unsupported time or place limits."""
PLAN_FIELDS = {"requirements": {"type": "array", "items": {"type": "string"}}}
EVIDENCE = """For the question and requirements, extract the most directly useful ORIGINAL
sources from this group. Return at most 16 sources. Ignore general topic matches. Prefer proof
of specific requested facts or distinct answer items, including incomplete but exact evidence.
For each chosen source provide 1 or 2 claims. Every claim needs an exact contiguous quote
from that source's text, a requirement index from 1, and a concise canonical item key. Use
the same lowercase item key for records of the same city, country, hotel, conference, paper,
transaction or trip endpoint. Include date, itinerary leg, occurrence or amount in the key
only when it identifies a separate requested item. For requested media ID lists, each matching
original is itself an item, even if scenes look similar. Scope and match exact entity,
activity, media type, and date. For a trip distinguish actual captured presence, planned route,
and document date. For payment distinguish evidence of charge and booking. When a travel
question needs departure and return, retain evidence for either endpoint. For a paper question
retain each distinct directly observed title. Use strength 4 for a directly supported requested
item or fact, 3 for a specific partial claim, and 2 for indirect corroboration. Return only
claims that can be supported by exact literal quote in supplied text. Do not answer the question."""
CLAIM = {"type": "object", "additionalProperties": False,
         "required": ["requirement", "item", "quote", "strength"],
         "properties": {"requirement": {"type": "integer"}, "item": {"type": "string"},
                        "quote": {"type": "string"},
                        "strength": {"type": "integer", "enum": [2, 3, 4]}}}
FIELDS = {"sources": {"type": "array", "items": {"type": "object", "additionalProperties": False,
          "required": ["source_id", "claims"],
          "properties": {"source_id": {"type": "string"}, "claims": {"type": "array", "items": CLAIM}}}}}
IDS = re.compile(r"\b(?:ids?|photos?|images?|videos?|media|slides)\b", re.I)
INTENT = re.compile(r"\b(?:recall|retrieve|find|report|show|captured)\b", re.I)


def normalize(text):
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def media_records(query, qtype):
    return qtype == "list_recall" and bool(IDS.search(query) and INTENT.search(query))


def union_round_robin(*lists):
    return list(dict.fromkeys(key for group in zip_longest(*lists) for key in group if key))


def valid_claims(result, shown, count, identifiers):
    accepted = {}
    for row in result["sources"][:16]:
        key = row["source_id"]
        if key not in shown or key in accepted:
            continue
        claims = []
        for claim in row["claims"][:2]:
            index, quote, item = (claim[field] for field in ("requirement", "quote", "item"))
            item = key if identifiers else normalize(item)[:150]
            if 1 <= index <= count and item and len(quote.strip()) >= 9 and quote in shown[key]:
                claims.append({**claim, "item": item})
        if claims:
            accepted[key] = claims
    return accepted


def choose(before, candidates, assessments, votes, mode, identifiers):
    prior = {key: rank for rank, key in enumerate(before)}
    order = {key: rank for rank, key in enumerate(candidates)}
    choices = list(dict.fromkeys([*before, *assessments]))
    repeated = Counter()
    covered = Counter()
    result = []
    history = []
    while choices and len(result) < 10:
        def locate(key):
            proofs = assessments.get(key, [])
            strongest = max((proof["strength"] for proof in proofs), default=0)
            value = {0: 0.0, 2: 0.8, 3: 1.8, 4: 3.1}[strongest]
            if key in prior:
                value += (1.30 if mode == "balanced" else 2.35) - 0.055 * prior[key]
            elif not proofs or strongest < 3:
                value -= 0.50
            value += 0.24 * log1p(votes[key])
            if proofs:
                requirement_bonus = max(0.32 / (1 + covered[proof["requirement"]]) for proof in proofs)
                item_bonus = max((0.75 if proof["strength"] == 4 else 0.45) / (1 + repeated[proof["item"]])
                                 for proof in proofs)
                repeat_cost = min(0.49 * repeated[proof["item"]] for proof in proofs)
                value += requirement_bonus + item_bonus - (0 if identifiers else repeat_cost)
            return value

        best = min(choices, key=lambda key: (-locate(key), order[key], key))
        history.append(dict(id=best, score=round(locate(best), 6), claims=assessments.get(best, [])))
        result.append(best)
        choices.remove(best)
        for proof in assessments.get(best, []):
            repeated[proof["item"]] += 1
            covered[proof["requirement"]] += 1
    return result, history


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame
    from cloud import TERRA, JUDGE

    helpers = read_module("witness_cards", ROOT / "selector-panels/run.py")
    grader = read_module("witness_api", ROOT / "event-graph-full/benchmark.py")
    metrics = read_module("witness_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    if hashes != read_module("witness_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Only the hash-pinned public ATM-Bench inputs can be used")
    sources = read_module("witness_sources", ROOT / "event-links/links.py").load_sources(args.data)
    current = helpers.keyed(ROOT / "selector-coverage/consensus-rankings.jsonl")
    panel_rank = helpers.keyed(ROOT / "selector-panels/rankings.jsonl")
    first = helpers.keyed(ROOT / "selector-set/rankings.jsonl")
    scoped = helpers.keyed(ROOT / "selector-scope/rankings.jsonl")
    challenge = helpers.keyed(ROOT / "selector-scope/challenge-rankings.jsonl")
    geographic = helpers.keyed(ROOT / "selector-geography/pools.jsonl")
    tags = helpers.keyed(ROOT / "event-hierarchy/source-tags.jsonl")
    qs = splits["Hard"]
    shape = [{"id": q["id"], "question": q["question"]} for q in qs]
    setups = {}
    frames = {}
    for q in qs:
        qid, query = q["id"], q["question"]
        control = current[qid]["contextual"]
        big = panel_rank[qid]
        grades = sorted(challenge[qid]["ratings"], key=lambda row: -row["score"])
        strong = [row["source_id"] for row in grades if row["score"] >= 3]
        nominations = sorted(big["nominations"], key=lambda row: -row["priority"])
        nominated = [row["id"] for row in nominations]
        geo = geographic[qid]["extra"]
        # Interleave candidate generators to retain independent retrieval routes.
        broad = union_round_robin(big["shortlist"], nominated, strong, geo)
        candidates = helpers.union(control, current[qid]["previous"], first[qid]["after"],
                                   scoped[qid]["scope_cross"][:6], broad)[:108]
        assert len(candidates) >= 10 and set(candidates) <= records.keys()
        votes = Counter()
        for ranking in (control, big["after"], first[qid]["after"], challenge[qid]["independent"],
                        scoped[qid]["scope_cross"], nominated[:20]):
            votes.update(ranking)
        weights = {word: 1 for word in set(terms(query)) - DIRECT_STOP}
        visible = {}
        for key in candidates:
            if key in sources and key in tags:
                if key not in frames:
                    frames[key] = frame(sources[key])
                raw = helpers.make_card(sources[key], tags[key], frames[key], weights, 1330)
                visible[key] = "Source type: " + tags[key]["source_kind"] + "\n" + raw
            else:
                visible[key] = helpers.clip(records[key]["text"], 1330)
        setups[qid] = control, candidates, visible, votes
    print("Hard witness candidates", sum(len(x[1]) for x in setups.values()), flush=True)
    if args.prepare_only:
        return

    plans, planning_usage = grader.gather(args.cache, "hard-witness-plan-v1", TERRA, PLAN, shape,
                                           PLAN_FIELDS, 10, 29000, args.workers)
    requests = []
    request_index = {}
    plan_by_q = {}
    is_id_query = {}
    for q in qs:
        qid, query = q["id"], q["question"]
        before, candidates, shown, _ = setups[qid]
        reqs = list(dict.fromkeys(text.strip() for text in plans[qid]["requirements"] if text.strip()))[:5]
        reqs = reqs or [query]
        plan_by_q[qid] = reqs
        is_id_query[qid] = media_records(query, q["qtype"])
        groups = ceil(len(candidates) / 25)
        for index in range(groups):
            group = candidates[index::groups]
            request_id = f"{qid}:{index+1}"
            request_index[request_id] = qid, {key: shown[key] for key in group}
            requests.append(dict(id=request_id, question=query, requirements=reqs,
                candidates=[dict(id=key, text=shown[key]) for key in group]))
    print("Hard witness panels", len(requests), flush=True)
    responses, assessment_usage = grader.gather(args.cache, "hard-witness-claims-v1", JUDGE, EVIDENCE,
                                                requests, FIELDS, 4, 155000, args.workers)
    quotes = defaultdict(dict)
    for request_id, result in responses.items():
        qid, shown = request_index[request_id]
        quotes[qid].update(valid_claims(result, shown, len(plan_by_q[qid]), is_id_query[qid]))

    traces = []
    packets = []
    changes = Counter()
    for q in qs:
        qid = q["id"]
        before, candidates, shown, votes = setups[qid]
        row = dict(id=qid, previous=before, candidate_ids=candidates, requirements=plan_by_q[qid],
                   claims=quotes[qid], decisions={})
        evidence = {}
        for mode in ("balanced", "guarded"):
            chosen, audited = choose(before, candidates, quotes[qid], votes, mode, is_id_query[qid])
            assert len(chosen) == len(set(chosen)) == 10 and set(chosen) <= set(candidates)
            assert sum(len(shown[key].encode()) for key in chosen) <= BUDGET
            row[mode] = chosen
            row["decisions"][mode] = audited
            evidence[mode] = [dict(id=key, text=shown[key]) for key in chosen]
            changes[mode] += set(chosen) != set(before)
        traces.append(row)
        packets.append(dict(id=qid, question=q["question"], evidence=evidence))
    args.output.mkdir(parents=True, exist_ok=True)
    grader.write_rows(args.output / "rankings.jsonl", traces)
    grader.write_rows(args.output / "reader-packets.jsonl", packets)
    report = dict(n=len(qs), data_sha256=hashes, evidence_budget_bytes=BUDGET,
                  source_reviews=sum(len(row["candidates"]) for row in requests),
                  valid_quoted_sources=sum(map(len, quotes.values())),
                  planning_usage=planning_usage, assessment_usage=assessment_usage, changes=dict(changes),
                  metrics=metrics.measure(qs, traces, ("previous", "balanced", "guarded"), comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("release", "data", "official", "cache", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--prepare-only", action="store_true")
    main(parser.parse_args())
