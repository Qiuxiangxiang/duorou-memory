"""Retrieve and rank evidence separately for answer requirements on public ATM Hard."""
import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
from math import log1p
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

PROMPT = """You are verifying one evidence requirement for a question. Compare the records
only on whether they establish part of THIS requirement, in the complete question's scope.
If the requirement is a collection, retain different directly evidenced occurrences, cities,
names, dates or items. A recorded location and date can prove observed presence. An itinerary,
ticket, booking, receipt and invitation have different roles. Identify which event is recorded
and whether its date is capture, scheduled, transaction or actual. For timeline endpoints
look for the explicit outbound or return leg, before/after boundary, or arrival evidence.
For travel use exact places and event names; a same city or conference reference alone is
not enough. Keep direct evidence that supports a requested item even if it proves only a
small part. Reject assumptions not visible in the excerpt. Score 3 for directly supported,
2 for concrete partial evidence, 1 for weak corroboration. Return up to 10 useful records,
strongest first. Provide an exact contiguous excerpt quote and a specific short fact per
record. Report evidence only, without conclusions or external facts."""
RATING = {"type": "object", "additionalProperties": False,
          "required": ["id", "quote", "fact", "strength"],
          "properties": {"id": {"type": "string"}, "quote": {"type": "string"},
                         "fact": {"type": "string"},
                         "strength": {"type": "integer", "enum": [1, 2, 3]}}}
FIELDS = {"ranked": {"type": "array", "items": RATING}}
TEMPORAL = re.compile(r"\b(?:how many (?:days|nights)|how long|duration|start and end)\b", re.I)


def union(*lists):
    return list(dict.fromkeys(key for group in lists for key in group))


def interleave(*lists):
    return list(dict.fromkeys(key for group in zip_longest(*lists) for key in group if key))


def choose(before, proofs, positions, people, conservative):
    coverage = Counter()
    all_found = union(before, list(proofs))
    history = []
    selection = []
    while all_found and len(selection) < 10:
        order = {k: n for n, k in enumerate(before)}

        def score(key):
            facts = proofs.get(key, [])
            top = max((f["strength"] for f in facts), default=0)
            role = max((0.65 * f["strength"] / (1 + coverage[f["lane"]]) for f in facts), default=0)
            prior = ((2.05 if conservative else 1.25) - .055 * order[key]) if key in order else 0
            bonus = .22 * log1p(people[key])
            return .82 * top + role + prior + bonus

        key = min(all_found, key=lambda i: (-score(i), positions[i], i))
        history.append(dict(id=key, score=round(score(key), 6), facts=proofs.get(key, [])))
        selection.append(key)
        all_found.remove(key)
        for lane in {f["lane"] for f in proofs.get(key, [])}:
            coverage[lane] += 1
    return selection, history


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from cloud import TERRA
    from duorou.retrieval import BM25, terms, rrf
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    helpers = read_module("lane_helpers", ROOT / "selector-panels/run.py")
    grader = read_module("lane_api", ROOT / "event-graph-full/benchmark.py")
    metrics = read_module("lane_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    if hashes != read_module("lane_data", ROOT / "iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Only pinned public ATM-Bench evidence can be used")
    sources = read_module("lane_sources", ROOT / "event-links/links.py").load_sources(args.data)
    event = helpers.keyed(ROOT / "event-graph-full/sources.jsonl")
    tags = helpers.keyed(ROOT / "event-hierarchy/source-tags.jsonl")
    base = helpers.keyed(ROOT / "selector-contribution/fusion-rankings.jsonl")
    witness = helpers.keyed(ROOT / "selector-witness/rankings.jsonl")
    review = helpers.keyed(ROOT / "selector-witness/review-rankings.jsonl")
    geo = helpers.keyed(ROOT / "selector-geography/pools.jsonl")
    existing = helpers.keyed(ROOT / "selector-panels/rankings.jsonl")
    scope = helpers.keyed(ROOT / "selector-scope/rankings.jsonl")
    full = BM25({key: record["text"] for key, record in records.items()})
    briefs = {}
    for key, source in sources.items():
        title = " ".join(source.fields.get(name, "") for name in ("short_summary", "short_caption", "ocr_text"))
        briefs[key] = " ".join([title, source.city, str(source.day), *event.get(key, {}).get("index_terms", [])])
    work_index = BM25(briefs)
    qs = splits["Hard"]
    setups = {}
    frames = {}
    jobs = []
    history = {}
    for q in qs:
        key, query = q["id"], q["question"]
        before = base[key]["ordering"]
        profile = plan({"question": query, "qtype": ""})
        words = set(terms(query)) - DIRECT_STOP
        cards = {}

        def body(source_id):
            if source_id in cards:
                return cards[source_id]
            if source_id in sources and source_id in tags:
                if source_id not in frames:
                    frames[source_id] = frame(sources[source_id])
                source_text, _ = project(sources[source_id], tags[source_id], frames[source_id],
                                         profile, words, 1700)
            else:
                source_text = helpers.clip(records[source_id]["text"], 1700)
            cards[source_id] = source_text
            return source_text

        claims = {**witness[key]["claims"], **review[key]["newly_quoted"]}
        prior = Counter()
        for ordering in (before, existing[key]["after"], scope[key]["scope_cross"], witness[key]["guarded"]):
            prior.update(ordering)
        setups[key] = (before, cards, prior)
        for src in before:
            body(src)
        if not (q["qtype"] == "open_end" or (q["qtype"] == "number" and TEMPORAL.search(query))):
            continue
        reqs = witness[key]["requirements"]
        for lane, requirement in enumerate(reqs, 1):
            lead = " ".join(dict.fromkeys([*terms(query), *terms(requirement)]))
            lookup = []
            for index in (full, work_index):
                lookup.extend((index.search(lead, 18), index.search(requirement, 13)))
            scores = rrf(lookup)
            lead_ids = sorted(scores, key=lambda src: (-scores[src],src))
            separate = [(src, max(f["strength"] for f in facts if f["requirement"] == lane))
                        for src, facts in claims.items() if any(f["requirement"] == lane for f in facts)]
            support = [src for src, _ in sorted(separate, key=lambda pair: (-pair[1],-prior[pair[0]],pair[0]))]
            region = geo[key]["extra"][:20]
            chosen = interleave(support, lead_ids, before, region, scope[key]["scope_cross"])[:27]
            if len(chosen) < 10:
                continue
            label = f"{key}:{lane}"
            history[label] = key, lane, set(chosen)
            jobs.append(dict(id=label, question=query, requirement=requirement,
                             sources=[dict(id=src, text=body(src)) for src in chosen]))
    print("Requirement-specific panels", len(jobs), "records", sum(len(x["sources"]) for x in jobs), flush=True)
    if args.prepare_only:
        return
    raw, usage = grader.gather(args.cache, "hard-requirement-lanes-v1", TERRA, PROMPT,
                               jobs, FIELDS, 3, 143000, args.workers)
    proofs = defaultdict(lambda: defaultdict(list))
    valids = 0
    for ident, output in raw.items():
        qid, lane, permitted = history[ident]
        cards = setups[qid][1]
        seen = set()
        for row in output["ranked"][:10]:
            src, text = row["id"], row["quote"]
            if src not in permitted or src in seen or len(text.strip()) < 9 or text not in cards[src] or not row["fact"].strip():
                continue
            proofs[qid][src].append({**row, "lane":lane})
            valids += 1
            seen.add(src)
    traces = []
    packets = []
    for q in qs:
        key, query = q["id"], q["question"]
        before, cards, votes = setups[key]
        positions = {src: i for i, src in enumerate(cards)}
        row = dict(id=key, previous=before, proof_count=sum(len(facts) for facts in proofs[key].values()))
        evidence = {}
        row["decisions"] = {}
        for arm in ("covered", "guarded"):
            top, decisions = choose(before, proofs[key], positions, votes, arm=="guarded") if proofs[key] else (before, [])
            assert len(top) == len(set(top)) == 10 and set(top) <= set(cards)
            row[arm] = top
            row["decisions"][arm] = decisions
            docs = [{"id":src, "text":cards[src]} for src in top]
            assert sum(len(doc["text"].encode()) for doc in docs) <= BUDGET
            evidence[arm] = docs
        traces.append(row)
        packets.append(dict(id=key, question=query, evidence=evidence))
    args.output.mkdir(parents=True, exist_ok=True)
    grader.write_rows(args.output / "rankings.jsonl", traces)
    grader.write_rows(args.output / "reader-packets.jsonl", packets)
    report = dict(n=len(qs), data_sha256=hashes, evidence_budget_bytes=BUDGET, usage=usage,
                  panels=len(jobs), reviewed_records=sum(len(x["sources"]) for x in jobs),
                  active_questions=len({key for key,_,_ in history.values()}), valid_evidence_links=valids,
                  metrics=metrics.measure(qs, traces, ("previous","covered","guarded"), comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("release", "data", "official", "cache", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    main(parser.parse_args())
