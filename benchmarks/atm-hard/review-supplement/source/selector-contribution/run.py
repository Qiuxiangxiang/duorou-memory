"""Compare contribution of quoted records using a shared question-level evidence ledger."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

PROMPT = """Rank original source IDs for the same question using the supplied quoted evidence
ledger. Select exactly ten sources when there are at least ten. Reconcile cross-source claim
keys before choosing. Each entry includes excerpts and may include earlier provisional claims.
Verify their usefulness against the source excerpt. A question asking for media IDs treats
every exact match as a distinct answer item. For other questions distinguish the actual facts
required: which named items or distinct events occurred, what dates or legs are established,
and what amount was charged. A source that gives a whole requested fact is more useful than
a general contextual mention. For visits, prioritize observed presence in the right scope;
also keep itinerary or ticket evidence that establishes a different date or transport leg.
A geographic capture date is the observation date. Separate visit from a home location,
transfer, plan, invitation, or nearby city with no applicable visit evidence. If a question
asks what dates or currency amounts, preserve records that establish both endpoints or
complementary transaction facts. Include independent, answer-bearing evidence across named
items and times before taking repetitive supporting records. Do not omit an important partial
fact just because another item has stronger evidence. For each chosen ID give a short verbatim
quote from its own excerpt and name the exact factual contribution. Return the ranking only.
Do not use sources outside the ledger, guess facts, or answer the question."""
FIELDS = {"ranked": {"type": "array", "items": {"type": "object", "additionalProperties": False,
          "required": ["id", "quote", "fact", "tier"],
          "properties": {"id": {"type": "string"}, "quote": {"type": "string"},
                         "fact": {"type": "string"}, "tier": {"type": "integer", "enum": [1, 2, 3]}}}}}
COLLECTION = re.compile(r"\b(?:all|which|list|times|nights|days|start and end|duration|dates?)\b", re.I)
MEDIA = re.compile(r"\b(?:photos?|images?|videos?|media|slides|ids?)\b", re.I)
MONEY = re.compile(r"\b(?:pay|paid|spend|spent|cost|reimburs\w*)\b", re.I)


def choose(before, response, shown, claims):
    selected = []
    proof = {}
    for row in response["ranked"][:10]:
        key, text = row["id"], row["quote"]
        if key in shown and key not in proof and len(text.strip()) >= 9 and text in shown[key] and row["fact"].strip():
            selected.append(key)
            proof[key] = row
    model = selected + [key for key in before if key not in selected]
    trusted = [key for key in selected if proof[key]["tier"] >= 2 and (key in before or key in claims)]
    # Require high-confidence differences and preserve explicit previous support when the
    # single new ordering disagrees with a second independent prior ranking.
    return model[:10], proof, trusted


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from cloud import JUDGE
    from duorou.retrieval import terms
    from memory import DIRECT_STOP, frame, plan, project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval

    helpers = read_module("contribution_helpers", ROOT / "selector-panels/run.py")
    grader = read_module("contribution_api", ROOT / "event-graph-full/benchmark.py")
    metrics = read_module("contribution_metrics", ROOT / "selector-corpus/run.py")
    sources = read_module("contribution_sources", ROOT / "event-links/links.py").load_sources(args.data)
    pins = read_module("contribution_pins", ROOT / "iterative-search/search.py")
    records, _, splits, hashes = load_inputs(args.data)
    if hashes != pins.PUBLIC_SHA256:
        raise ValueError("Expected pinned public ATM-Bench data")
    stage = helpers.keyed(ROOT / "selector-witness/rankings.jsonl")
    review = helpers.keyed(ROOT / "selector-witness/review-rankings.jsonl")
    tags = helpers.keyed(ROOT / "event-hierarchy/source-tags.jsonl")
    panel = helpers.keyed(ROOT / "selector-panels/rankings.jsonl")
    qs = splits["Hard"]
    setup = {}
    requests = []
    frames = {}
    for q in qs:
        qid, query = q["id"], q["question"]
        base = review[qid]["review"]
        first = stage[qid]
        claims = {**first["claims"], **review[qid]["newly_quoted"]}
        # Spread the candidate lists by original retrieval order, not model preferences.
        ids = helpers.union(base, first["previous"], [key for key in first["candidate_ids"] if key in claims])
        assert set(ids) <= records.keys()
        tokens = set(terms(query)) - DIRECT_STOP
        requirements = plan({"question": query, "qtype": ""})
        scope = COLLECTION.search(query) and not (MEDIA.search(query) or MONEY.search(query))
        shown = {}
        candidates = []
        for key in ids:
            if key not in sources or key not in tags:
                body = helpers.clip(records[key]["text"], 1250)
            else:
                if key not in frames:
                    frames[key] = frame(sources[key])
                body, _ = project(sources[key], tags[key], frames[key], requirements, tokens, 1030)
            # Preserve every previously verified quote alongside the excerpt.
            prefix = [claim["quote"] for claim in claims.get(key, []) if claim["quote"] not in body]
            text = "\n".join([*prefix, body])
            assert len(text.encode()) <= 3000
            shown[key] = text
            candidates.append(dict(
                id=key, type=tags.get(key, {}).get("source_kind", ""), excerpt=text,
                claim_keys=[dict(item=c["item"], strength=c["strength"], requirement=c["requirement"])
                            for c in claims.get(key, [])]))
        # Deterministic interleaving reduces block order for incumbent cards.
        ordered = candidates[::2] + candidates[1::2]
        setup[qid] = base, ids, shown, claims, scope
        if len(ids) > 10:
            requests.append(dict(id=qid, question=query, requirements=first["requirements"], ledger=ordered))
    print("Hard compact ledgers", len(requests), "questions", sum(len(x["ledger"]) for x in requests),
          "records", flush=True)
    if args.prepare_only:
        return

    outputs, usage = grader.gather(args.cache, "hard-compact-contribution-v1", JUDGE, PROMPT,
                                   requests, FIELDS, 3, 141000, args.workers)
    traces = []
    packets = []
    changes = Counter()
    eligible_counts = []
    for q in qs:
        qid, query = q["id"], q["question"]
        before, available, shown, confirmed, scope = setup[qid]
        model, proof, trusted = choose(before, outputs.get(qid, {"ranked": []}), shown, confirmed)
        # Use scope coverage on aggregate facts only. Exact-media and payment selections
        # have task-specific baselines. The separate unrestricted arm is still scored.
        contextual = model if scope and len(proof) >= 8 else before
        # Preserve candidates already chosen by two separate rankers. Add trusted records
        # before falling back to the previous ranking.
        strong = set(stage[qid]["previous"]) & set(panel[qid]["after"])
        agreed = [key for key in before if key in strong]
        guarded = helpers.union(agreed, trusted, before)[:10] if scope else before
        row = dict(id=qid, previous=before, model=model, contextual=contextual, guarded=guarded,
                   eligible=available, model_quotes=list(proof.values()))
        arms = ("model", "contextual", "guarded")
        evidence = {}
        for arm in arms:
            chosen = row[arm]
            assert len(chosen) == len(set(chosen)) == 10 and set(chosen) <= set(available)
            items = [dict(id=key, text=shown[key]) for key in chosen]
            assert sum(len(item["text"].encode()) for item in items) <= BUDGET
            evidence[arm] = items
            changes[arm] += set(chosen) != set(before)
        gold = set(q["evidence_ids"])
        eligible_counts.append((len(gold.intersection(available)), len(gold)))
        traces.append(row)
        packets.append(dict(id=qid, question=query, evidence=evidence))
    args.output.mkdir(parents=True, exist_ok=True)
    grader.write_rows(args.output / "rankings.jsonl", traces)
    grader.write_rows(args.output / "reader-packets.jsonl", packets)
    report = dict(n=len(qs), data_sha256=hashes, evidence_budget_bytes=BUDGET, usage=usage,
                  reviewed_questions=len(requests), changed_questions=dict(changes),
                  valid_new_judgments=sum(len(row["model_quotes"]) for row in traces),
                  eligibility_gold=sum(n for n, _ in eligible_counts),
                  eligibility_ceiling_at_10=mean(min(n, 10)/d for n, d in eligible_counts),
                  metrics=metrics.measure(qs, traces, ("previous", *arms), comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("release", "data", "official", "cache", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--prepare-only", action="store_true")
    main(parser.parse_args())
