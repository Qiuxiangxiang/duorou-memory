"""Evaluate fixed relevance rank agreements on the full public Hard split."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

MEDIA_COLLECTION = re.compile(
    r"\b(?:photos?|images?|videos?|media|slides)\b", re.I)
FINANCE = re.compile(r"\b(?:paid|pay|spend|spent|cost|reimburs\w*)\b", re.I)


def use_question_scope(question, before, geographic, fused):
    if geographic:
        return geographic, "geographic_coverage"
    if MEDIA_COLLECTION.search(question) or FINANCE.search(question):
        return before, "precise_source_collection"
    return fused, "relevance_agreement"


def consensus(rankings, weights):
    fused = defaultdict(float)
    for ranked, weight in zip(rankings, weights, strict=True):
        for rank, key in enumerate(ranked, 1):
            fused[key] += weight * 15 / (15 + rank)
    return sorted(fused, key=lambda key: (-fused[key], key))[:10]


def promote(before, independent, alternative, coverage, scope, nominations):
    endorsements = Counter()
    for ordering in (independent, alternative, coverage, scope, nominations):
        endorsements.update(ordering[:8])
    changes = []
    ranking = list(before)
    finalists = (key for key, votes in endorsements.most_common() if key not in before and votes >= 3)
    weakest = sorted(reversed(before[5:]), key=lambda key: endorsements[key])
    for new in finalists:
        if not weakest or len(changes) == 2:
            break
        old = weakest[0]
        if endorsements[new] - endorsements[old] < 2:
            continue
        weakest.pop(0)
        ranking[ranking.index(old)] = new
        changes.append({"added": new, "removed": old, "new_votes": endorsements[new],
                        "old_votes": endorsements[old]})
    return ranking, changes


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memory import DIRECT_STOP, frame, plan, project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval

    helpers = read_module("agreement_helpers", ROOT / "selector-panels/run.py")
    metrics = read_module("agreement_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("agreement_data", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    cover = helpers.keyed(args.coverage / "rankings.jsonl")
    past = helpers.keyed(ROOT / "selector-set/rankings.jsonl")
    panels = helpers.keyed(ROOT / "selector-panels/rankings.jsonl")
    scope = helpers.keyed(ROOT / "selector-scope/rankings.jsonl")
    graded = helpers.keyed(ROOT / "selector-scope/challenge-rankings.jsonl")
    sources = read_module("agreement_sources", ROOT / "event-links/links.py").load_sources(args.data)
    tags = helpers.keyed(ROOT / "event-hierarchy/source-tags.jsonl")
    frames = {}
    traces = []
    packets = []
    qs = splits["Hard"]
    for q in qs:
        qid, query = q["id"], q["question"]
        before = cover[qid]["previous"]
        independent = graded[qid]["independent"]
        previous_set = past[qid]["after"]
        geographic = cover[qid]["coverage"] if cover[qid]["decisions"] else []
        local = scope[qid]["scope_cross"]
        nominated = sorted(panels[qid]["nominations"], key=lambda item: -item["priority"])
        nomination_top10 = [item["id"] for item in nominated if item["priority"] == 3][:10]
        full = consensus([before, independent, previous_set, geographic, local, nomination_top10],
                         [5, 3, 2, 4, 1, 2])
        guarded, changes = promote(before, independent, previous_set, geographic, local, nomination_top10)
        contextual, route = use_question_scope(query, before, geographic, full)
        trace = dict(id=qid, previous=before, consensus=full, guarded=guarded,
                     contextual=contextual, contextual_route=route, changes=changes)
        evidence = {}
        scope_type = plan({"question": query, "qtype": ""})
        query_tokens = set(terms(query)) - DIRECT_STOP
        for arm in ("consensus", "guarded", "contextual"):
            chosen = trace[arm]
            assert len(chosen) == len(set(chosen)) == 10 and set(chosen) <= records.keys()
            excerpts = []
            for key in chosen:
                if key in sources and key in tags:
                    if key not in frames:
                        frames[key] = frame(sources[key])
                    content, spans = project(sources[key], tags[key], frames[key], scope_type, query_tokens, 3050)
                else:
                    content, spans = helpers.clip(records[key]["text"], 3050), []
                excerpts.append(dict(id=key, text=content, source_spans=spans))
            assert sum(len(item["text"].encode()) for item in excerpts) <= BUDGET
            evidence[arm] = excerpts
        traces.append(trace)
        packets.append(dict(id=qid, question=query, evidence=evidence))
    arms = ("previous", "consensus", "guarded", "contextual")
    report = dict(n=len(qs), data_sha256=hashes, evidence_budget_bytes=BUDGET, model_api_tokens=0,
                  metrics=metrics.measure(qs, traces, arms, comprehensive_eval),
                  changed_questions={arm: sum(set(row["previous"]) != set(row[arm]) for row in traces) for arm in arms})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "consensus-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in (("consensus-rankings", traces), ("consensus-reader-packets", packets)):
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("release", "data", "official", "coverage", "output"):
        parser.add_argument(arg, type=Path)
    main(parser.parse_args())
