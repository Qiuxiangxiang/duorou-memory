"""Test a compressed geographic shortlist with an identical evidence-aware model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from cloud import JUDGE
from evaluate import BUDGET, read_module
from run import load_inputs

PROMPT = """For each question, rank up to ten original records in order of evidence value.
The aim is to help a reader answer every distinct part of the question. Return the source ID
and a short exact contiguous quote for each chosen source. Use only the supplied cards.
A literal observed city or country and its capture date support presence. A travel document
can establish a route, but distinguish planned travel from observed presence. An aerial view
taken from an airplane does not show a ground visit. If dates are requested, a new city's
first or last observed date can matter more than another shot of an already supported day.
A terminal record can establish a real itinerary endpoint. For geographic inventories,
prioritize coverage and then the strongest evidence for each place. Use up to ten distinct
source IDs and no text outside the schema. Do not provide the answer."""
FIELDS = {"chosen": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["source", "quote"],
    "properties": {"source": {"type": "string"}, "quote": {"type": "string"}}}}}
ARMS = ("flat_model", "facets_model")


def extends_place_coverage(reference, challenger, place_values):
    prior = {tuple(place_values[key]) for key in reference if key in place_values}
    proposed = {tuple(place_values[key]) for key in challenger if key in place_values}
    old_places = {place for place, _ in prior}
    new_places = {place for place, _ in proposed}
    return bool(old_places) and old_places < new_places and len(proposed) >= len(prior)


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()),
                    str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame

    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("facet_model_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    files = read_module("facet_model_files", ROOT / "selector-source-ranking/run.py")
    metric = read_module("facet_model_metric", ROOT / "selector-corpus/run.py")
    panel = read_module("facet_model_panel", ROOT / "selector-panels/run.py")
    runner = read_module("facet_model_runner", ROOT / "event-graph-full/benchmark.py")
    sources = read_module("facet_model_sources", ROOT / "event-links/links.py").load_sources(args.data)
    tags = files.read_rows("event-hierarchy/source-tags.jsonl")
    with (args.output / "rankings.jsonl").open() as stream:
        previous = {row["id"]: row for row in map(json.loads, stream)}

    requests, cards, pools, frames = [], {}, {}, {}
    for q in splits["Hard"]:
        qid, query = q["id"], q["question"]
        row = previous[qid]
        if "shortlists" not in row:
            continue
        weights = {word: 1 for word in set(terms(query)) - DIRECT_STOP}
        for arm, short in zip(ARMS, ("flat", "facets"), strict=True):
            request_id = f"{qid}:{arm}"
            shortlist = row["shortlists"][short]
            pools[request_id] = shortlist
            shown = {}
            for key in shortlist:
                if key in sources and key in tags:
                    if key not in frames:
                        frames[key] = frame(sources[key])
                    shown[key] = panel.make_card(sources[key], tags[key], frames[key], weights, 850)
                else:
                    shown[key] = panel.clip(records[key]["text"], 850)
            cards[request_id] = shown
            requests.append(dict(id=request_id, question=query,
                                 candidates=[dict(id=key, evidence=shown[key]) for key in sorted(shown)]))
    outputs, usage = runner.gather(args.cache, "hard-facet-shortlist-rerank", JUDGE, PROMPT,
                                   requests, FIELDS, 1, 160000, 6)
    rows, packets = [], []
    accepted = dict.fromkeys(ARMS, 0)
    for q in splits["Hard"]:
        qid = q["id"]
        old = previous[qid]
        row = dict(id=qid, reference=old["reference"])
        evidence = {}
        for arm in ARMS:
            request_id = f"{qid}:{arm}"
            if request_id not in outputs:
                row[arm] = old["reference"]
                continue
            chosen = {}
            for proposal in outputs[request_id]["chosen"][:10]:
                doc, quote = proposal["source"], proposal["quote"].strip()
                if doc in cards[request_id] and len(quote) >= 8 and quote in cards[request_id][doc]:
                    chosen.setdefault(doc, quote)
            accepted[arm] += len(chosen)
            ranked = list(dict.fromkeys([*chosen, *pools[request_id]]))[:10]
            assert len(ranked) == len(set(ranked)) == 10
            row[arm] = ranked
            evidence[arm] = [dict(id=doc, text=cards[request_id][doc], quote=chosen.get(doc,""))
                             for doc in ranked]
            assert sum(len(doc["text"].encode()) for doc in evidence[arm]) <= BUDGET
        covered_more = extends_place_coverage(row["reference"], row["facets_model"],
                                             old.get("place_values", {}))
        row["coverage_guard"] = row["facets_model"] if covered_more else row["reference"]
        rows.append(row)
        if evidence:
            packets.append(dict(id=qid, question=q["question"], evidence=evidence))
    report = dict(n=len(rows), data_sha256=hashes, requests=len(requests), model=JUDGE, usage=usage,
                  valid_quotes=accepted,
                  changed_by_coverage_guard=sum(row["coverage_guard"] != row["reference"] for row in rows),
                  metrics=metric.measure(splits["Hard"], rows, ("reference", *ARMS, "coverage_guard"),
                                         comprehensive_eval))
    runner.write_rows(args.output / "model-rankings.jsonl", rows)
    runner.write_rows(args.output / "model-reader-packets-routed.jsonl", packets)
    (args.output / "model-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "official", "cache", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
