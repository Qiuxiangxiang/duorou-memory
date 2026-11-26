"""Shortlist place evidence independently by answer place, capture date, and trip segment."""
import argparse
from collections import Counter, defaultdict
from datetime import timedelta
from itertools import zip_longest
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

LIMIT = 30
K = 10
ARMS = ("reference", "flat", "facets", "guarded_facets")


def interleave(routes, limit):
    return list(dict.fromkeys(key for tier in zip_longest(*routes)
                              for key in tier if key is not None))[:limit]


def date_episodes(active, groups):
    by_group = defaultdict(list)
    for doc in active:
        by_group[groups[doc]].append(doc)
    episode = {}
    for group, keys in by_group.items():
        previous = start = None
        for key in sorted(keys, key=lambda key: (active[key]["when"], key)):
            day = active[key]["day"]
            if previous is None or day - previous > timedelta(days=4):
                start = day
            episode[key] = group, start
            previous = day
    return episode


def stratified_candidates(active, group, relevance, limit):
    by_place = defaultdict(list)
    by_day = defaultdict(list)
    by_episode = defaultdict(list)
    episodes = date_episodes(active, group)

    def order(keys, bonus=None):
        return sorted(keys, key=lambda key: (-(relevance[key] + (bonus(key) if bonus else 0)), key))

    for key in active:
        by_place[group[key]].append(key)
        by_day[group[key], active[key]["day"]].append(key)
        by_episode[episodes[key]].append(key)

    flat = order(active)
    # A place gets a shortlist seat through its strongest observed source.
    place_reps = [order(keys)[0] for keys in by_place.values()]
    date_reps = [order(keys, lambda key: 0.08 * active[key]["explicit"])[0] for keys in by_day.values()]
    capture_routes = []
    for keys in by_episode.values():
        # End dates can be grounded by any record from those days. Use exact dates
        # instead of promoting an arbitrarily chosen first or last photo.
        first = min(active[key]["day"] for key in keys)
        last = max(active[key]["day"] for key in keys)
        capture_routes.append(order([key for key in keys if active[key]["day"] in {first, last}],
                                     lambda key: 0.12 * active[key]["transition"])[0])
    transitions = [key for key in active if active[key]["transition"] and active[key]["boundary"]]
    lanes = dict(topic=flat, place=order(place_reps), date=order(date_reps),
                 episode=order(capture_routes), travel=order(transitions))
    selected = interleave(list(lanes.values()), limit)
    return flat[:limit], selected, lanes


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()),
                    str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    visit = read_module("facet_short_visit", ROOT / "selector-visit-atlas/run.py")
    geo = read_module("facet_short_geo", ROOT / "selector-geography/build.py")
    files = read_module("facet_short_files", ROOT / "selector-source-ranking/run.py")
    metrics = read_module("facet_short_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("facet_short_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    sources = read_module("facet_short_sources", ROOT / "event-links/links.py").load_sources(args.data)
    tags = files.read_rows("event-hierarchy/source-tags.jsonl")
    prior = files.read_rows("selector-visit-atlas/rankings.jsonl")
    geography = files.read_rows("selector-geography/pools.jsonl")
    scope = files.read_rows("selector-scope/rankings.jsonl")
    atlas, seen_dates = visit.load_atlas(args.data, geo)
    scores = defaultdict(dict)
    with gzip.open(ROOT / "selector-proof/all-scores.jsonl.gz", "rt") as stream:
        for qid, doc, topic, _, direct, *_ in map(json.loads, stream):
            scores[qid][doc] = 0.35 * topic + 0.65 * direct

    args.output.mkdir(parents=True, exist_ok=True)
    rows, diagnostics, frames = [], [], {}
    with ((args.output / "rankings.jsonl").open("w") as rankings,
          (args.output / "reader-packets.jsonl").open("w") as packets):
        for question in splits["Hard"]:
            qid, query = question["id"], question["question"]
            reference = prior[qid]["guarded_grounded"]
            row = dict(id=qid, **{arm:reference for arm in ARMS})
            route = geography[qid]["route"]
            query_type = visit.PLACE_QUERY.search(query)
            if query_type and route:
                years = scope[qid]["scope_profile"]["years"]
                active = {key:item for key,item in visit.active_captures(query, atlas, route, years, reference).items()
                          if not item["airborne"]}
                by_country = query_type[1].casefold() == "countries"
                dated = bool(visit.DATES.search(query))
                groups = {key:(item["country"] if by_country else item["city"]) for key,item in active.items()}
                values = {key:(place, active[key]["day"] if dated else None) for key,place in groups.items()}
                relevance = {key: scores[qid][key] + .07 * item["explicit"] + .11 * item["transition"]
                             for key,item in active.items()}
                flat, facets, lanes = stratified_candidates(active, groups, relevance, LIMIT)
                for name, shortlist in (("flat", flat), ("facets", facets)):
                    ranked = visit.coverage(shortlist, relevance, values, 0.30)
                    row[name] = visit.adapter_union(ranked, reference)[:K]
                confidence = {key:(item["explicit"] or len(seen_dates[item["country"], item["city"]]) > 1)
                              for key,item in active.items()}
                guarded_order = visit.coverage(facets, relevance, values, 0.30, LIMIT)
                row["guarded_facets"] = visit.guarded_swaps(reference, guarded_order, active, values, confidence)
                row["shortlists"] = dict(flat=flat, facets=facets)
                row["nominated_by"] = {key:[lane for lane,ids in lanes.items() if key in ids[:LIMIT]]
                                       for key in facets}
                row["place_values"] = {key:[values[key][0], str(values[key][1] or "")]
                                       for key in visit.adapter_union(flat, facets, reference) if key in values}
                gold = set(question["evidence_ids"])
                diagnostics.append(dict(id=qid, gold=len(gold), in_scope=len(gold & active.keys()),
                    flat_shortlist=len(gold & set(flat)), facet_shortlist=len(gold & set(facets)),
                    flat_with_baseline=len(gold & set(flat + reference)),
                    facets_with_baseline=len(gold & set(facets + reference)),
                    unique_places_flat=len({groups[key] for key in flat}),
                    unique_places_facets=len({groups[key] for key in facets}),
                    final={arm:len(gold & set(row[arm])) for arm in ARMS}))

            requested = plan({"question": query, "qtype": ""})
            tokens = set(terms(query)) - DIRECT_STOP
            evidence = {}
            for arm in ARMS:
                assert len(row[arm]) == len(set(row[arm])) == K and set(row[arm]) <= records.keys()
                docs = []
                for key in row[arm]:
                    if key in sources and key in tags:
                        if key not in frames:
                            frames[key] = frame(sources[key])
                        body, spans = project(sources[key], tags[key], frames[key], requested, tokens, 3050)
                    else:
                        body, spans = metrics.utf8_prefix(records[key]["text"], 3050), []
                    docs.append(dict(id=key, text=body, source_spans=spans))
                assert sum(len(doc["text"].encode()) for doc in docs) <= BUDGET
                evidence[arm] = docs
            rankings.write(json.dumps(row, ensure_ascii=False) + "\n")
            packets.write(json.dumps(dict(id=qid, question=query, evidence=evidence), ensure_ascii=False) + "\n")
            rows.append(row)

    summary = dict(n=len(rows), data_sha256=hashes, shortlist_limit=LIMIT, budget=BUDGET,
                  new_model_api_tokens=0, diagnostics=diagnostics,
                  metrics=metrics.measure(splits["Hard"],rows,ARMS,comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "official", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
