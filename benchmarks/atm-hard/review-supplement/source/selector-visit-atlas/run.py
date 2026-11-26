"""Rank original capture evidence for place inventories and trip transitions."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import re
from statistics import median, mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

PLACE_QUERY = re.compile(r"\b(cities|countries)\b", re.I)
TRIP = re.compile(r"\btrip\b", re.I)
DATES = re.compile(r"\b(?:specific dates?|on what dates?)\b", re.I)
AIRCRAFT = re.compile(r"\b(?:aircraft|airplane|aeroplane|plane)\b", re.I)
ALOFT = re.compile(r"\b(?:looking down at|aerial view|in[- ]flight|city below|landscape below|"
                   r"flying over|from high above|view from above|above the clouds)\b", re.I)
ARMS = ("reference", "scope_only", "transition", "guarded", "grounded", "guarded_grounded")


def load_atlas(data, geo):
    atlas = {}
    for kind in ("image", "video"):
        for entry in json.loads((data / f"{kind}_batch_results.json").read_bytes()):
            source = geo.extract_capture(entry, kind)
            if not source:
                continue
            try:
                when = datetime.fromisoformat(entry["timestamp"]).replace(tzinfo=None)
            except ValueError:
                continue
            city = source["city"]
            caption = geo.normalize(entry.get("short_caption") or "")
            description = " ".join([entry.get("short_caption") or "", entry.get("caption") or ""])
            source.update(when=when, explicit=f" {city} " in f" {caption} ",
                          airborne=bool(AIRCRAFT.search(description) and ALOFT.search(description)),
                          transition=False, international=False)
            atlas[source["id"]] = source
    ordered = sorted(atlas.values(), key=lambda item: (item["when"], item["id"]))
    for before, after in zip(ordered, ordered[1:]):
        if after["when"] - before["when"] > timedelta(days=3):
            continue
        if (before["country"], before["city"]) != (after["country"], after["city"]):
            for source in (before, after):
                source["transition"] = True
                source["international"] |= before["country"] != after["country"]
    dates_by_place = defaultdict(set)
    for source in atlas.values():
        dates_by_place[source["country"], source["city"]].add(source["day"])
    return atlas, dates_by_place


def active_captures(query, atlas, route, years, prior):
    countries = set(route["countries"])
    active = {key: row for key, row in atlas.items()
              if row["country"] in countries and (not years or row["day"].year in years)}
    anchors = [row["day"].toordinal() for key, row in active.items() if key in prior]
    if TRIP.search(query) and years and anchors:
        center = median(anchors)
        active = {key: row for key, row in active.items()
                  if abs(row["day"].toordinal() - center) <= 30}
        if not active:
            return {}
        first = min(row["when"] for row in active.values())
        # Keep an adjacent outside-region transport record when it can prove departure.
        active.update({key: row for key, row in atlas.items() if row["country"] not in countries
                       and row["international"] and row["boundary"]
                       and timedelta(0) <= first - row["when"] <= timedelta(hours=48)})
    return active


def coverage(ids, scores, keys, bonus, limit=10):
    covered = Counter()
    ranked = []
    rest = set(ids)
    while rest and len(ranked) < limit:
        doc = min(rest, key=lambda key: (-(scores[key] + bonus / (1 + 3 * covered[keys[key]])), key))
        ranked.append(doc)
        rest.remove(doc)
        covered[keys[doc]] += 1
    return ranked


def guarded_swaps(prior, ranked, active, group, presence):
    # Keep the best current copy for each documented value and preserve other proof types.
    selected = prior[:]
    for challenger in ranked:
        old_groups = Counter(group[key] for key in selected if key in group)
        if challenger in selected or old_groups[group[challenger]]:
            continue
        if not presence[challenger]:
            continue
        remove = next((key for key in reversed(selected)
                       if key in active and old_groups[group[key]] > 1), None)
        if remove is None:
            break
        selected[selected.index(remove)] = challenger
        if len(set(selected) - set(prior)) == 2:
            break
    return selected


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()),
                    str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    geo = read_module("visit_atlas_geo", ROOT / "selector-geography/build.py")
    adapter = read_module("visit_atlas_files", ROOT / "selector-source-ranking/run.py")
    metric = read_module("visit_atlas_eval", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("visit_atlas_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    sources = read_module("visit_atlas_sources", ROOT / "event-links/links.py").load_sources(args.data)
    tags = adapter.read_rows("event-hierarchy/source-tags.jsonl")
    base = adapter.read_rows("selector-matching/rankings.jsonl")
    geographic = adapter.read_rows("selector-geography/pools.jsonl")
    scope = adapter.read_rows("selector-scope/rankings.jsonl")
    atlas, dates_by_place = load_atlas(args.data, geo)
    scores = defaultdict(dict)
    with gzip.open(ROOT / "selector-proof/all-scores.jsonl.gz", "rt") as stream:
        for qid, doc, topic, _, direct, *_ in map(json.loads, stream):
            scores[qid][doc] = topic, direct

    args.output.mkdir(parents=True, exist_ok=True)
    traces, frames, cases = [], {}, []
    with ((args.output / "rankings.jsonl").open("w") as rankings,
          (args.output / "reader-packets.jsonl").open("w") as packets):
        for question in splits["Hard"]:
            qid, query = question["id"], question["question"]
            prior = base[qid]["episode_delta"]
            row = {arm:prior for arm in ARMS}
            matched = PLACE_QUERY.search(query)
            route = geographic[qid]["route"]
            active = active_captures(query, atlas, route, scope[qid]["scope_profile"]["years"], prior) if matched and route else {}
            details = {}
            if active:
                by_country = matched[1].casefold() == "countries"
                date_values = bool(DATES.search(query))
                group = {key:(item["country"] if by_country else item["city"])
                         for key,item in active.items()}
                result_group = {key:(label, active[key]["day"].isoformat() if date_values else "")
                                for key,label in group.items()}
                relevance = {key:0.35 * scores[qid][key][0] + 0.65 * scores[qid][key][1]
                             for key in active}
                confidence = {key:source["explicit"] or len(dates_by_place[source["country"],source["city"]]) > 1
                              for key,source in active.items()}
                with_transition = {key:value + .11 * active[key]["transition"]
                                   + .08 * active[key]["explicit"]
                                   + .09 * bool(TRIP.search(query) and active[key]["boundary"])
                                   for key,value in relevance.items()}
                simple = coverage(active, relevance, result_group, .30, 30)
                shifted = coverage(active, with_transition, result_group, .30, 30)
                ground = coverage((key for key in active if not active[key]["airborne"]),
                                  with_transition, result_group, .30, 30)
                row["scope_only"] = adapter_union(simple, prior)[:10]
                row["transition"] = adapter_union(shifted, prior)[:10]
                row["guarded"] = guarded_swaps(prior, shifted, active, result_group, confidence)
                row["grounded"] = adapter_union(ground, prior)[:10]
                row["guarded_grounded"] = guarded_swaps(prior, ground, active, result_group, confidence)
                details = {key:dict(group=result_group[key], score=with_transition[key],
                                    transition=active[key]["transition"], explicit=active[key]["explicit"],
                                    airborne=active[key]["airborne"])
                           for key in adapter_union(simple, shifted, ground, prior) if key in active}
            row.update(id=qid, candidate_count=len(active), shortlisted=details)
            requested = plan({"question": query, "qtype": ""})
            tokens = set(terms(query)) - DIRECT_STOP
            evidence = {}
            for arm in ARMS:
                assert len(row[arm]) == len(set(row[arm])) == 10 and set(row[arm]) <= records.keys()
                documents = []
                for key in row[arm]:
                    if key in sources and key in tags:
                        if key not in frames:
                            frames[key] = frame(sources[key])
                        content, spans = project(sources[key], tags[key], frames[key], requested, tokens, 3050)
                    else:
                        content, spans = metric.utf8_prefix(records[key]["text"], 3050), []
                    documents.append(dict(id=key, text=content, source_spans=spans))
                assert sum(len(doc["text"].encode()) for doc in documents) <= BUDGET
                evidence[arm] = documents
            ranks_gold = set(question["evidence_ids"])
            if active:
                known_gold = ranks_gold.intersection(active)
                case = dict(id=qid, relevant=len(ranks_gold), active_gold=len(known_gold),
                            ground_gold=sum(not active[key]["airborne"] for key in known_gold),
                            short_gold=len(set(details)&ranks_gold),
                            **{arm:len(set(row[arm])&ranks_gold) for arm in ARMS})
                for arm in ("reference", "grounded", "guarded_grounded"):
                    values = {result_group[key] for key in row[arm]
                              if key in active and not active[key]["airborne"]}
                    case[arm+"_place_values"] = len(values)
                    case[arm+"_matching_gold_values"] = len(values & {result_group[key] for key in known_gold})
                cases.append(case)
            rankings.write(json.dumps(row, ensure_ascii=False) + "\n")
            packets.write(json.dumps(dict(id=qid, question=query, evidence=evidence), ensure_ascii=False) + "\n")
            traces.append(row)
    report = dict(n=len(traces), data_sha256=hashes, capture_sources=len(atlas),
                  possible_airborne=sum(source["airborne"] for source in atlas.values()),
                  directly_routed=len(cases), cases=cases, budget=BUDGET, new_model_api_tokens=0,
                  metrics=metric.measure(splits["Hard"],traces,ARMS,comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def adapter_union(*groups):
    return list(dict.fromkeys(doc for group in groups for doc in group))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "official", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
