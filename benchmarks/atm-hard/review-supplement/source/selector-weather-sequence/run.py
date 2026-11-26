"""Identify a same-day outdoor weather event from the captured sequence."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

WEATHER = re.compile(r"\b(?:weather|hail(?:storm|stones)?|storm|rain|snow|sleet|thunder)\b", re.I)
CHANGE = re.compile(r"\b(?:suddenly|started|turned|became|approach\w*|before|after|onset)\b", re.I)
SINGLE_DAY = re.compile(r"\b(?:a day|one day|that day|the day|that afternoon|that morning)\b", re.I)
VIDEOS = re.compile(r"\b(?:videos?|clips?|recordings?)\b", re.I)
ONSET = re.compile(r"\b(?:storm clouds?|stormy sk(?:y|ies)|ominous (?:clouds?|sk(?:y|ies))|"
                   r"dark clouds?|thunderhead\w*|approaching storm)\b", re.I)
PRECIPITATION = re.compile(r"\b(?:hail(?:storm|stones)?|rainstorm|downpour|snowstorm|"
                           r"(?:heavy|falling) (?:rain|snow|sleet))\b", re.I)
AFTERMATH = re.compile(r"\b(?:wet|soaked|flooded|rain.slicked|puddles?)\b", re.I)
ARMS = ("reference", "day_only", "phase_tight", "phase_broad")


def applies(query):
    return all(pattern.search(query) for pattern in (WEATHER, CHANGE, SINGLE_DAY, VIDEOS))


def videos(data, geography):
    result = {}
    for row in json.loads((data / "video_batch_results.json").read_bytes()):
        loc = geography.extract_capture(row, "video")
        if not loc:
            continue
        try:
            when = datetime.fromisoformat(row["timestamp"]).replace(tzinfo=None)
        except ValueError:
            continue
        caption = "\n".join(row.get(field) or "" for field in ("short_caption", "caption"))
        result[loc["id"]] = dict(when=when, place=loc["city"], country=loc["country"],
                                 onset=bool(ONSET.search(caption)),
                                 precipitation=bool(PRECIPITATION.search(caption)),
                                 aftermath=bool(AFTERMATH.search(caption)))
    return result


def nominate(records, prior, scores):
    days = defaultdict(list)
    for key, video in records.items():
        days[(video["country"], video["place"], video["when"].date())].append(key)
    seed_rank = {key: rank for rank, key in enumerate(prior)}
    episodes = []
    for bucket, keys in days.items():
        anchors = [key for key in keys if key in seed_rank and
                   (records[key]["onset"] or records[key]["precipitation"])]
        if not anchors:
            continue
        support = sum((10 - seed_rank[key]) / 10 for key in anchors)
        topic = max(scores[key] for key in anchors)
        centers = [records[key]["when"] for key in anchors]
        onset_clips = sum(records[key]["onset"] for key in anchors)
        after = [key for key in keys if records[key]["aftermath"]
                 and timedelta(0) < records[key]["when"] - max(centers) <= timedelta(minutes=45)]
        # An onset followed by a visibly wet scene distinguishes a changing event
        # from other days with topically relevant but static outdoor recordings.
        progression = onset_clips > 0 and bool(after)
        episodes.append(dict(bucket=bucket, keys=keys, anchors=anchors, aftermath=after,
                             topic=topic, strength=support + 0.15*topic,
                             phase_strength=support + 0.15*topic + 0.75*progression,
                             progression=progression))
    return episodes


def order_episode(episode, records, scores, radius):
    seed_times = [records[key]["when"] for key in episode["anchors"]]
    def distance(key):
        return min(abs(records[key]["when"] - time) for time in seed_times)
    nearby = [key for key in episode["keys"] if distance(key) <= timedelta(minutes=radius)]
    return sorted(nearby, key=lambda key: (
        key not in episode["anchors"],
        key not in episode["aftermath"],
        -(scores[key] - 0.001 * distance(key).total_seconds() / 60),
        key))


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()),
                    str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("weather_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    previous = read_module("weather_files", ROOT / "selector-source-ranking/run.py")
    geo = read_module("weather_geo", ROOT / "selector-geography/build.py")
    metric = read_module("weather_eval", ROOT / "selector-corpus/run.py")
    sources = read_module("weather_sources", ROOT / "event-links/links.py").load_sources(args.data)
    tags = previous.read_rows("event-hierarchy/source-tags.jsonl")
    base = previous.read_rows("selector-facet-shortlist/model-rankings.jsonl")
    catalogue = videos(args.data, geo)
    questions = splits["Hard"]
    routed = {q["id"] for q in questions if applies(q["question"])}
    score = defaultdict(dict)
    with gzip.open(ROOT / "selector-proof/all-scores.jsonl.gz", "rt") as stream:
        for qid, key, topic, _, direct, *_ in map(json.loads, stream):
            if qid in routed and key in catalogue:
                score[qid][key] = 0.35 * topic + 0.65 * direct

    args.output.mkdir(parents=True, exist_ok=True)
    rows, packets, frames, diagnostics = [], [], {}, []
    for q in questions:
        qid, query = q["id"], q["question"]
        prior = base[qid]["coverage_guard"]
        row = dict(id=qid, **{arm:prior for arm in ARMS})
        episodes = nominate(catalogue, prior, score[qid]) if qid in routed else []
        if episodes:
            flat = min(episodes, key=lambda ep:(-ep["strength"], ep["bucket"]))
            timeline = min(episodes, key=lambda ep:(-ep["phase_strength"], ep["bucket"]))
            candidates = {}
            for arm, episode, radius in (("day_only", flat, 75), ("phase_tight", timeline, 45),
                                         ("phase_broad", timeline, 75)):
                if arm != "day_only" and not episode["progression"]:
                    candidates[arm] = []
                    continue
                selected = order_episode(episode, catalogue, score[qid], radius)
                candidates[arm] = selected
                row[arm] = list(dict.fromkeys([*selected, *prior]))[:10]
            row["candidate_sources"] = candidates
            row["chosen_day"] = {arm:[*episode["bucket"][:2], episode["bucket"][2].isoformat()]
                                 for arm,episode in (("day_only",flat),("phase",timeline))}
            row["episode_candidates"] = [
                dict(place=e["bucket"][1], date=e["bucket"][2].isoformat(), anchors=e["anchors"],
                     aftermath=e["aftermath"], scores=[e["strength"],e["phase_strength"]])
                for e in episodes]
            gold = set(q["evidence_ids"])
            diagnostics.append(dict(id=qid, gold=len(gold), days=len(episodes),
                hit={arm:len(gold & set(row[arm])) for arm in ARMS},
                candidate_gold={arm:len(gold & set(ids)) for arm,ids in candidates.items()}))

        wanted = plan({"question":query,"qtype":""})
        tokens = set(terms(query)) - DIRECT_STOP
        evidence = {}
        for arm in ARMS:
            assert len(row[arm]) == len(set(row[arm])) == 10 and set(row[arm]) <= records.keys()
            docs = []
            for doc in row[arm]:
                if doc in sources and doc in tags:
                    if doc not in frames:
                        frames[doc] = frame(sources[doc])
                    body, spans = project(sources[doc], tags[doc], frames[doc], wanted, tokens, 3050)
                else:
                    body, spans = metric.utf8_prefix(records[doc]["text"], 3050), []
                docs.append(dict(id=doc, text=body, source_spans=spans))
            assert sum(len(x["text"].encode()) for x in docs)<=BUDGET
            evidence[arm] = docs
        rows.append(row)
        packets.append(dict(id=qid, question=query, evidence=evidence))
    for name, items in (("rankings", rows), ("reader-packets", packets)):
        with (args.output / f"{name}.jsonl").open("w") as file:
            for item in items:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    output = dict(n=len(rows), data_sha256=hashes, videos_with_capture_metadata=len(catalogue),
                  routed_questions=len(routed), diagnostics=diagnostics,
                  budget=BUDGET, new_model_api_tokens=0,
                  metrics=metric.measure(questions,rows,ARMS,comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","output"):
        p.add_argument(name,type=Path)
    main(p.parse_args())
