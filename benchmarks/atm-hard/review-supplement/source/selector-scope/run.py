"""Evaluate full-corpus relevance ranked with query scope and source fields."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import read_module
from run import load_inputs

corpus = read_module("corpus_relevance", ROOT / "selector-corpus/run.py")
BUDGET, BYTES_PER_SOURCE, K = corpus.BUDGET, corpus.BYTES_PER_SOURCE, corpus.K
descending, lexical_scores, measure = corpus.descending, corpus.lexical_scores, corpus.measure
relevance_view, utf8_prefix = corpus.relevance_view, corpus.utf8_prefix

YEAR = re.compile(r"\b20\d{2}\b")
PHOTO = re.compile(r"\b(?:photos?|images?)\b", re.I)
VIDEO = re.compile(r"\b(?:videos?|slow.motion|timelapse)\b", re.I)
MEDIA_QUESTION = re.compile(r"\b(?:photos?|images?|videos?|media|slides)\b", re.I)
ALL_IDS = re.compile(r"\b(?:ids?|photos?|videos?|images?|media)\b", re.I)
ROLE_PATTERNS = {
    "payment": re.compile(r"\b(?:paid|pay|spend|spent|cost|reimburs\w*)\b", re.I),
    "lodging": re.compile(r"\b(?:hotels?|accommodation|nights)\b", re.I),
    "attendance": re.compile(r"\b(?:attend\w*|went to|conference)\b", re.I),
    "visit": re.compile(r"\b(?:cit(?:y|ies)|countries|visit\w*|stayed)\b", re.I),
}
EVIDENCE_PATTERNS = {
    "payment": re.compile(r"\b(?:amount paid|payment received|paid in full|receipt|invoice|charged|debit)\b", re.I),
    "lodging": re.compile(r"\b(?:hotel|check.in|check.out|guest|reservation|accommodation)\b", re.I),
    "attendance": re.compile(r"\b(?:badge|conference program|registration confirmation|attended|session)\b", re.I),
    "visit": re.compile(r"\b(?:boarding pass|itinerary|departure|arrivals?|check.in|check.out|flight)\b", re.I),
}
STOP_SCOPE = {"home", "city", "grace", "central", "king", "conference", "paris time"}


def normalize_keys(text):
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def contains(text, phrase):
    return f" {phrase} " in f" {text} "


def active_modalities(text):
    if not MEDIA_QUESTION.search(text):
        return set()
    if PHOTO.search(text) and VIDEO.search(text):
        return {"image", "video"}
    if PHOTO.search(text):
        return {"image"}
    if VIDEO.search(text) and ALL_IDS.search(text):
        return {"video"}
    return {"image", "video"}


def rrf_signal(scores):
    order = descending(scores)
    active = order[scores[order] > 0]
    reciprocal = np.zeros(len(scores))
    reciprocal[active] = 60 / (60 + np.arange(1, len(active) + 1))
    return reciprocal


def read_base(path, positions, question_ids):
    scores = {key: np.zeros(len(positions)) for key in question_ids}
    seen = defaultdict(set)
    with gzip.open(path, "rt") as file:
        for line in file:
            key, _, source, _, _, _, cosine, _ = json.loads(line)
            seen[key].add(source)
            scores[key][positions[source]] = cosine
    assert set(seen) == set(question_ids) and all(len(group) == len(positions) for group in seen.values())
    return scores


def scope(text, known_places, europe, north_america):
    norm = normalize_keys(text)
    places = {key for key in known_places if contains(norm, key)}
    if contains(norm, "european") or contains(norm, "europe"):
        places |= europe
    if contains(norm, "north america"):
        places |= north_america
    years = {int(hit) for hit in YEAR.findall(text)}
    if len(years) == 2 and re.search(r"\b(?:from|between|past)\b", text, re.I):
        years = set(range(min(years), max(years) + 1))
    elif len(years) == 1 and re.search(r"\bpast two years\b", text, re.I):
        end = next(iter(years))
        years = {end - 2, end - 1, end}
    return {"places": sorted(places), "years": sorted(years),
            "modalities": sorted(active_modalities(text)),
            "roles": [role for role, pattern in ROLE_PATTERNS.items() if pattern.search(text)]}


def score_question(question, alternates, features, places, indices, positions, dense, terms,
                   europe, north_america):
    requirements = scope(question, places, europe, north_america)
    queries = (question, *alternates)
    views = []
    for name, index in indices.items():
        primary = rrf_signal(lexical_scores(index, question, positions, terms))
        expanded = np.maximum.reduce([
            rrf_signal(lexical_scores(index, query, positions, terms)) for query in queries])
        views.append((name, 0.35 * primary + 0.65 * expanded))
    view_scores = dict(views)
    topic = (0.42 * view_scores["full"] + 0.32 * view_scores["identity"] +
             0.16 * view_scores["ocr"] + 0.10 * rrf_signal(np.maximum(dense, 0)))
    factors = np.ones(len(features))
    geo = np.zeros(len(features))
    roles = np.zeros(len(features))
    asked_years = set(requirements["years"])
    modalities = set(requirements["modalities"])
    wanted_places = set(requirements["places"])
    travel = "visit" in requirements["roles"]
    for idx, (kind, locations, year, printed_years, source_roles) in enumerate(features):
        location_match = not wanted_places.isdisjoint(locations)
        if location_match:
            geo[idx] = 1
            factors[idx] += 0.34
        elif wanted_places and kind != "email" and locations:
            factors[idx] -= 0.14
        if asked_years:
            if year in asked_years or not asked_years.isdisjoint(printed_years):
                factors[idx] += 0.14
            else:
                factors[idx] -= 0.15
        if modalities:
            factors[idx] += 0.22 if kind in modalities else -0.45
        matched = set(requirements["roles"]) & source_roles
        if matched:
            roles[idx] = 1 if "payment" in matched else 0.6
    evidence_signal = 0.19 * topic * roles
    location_signal = (0.07 if travel else 0.035) * geo
    return topic, topic * factors + evidence_signal + location_signal, requirements


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from duorou.retrieval import BM25, terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    args.output.mkdir(parents=True, exist_ok=True)
    records, kinds, data_splits, hashes = load_inputs(args.data)
    pinned = read_module("pinned_public_data", ROOT / "iterative-search/search.py")
    assert hashes == pinned.PUBLIC_SHA256
    questions = data_splits["Hard"]
    old = {r["id"]: r for r in map(json.loads, (args.baseline / "rankings.jsonl").open())}
    rewrite_bytes = args.rewrites.read_bytes()
    rewrite_data = json.loads(rewrite_bytes)
    rewrites = {row["id"]: row["queries"] for row in rewrite_data["items"]}
    assert {q["id"] for q in questions} == old.keys() == rewrites.keys()
    reranker = read_module("scope_reranker", ROOT / "rerank/run.py")
    geographer = read_module("scope_geography", ROOT / "selector-geography/build.py")
    source_module = read_module("scope_sources", ROOT / "event-links/links.py")
    sources = source_module.load_sources(args.data)
    with (ROOT / "event-hierarchy/source-tags.jsonl").open() as stream:
        tags = {row["id"]: row for row in map(json.loads, stream)}
    captures = {}
    for kind in ("image", "video"):
        for obj in json.loads((args.data / f"{kind}_batch_results.json").read_bytes()):
            row = geographer.extract_capture(obj, kind)
            if row:
                captures[row["id"]] = row

    ids = sorted(records)
    positions = {key: index for index, key in enumerate(ids)}
    embedding_scores = read_base(args.baseline / "corpus-ranks.jsonl.gz", positions, old)
    full = {}
    identity = {}
    ocr = {}
    features = []
    known_places = set()
    for key in ids:
        row = records[key]["text"]
        source = sources.get(key)
        capture = captures.get(key)
        fields = source.fields if source else {}
        summary = fields.get("short_summary") or fields.get("short_caption") or ""
        subject = re.search(r"(?im)^subject:\s*([^\n]+)", fields.get("detail", ""))
        locations = {capture["city"], capture["country"]} if capture else set()
        locations = {value for value in locations if value not in STOP_SCOPE and len(value) > 3}
        known_places.update(locations)
        full[key] = row
        identity[key] = " ".join([summary, subject[1] if subject else "", *sorted(locations)])
        ocr[key] = fields.get("ocr_text", "")
        date_text = " ".join(tags.get(key, {}).get("other_time_text", {}).get("printed_dates", []))
        year = source.day.year if source else 0
        source_roles = {role for role, pattern in EVIDENCE_PATTERNS.items() if pattern.search(row)}
        features.append((kinds[key], locations, year, {int(n) for n in YEAR.findall(date_text)}, source_roles))
    indices = {name: BM25(view) for name, view in (("full", full), ("identity", identity), ("ocr", ocr))}
    cross = reranker.CrossEncoder(args.cross_model)
    traces = []
    timings = []
    shortlist_diagnostic = []
    frames = {}
    arms = ("expanded", "scope", "scope_cross")
    with (args.output / "rankings.jsonl").open("w") as rankings, \
            gzip.open(args.output / "all-scores.jsonl.gz", "wt") as all_scores, \
            (args.output / "reader-packets.jsonl").open("w") as packets:
        for number, question in enumerate(questions, 1):
            start = perf_counter()
            key, query = question["id"], question["question"]
            topic, scores, profile = score_question(
                query, rewrites[key], features, known_places, indices, positions, embedding_scores[key],
                terms, geographer.EUROPE, geographer.NORTH_AMERICA)
            order = descending(scores)
            finalists = order[:160]
            labels = [ids[pos] for pos in finalists]
            views = [relevance_view(full[label], reranker) for label in labels]
            logits = np.array(cross.score(query, views))
            cross_ranks = descending(logits)
            rerank_signal = np.zeros(len(finalists))
            rerank_signal[cross_ranks] = 60 / (60 + np.arange(1, len(finalists) + 1))
            final_scores = 0.56 * scores[finalists] + 0.44 * rerank_signal
            reranked = finalists[descending(final_scores)]
            trace = {"id": key, "previous": old[key]["previous"], "corpus_cross": old[key]["relevance_cross"],
                     "queries": rewrites[key], "scope_profile": profile}
            packet = {}
            requirements = plan({"question": query, "qtype": ""})
            tokens = set(terms(query)) - DIRECT_STOP
            for arm, order_by_arm in zip(arms, (descending(topic), order, reranked), strict=True):
                trace[arm] = [ids[pos] for pos in order_by_arm[:K]]
                assert len(set(trace[arm])) == K
                evidence = []
                for label in trace[arm]:
                    if label in sources and label in tags:
                        if label not in frames:
                            frames[label] = frame(sources[label])
                        content, spans = project(sources[label], tags[label], frames[label], requirements,
                                                 tokens, BYTES_PER_SOURCE)
                    else:
                        content, spans = utf8_prefix(full[label], BYTES_PER_SOURCE), []
                    evidence.append(dict(id=label, text=content, source_spans=spans))
                assert sum(len(item["text"].encode()) for item in evidence) <= BUDGET
                packet[arm] = evidence
            gold = set(question["evidence_ids"])
            shortlist_diagnostic.append((len(gold.intersection(labels)), len(gold)))
            raw_cross = dict(zip(labels, map(float, logits), strict=True))
            for rank, pos in enumerate(order, 1):
                label = ids[pos]
                line = [key, rank, label, float(topic[pos]), float(scores[pos]), raw_cross.get(label)]
                all_scores.write(json.dumps(line, separators=(",", ":")) + "\n")
            rankings.write(json.dumps(trace, ensure_ascii=False) + "\n")
            packets.write(json.dumps(dict(id=key, question=query, evidence=packet), ensure_ascii=False) + "\n")
            traces.append(trace)
            timings.append(perf_counter() - start)
            print(f"Hard scope {number}/{len(questions)}", flush=True)
    stats = dict(n=len(questions), data_sha256=hashes, scored_pairs=len(questions) * len(ids),
                 rewrites_sha256=hashlib.sha256(rewrite_bytes).hexdigest(),
                 rewrites_usage=rewrite_data["usage"], new_model_api_tokens=0,
                 cross_sha256=cross.hashes, evidence_budget_bytes=BUDGET,
                 metrics=measure(questions, traces, ("previous", "corpus_cross", *arms), comprehensive_eval),
                 candidate_top160_recall=mean(n/d for n, d in shortlist_diagnostic),
                 candidate_top160_gold=sum(n for n, _ in shortlist_diagnostic),
                 mean_seconds_per_question=mean(timings))
    (args.output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("release", "data", "official", "cross_model", "baseline", "rewrites", "output"):
        parser.add_argument(arg, type=Path)
    main(parser.parse_args())
