"""Compare per-source ranking with budgeted coverage of supported answer values."""
import argparse
from collections import defaultdict
from datetime import timedelta
import gzip
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

K = 10
COUNTRIES = re.compile(r"\bcountries\b", re.I)
CITIES = re.compile(r"\bcities\b", re.I)
VISITS = re.compile(r"\bhow many (?:distinct )?(?:times|visits|trips)\b", re.I)
DATED = re.compile(r"\b(?:dates?|nights?|days?|visits?)\b", re.I)
INTERVAL = re.compile(r"\b(?:specific dates?|how many (?:days|nights))\b", re.I)
DAY = re.compile(r"\b(20\d\d)\s(0[1-9]|1[0-2])\s(0[1-9]|[12]\d|3[01])\b")


def normalized_key(value):
    return " ".join(re.findall(r"[^\W_]+", value.casefold()))


def capture_episodes(captures):
    """Group adjacent observed days of a city into a provisional visit."""
    by_place = defaultdict(list)
    for source in captures.values():
        by_place[source["country"], source["city"]].append(source)
    episodes = {}
    for (country, city), records in by_place.items():
        start = previous = None
        for record in sorted(records, key=lambda row: (row["day"], row["id"])):
            if previous is None or record["day"] - previous > timedelta(days=4):
                start = record["day"]
            previous = record["day"]
            episodes[record["id"]] = f"{city}, {country}, visit starting {start.isoformat()}"
    return episodes


def claim_index(qid, query, available, texts, captures, inputs, proof, episodes=None):
    """Build answer groups from source-local claims, preserving each original ID."""
    providers = defaultdict(dict)

    def add(doc, item, quote, strength, origin):
        if strength < 2 or doc not in available or not proof.valid_quote(quote, texts[doc]):
            return
        key = normalized_key(item)
        if not key:
            return
        capture = captures.get(doc)
        if capture and episodes is not None and (VISITS.search(query) or INTERVAL.search(query)):
            key = episodes[doc]
        elif capture and COUNTRIES.search(query):
            key = capture["country"]
        elif capture and CITIES.search(query):
            key = capture["city"]
        elif not (VISITS.search(query) or DATED.search(query)):
            key = DAY.sub("", key).strip()
        if not key or len(key) < 3:
            return
        by_source = providers[key].setdefault(doc, {})
        by_source[origin] = max(by_source.get(origin, 0), strength)

    for name, field in (("witness", "claims"), ("review", "newly_quoted")):
        for doc, rows in inputs[name][qid][field].items():
            for row in rows:
                add(doc, row["item"], row["quote"], row["strength"] - 1, "witness")
    for row in inputs["partial"][qid]["graded"]:
        add(row["id"], row["item"], row["quote"], row["score"], "partial")
    for row in inputs["answer"][qid]["partial_statements"]:
        strength = {"context": 1, "partial": 2, "direct": 3}[row["role"]]
        add(row["id"], row["item"], row["quote"], strength, "answer")

    groups = {}
    for item, supporters in providers.items():
        strong = {doc for doc, values in supporters.items() if max(values.values()) == 3}
        agreed = {doc for doc, values in supporters.items() if len(values) > 1}
        eligible = strong | agreed
        if eligible:
            # Provider sets can be identical when judges use aliases. Credit that set once.
            signature = tuple(sorted(eligible))
            group = groups.setdefault(signature, set())
            group.add(item)
    return groups


def solve(candidates, quality, groups, prior, bonus, max_new=None):
    """Maximize source quality and the number of distinct supported answer values."""
    positions = {doc: pos for pos, doc in enumerate(candidates)}
    source_count = len(candidates)
    variables = source_count + len(groups)
    matrix = lil_matrix((1 + len(groups) + int(max_new is not None), variables))
    upper = np.zeros(matrix.shape[0])
    lower = np.full(matrix.shape[0], -np.inf)
    matrix[0, :source_count] = 1
    upper[0] = lower[0] = K
    # Multiple aliases with overlapping providers should not multiply single-source credit.
    incidence = defaultdict(int)
    for docs in groups:
        for doc in docs:
            incidence[doc] += 1
    for i, docs in enumerate(groups, 1):
        for doc in docs:
            matrix[i, positions[doc]] = -1
        matrix[i, source_count + i - 1] = 1
    if max_new is not None:
        for doc in set(candidates) - set(prior):
            matrix[-1, positions[doc]] = 1
        upper[-1] = max_new

    weights = [-quality[doc] - 1e-7 * (source_count - pos)
               for pos, doc in enumerate(candidates)]
    weights += [-bonus / max(incidence[doc] for doc in docs) for docs in groups]
    result = milp(weights, integrality=np.ones(variables), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"mip_rel_gap": 0.0, "time_limit": 10.0})
    if not result.success:
        raise RuntimeError(f"Source allocation failed: {result.message}")
    selected = [doc for i, doc in enumerate(candidates) if result.x[i] > .5]
    assert len(selected) == K
    return sorted(selected, key=lambda doc: (-quality[doc], doc))


def episode_delta(prior, flat, episodes):
    """Apply a single swap when it comes solely from changing visit grouping."""
    added = set(episodes) - set(flat)
    removed = set(flat) - set(episodes)
    if len(added) != 1 or len(removed) != 1:
        return prior
    add, remove = next(iter(added)), next(iter(removed))
    if remove not in prior or add in prior:
        return prior
    return [add if key == remove else key for key in prior]


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("matching_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    proof = read_module("matching_proof", ROOT / "selector-proof/run.py")
    helper = read_module("matching_files", ROOT / "selector-source-ranking/run.py")
    geo = read_module("matching_geo", ROOT / "selector-geography/build.py")
    metrics = read_module("matching_eval", ROOT / "selector-calibration/run.py")
    sources = read_module("matching_sources", ROOT / "event-links/links.py").load_sources(args.data)
    tags = helper.read_rows("event-hierarchy/source-tags.jsonl")
    previous = helper.read_rows("selector-proof/rankings.jsonl")
    grade_files = {name:helper.read_rows(path) for name, (path, _) in helper.RATING_FILES.items()
                   if name != "scope"}
    grade_files["answer"] = helper.read_rows("selector-answer-trace/rankings.jsonl")
    captures = {}
    for kind in ("image", "video"):
        for raw in json.loads((args.data / f"{kind}_batch_results.json").read_bytes()):
            capture = geo.extract_capture(raw, kind)
            if capture:
                captures[capture["id"]] = capture

    episodes = capture_episodes(captures)
    raw_text = {doc:proof.normalize(row["text"]) for doc,row in records.items()}
    by_query = defaultdict(dict)
    with gzip.open(ROOT / "selector-proof/all-scores.jsonl.gz", "rt") as stream:
        for qid, doc, _, _, direct, anchored, _, _ in map(json.loads, stream):
            by_query[qid][doc] = direct, anchored

    args.output.mkdir(parents=True, exist_ok=True)
    rows, frames, cases = [], {}, []
    arms = ("reference", "direct", "anchored", "scoped_anchored", "coverage",
            "guarded_coverage", "episode_coverage", "episode_delta")
    with ((args.output / "rankings.jsonl").open("w") as rankings,
          (args.output / "reader-packets.jsonl").open("w") as packets):
        for question in splits["Hard"]:
            qid, query = question["id"], question["question"]
            before = previous[qid]
            prior = before["reference"]
            scores = by_query[qid]
            shortlist = sorted(scores, key=lambda doc:(-scores[doc][0], doc))[:40]
            available = sorted(set(shortlist) | set(prior))
            quality = {doc:scores[doc][1] for doc in available}
            reference = sorted(available, key=lambda doc:(-quality[doc], doc))[:10]
            assert reference == before["anchored"]
            # Exact record queries and scalar questions already have their own ranking.
            # Coverage is for requests that ask for multiple answer facts.
            request = proof.requirements(query)
            enumerated = question["qtype"] == "open_end" and not (
                request == {"media"} or not request and not VISITS.search(query))
            groups = claim_index(qid, query, available, raw_text, captures, grade_files, proof) if enumerated else {}
            coverage = solve(available, quality, groups, prior, 0.46) if groups else prior
            guarded = solve(available, quality, groups, prior, 0.46, max_new=2) if groups else prior
            episode_groups = (claim_index(qid, query, available, raw_text, captures, grade_files, proof, episodes)
                              if groups and (VISITS.search(query) or INTERVAL.search(query)) else groups)
            episode_coverage = solve(available, quality, episode_groups, prior, 0.46) if episode_groups else prior
            row = dict(id=qid, reference=prior, direct=before["direct"], anchored=before["anchored"],
                       scoped_anchored=reference if groups else prior,
                       coverage=coverage, guarded_coverage=guarded, episode_coverage=episode_coverage,
                       episode_delta=episode_delta(prior, coverage, episode_coverage),
                       candidates=available,
                       claim_groups=[dict(ids=list(ids), labels=sorted(labels)) for ids,labels in groups.items()],
                       episode_claim_groups=[dict(ids=list(ids), labels=sorted(labels))
                                             for ids,labels in episode_groups.items()])
            plan_for_reader = plan({"question": query, "qtype": ""})
            tokens = set(terms(query)) - DIRECT_STOP
            evidence = {}
            for arm in arms:
                ids = row[arm]
                assert len(ids) == len(set(ids)) == K and set(ids) <= records.keys()
                packet = []
                for doc in ids:
                    if doc in sources and doc in tags:
                        if doc not in frames:
                            frames[doc] = frame(sources[doc])
                        body, spans = project(sources[doc], tags[doc], frames[doc], plan_for_reader, tokens, 3050)
                    else:
                        body = records[doc]["text"].encode()[:3050].decode(errors="ignore")
                        spans = []
                    packet.append(dict(id=doc, text=body, source_spans=spans))
                assert sum(len(doc["text"].encode()) for doc in packet) <= BUDGET
                evidence[arm] = packet
            rankings.write(json.dumps(row, ensure_ascii=False) + "\n")
            packets.write(json.dumps(dict(id=qid, question=query, evidence=evidence), ensure_ascii=False) + "\n")
            rows.append(row)
            gold = set(question["evidence_ids"])
            cases.append(dict(id=qid, gold=len(gold), candidate_gold=len(gold & set(available)),
                              groups=len(groups), **{arm: len(gold & set(row[arm])) for arm in arms}))

    report = dict(n=len(rows), data_sha256=hashes, budget=BUDGET, new_model_api_tokens=0,
                  activated=sum(case["groups"] > 0 for case in cases),
                  candidate_pairs=sum(len(row["candidates"]) for row in rows),
                  capture_episodes=len(set(episodes.values())),
                  cases=cases,
                  metrics=metrics.record_metrics(splits["Hard"],rows,arms,comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: val for key, val in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "official", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
