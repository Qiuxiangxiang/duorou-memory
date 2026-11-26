"""Separate topic relevance from source-local evidence that can answer the question."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

PAY = re.compile(r"\b(?:pay|paid|spent|spend|reimburs\w*|total cost)\b", re.I)
WHEN = re.compile(r"\b(?:when|date[s]?|what years?|specific years|how long|duration|how many (?:days|nights)|start and end)\b", re.I)
WHERE = re.compile(r"\b(?:cities|countries|which hotels?|which hotel was|which .*restaurant)\b", re.I)
MEDIA = re.compile(r"\b(?:ids?|photos?|images?|videos?|media|slides)\b", re.I)
MEDIA_REQUEST = re.compile(r"\b(?:find|show|list|recall|retrieve|report|which|collection)\b", re.I)
PAPER = re.compile(r"\b(?:papers|titles)\b", re.I)
CURRENCY = re.compile(r"(?:[$€£¥]\s*\d|\b(?:EUR|USD|GBP|CAD|CNY|JPY|CLP|RMB)\s*\d|\d[\d., ]*\s*(?:EUR|USD|GBP|CAD|CNY|JPY|CLP|RMB|yuan|euros?|pounds?|dollars?)\b)", re.I)
PAID = re.compile(r"\b(?:total paid|amount paid|paid in full|payment received|receipt|charged|refund\w*|transaction|card ending|contactless)\b", re.I)
TITLE = re.compile(r"\b(?:paper title|poster title|title of the paper|title:)\s*", re.I)
DATE = re.compile(r"\b(?:20\d\d[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b|(?:[01]?\d|2[0-3]):[0-5]\d|\d+\s+(?:minutes?|hours?|days?|nights?))", re.I)
PLACE = re.compile(r"\b(?:hotel|resort|hostel|restaurant|inn|suites|apartments)\b", re.I)
ARMS = ("topic", "judgments_only", "direct", "anchored")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def valid_quote(quote, literal):
    needle = normalize(quote)
    return len(needle) >= 9 and needle in literal


def requirements(question):
    wanted = {key for key, pattern in (("money", PAY), ("time", WHEN), ("place", WHERE),
                                       ("title", PAPER)) if pattern.search(question)}
    if MEDIA.search(question) and MEDIA_REQUEST.search(question):
        wanted.add("media")
    return wanted


def value_support(wanted, quote, tag):
    """Recognize an explicit requested value near the validated citation."""
    kind = tag.get("source_kind")
    observed = kind in ("image", "video")
    values = set()
    if "money" in wanted and CURRENCY.search(quote):
        values.add("charged" if PAID.search(quote) else "quoted_amount")
    if "time" in wanted and (DATE.search(quote) or (observed and DATE.search(tag.get("recorded_time", "")))):
        values.add("time")
    if "place" in wanted and (PLACE.search(quote) or (observed and tag.get("recorded_place"))):
        values.add("place")
    if "media" in wanted and observed:
        values.add("media")
    if "title" in wanted and TITLE.search(quote):
        values.add("title")
    return values


def judgments(qid, wanted, original, tags, inputs):
    grades = defaultdict(dict)
    evidence = defaultdict(set)

    def add(doc, grader, grade, quote):
        if doc not in original:
            return
        if quote and valid_quote(quote, original[doc]):
            found = value_support(wanted, quote, tags.get(doc, {}))
            if found:
                evidence[doc].update(found)
        elif grade >= 0.5:
            return
        grades[doc][grader] = max(grade, grades[doc].get(grader, 0))

    for row in inputs["scope"][qid]["ratings"]:
        add(row["source_id"], "scope", {0: 0, 1: .15, 2: .34, 3: .67, 4: 1}[row["score"]],
            row["quote"])
    for row in inputs["partial"][qid]["graded"]:
        add(row["id"], "partial", {0: 0, 1: .15, 2: .67, 3: 1}[row["score"]], row["quote"])
    for path, field in (("witness", "claims"), ("review", "newly_quoted")):
        for doc, rows in inputs[path][qid][field].items():
            for row in rows:
                add(doc, "witness", {2: .34, 3: .67, 4: 1}[row["strength"]], row["quote"])
    for row in inputs["answer"][qid]["partial_statements"]:
        add(row["id"], "answer", {"context": .15, "partial": .67, "direct": 1}[row["role"]], row["quote"])
    return grades, evidence


def support(topic, grades):
    if not grades:
        return 0.28 * topic
    # A verified partial answer remains valuable. Independent agreement contributes
    # more than a lone grade among contradictory judgments.
    values = list(grades.values())
    average = (sum(values) + 0.5 * topic) / (len(values) + 0.5)
    return 0.17 * topic + 0.36 * max(values) + 0.36 * average


def value_bonus(flags):
    return 0.11 * min(1, sum(0.5 if flag == "quoted_amount" else 1 for flag in flags))


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()),
                    str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    ranker = read_module("proof_base", ROOT / "selector-source-ranking/run.py")
    metrics = read_module("proof_metrics", ROOT / "selector-corpus/run.py")
    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("proof_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    sources = read_module("proof_sources", ROOT / "event-links/links.py").load_sources(args.data)
    tags = ranker.read_rows("event-hierarchy/source-tags.jsonl")
    reference = ranker.read_rows("selector-source-ranking/rankings.jsonl")
    inputs = {name:ranker.read_rows(path) for name, (path, _) in ranker.RATING_FILES.items()}
    inputs["answer"] = ranker.read_rows("selector-answer-trace/rankings.jsonl")
    original = {doc: normalize(row["text"]) for doc, row in records.items()}
    topic_by_q = defaultdict(dict)
    with gzip.open(ROOT / "selector-source-ranking/all-scores.jsonl.gz", "rt") as file:
        for row in map(json.loads, file):
            question, doc, _, retrieval, *_ = row
            topic_by_q[question][doc] = retrieval

    args.output.mkdir(parents=True, exist_ok=True)
    traces, frames = [], {}
    diagnostics = defaultdict(int)
    with ((args.output / "rankings.jsonl").open("w") as ranks,
          (args.output / "reader-packets.jsonl").open("w") as packets,
          gzip.open(args.output / "all-scores.jsonl.gz", "wt") as scores):
        for question in splits["Hard"]:
            qid, query = question["id"], question["question"]
            topic = topic_by_q[qid]
            assert topic.keys() == records.keys()
            wanted = requirements(query)
            grades, evidence = judgments(qid, wanted, original, tags, inputs)
            prior = reference[qid]["reference"]
            prior_scores = {doc: 1 / (1 + rank / 12) for rank, doc in enumerate(prior)}
            judgments_only = {doc:support(value, grades.get(doc, {})) for doc,value in topic.items()}
            direct = {doc:value + value_bonus(evidence.get(doc, ()))
                      for doc,value in judgments_only.items()}
            anchored = {doc:value + 0.57 * prior_scores.get(doc, 0) for doc,value in direct.items()}
            score_map = dict(topic=topic, judgments_only=judgments_only, direct=direct, anchored=anchored)
            row = dict(id=qid, reference=prior, previous_direct=reference[qid]["verified"],
                       requested_values=sorted(wanted))
            row.update({arm:sorted(values,key=lambda doc:(-values[doc],doc))[:10]
                        for arm,values in score_map.items()})
            assert row["topic"] == reference[qid]["retrieval"]
            for doc in sorted(direct, key=lambda key:(-direct[key],key)):
                scores.write(json.dumps(
                    [qid, doc, *(score_map[arm][doc] for arm in ARMS), grades.get(doc, {}),
                     sorted(evidence.get(doc, ()))], separators=(",", ":")) + "\n")
            gold = set(question["evidence_ids"])
            diagnostics["judged_pairs"] += len(grades)
            diagnostics["pairs_with_requested_values"] += len(evidence)
            diagnostics["anchored_added_gold"] += len((set(row["anchored"]) - set(prior)) & gold)
            diagnostics["anchored_lost_gold"] += len((set(prior) - set(row["anchored"])) & gold)
            diagnostics["anchored_questions_changed"] += set(prior) != set(row["anchored"])
            wanted_projection = plan({"question":query,"qtype":""})
            tokens = set(terms(query)) - DIRECT_STOP
            packet = {}
            for arm in ("reference", "previous_direct", *ARMS):
                ids = row[arm]
                assert len(ids) == len(set(ids)) == 10 and set(ids) <= records.keys()
                docs = []
                for doc in ids:
                    if doc in sources and doc in tags:
                        if doc not in frames:
                            frames[doc] = frame(sources[doc])
                        body, spans = project(sources[doc], tags[doc], frames[doc], wanted_projection, tokens, 3050)
                    else:
                        body, spans = metrics.utf8_prefix(records[doc]["text"], 3050), []
                    docs.append(dict(id=doc, text=body, source_spans=spans))
                assert sum(len(item["text"].encode()) for item in docs) <= BUDGET
                packet[arm] = docs
            ranks.write(json.dumps(row, ensure_ascii=False) + "\n")
            packets.write(json.dumps(dict(id=qid, question=query, evidence=packet), ensure_ascii=False) + "\n")
            traces.append(row)
    report = dict(n=len(traces), sources=len(records), data_sha256=hashes, budget=BUDGET,
                  new_model_api_tokens=0, diagnostics=dict(diagnostics),
                  all_scores_columns=["qid", "source", *ARMS, "grades", "values"],
                  metrics=metrics.measure(splits["Hard"], traces, ("reference","previous_direct",*ARMS),
                                          comprehensive_eval))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","output"):
        parser.add_argument(name,type=Path)
    main(parser.parse_args())
