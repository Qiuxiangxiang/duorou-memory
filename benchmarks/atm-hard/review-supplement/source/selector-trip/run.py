"""Use source-grounded trip phases to expand candidates and rank original evidence."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from itertools import zip_longest
import json
from math import log1p
from pathlib import Path
import re
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

TRAVEL = re.compile(r"\b(?:visits?|visited|visted|trip|travel\w*|hotels?|stayed|nights?|cities|countries|"
                    r"conferences?|duration|depart\w*|return\w*)\b", re.I)
PRECISE_SOURCE = re.compile(r"\b(?:ids?|reimburs\w*|pay|paid|cost|spend|spent)\b", re.I)
ROLE = ("outbound", "arrival", "visit", "lodging", "attendance", "return", "other")
ROLE_WORDS = {
    "outbound": "departure outbound itinerary boarding pass origin",
    "arrival": "arrival transit destination landed boarding pass",
    "visit": "visited photographed capture observed presence",
    "lodging": "hotel accommodation reservation check-in check-out stay",
    "attendance": "attended conference badge poster session keynote",
    "return": "return flight inbound departure boarding pass itinerary",
    "other": "",
}
PLAN = """Use the question and evidence excerpts to plan searches for missing original records.
Identify the concrete phases or distinct requested facts that need separate proof. Travel
requests may need an outbound leg, on-site observations, lodging, separate city visits, and
the final return date. An event visit is different from an invitation. A planned booking
and a completed stay are different. For each useful search give a role, specific search
words, a city or country filter only when grounded, and optional inclusive ISO start/end
dates. Use dates from the question or an actual source quote, and leave dates blank when
unknown. If a proposed date comes from an excerpt, cite that ID and an exact quote containing
its calendar text. Do not infer a single trip window for a question about multiple visits.
Create 2 to 6 complementary searches. Preserve broad scope when the question is broad.
Do not answer the question."""
PLAN_ROW = {"type":"object","additionalProperties":False,
    "required":["role","query","place","start","end","source_id","quote"],
    "properties":{k:{"type":"string"} for k in ("query","place","start","end","source_id","quote")}}
PLAN_ROW["properties"]["role"] = {"type":"string","enum":list(ROLE)}
PLAN_FIELDS = {"searches":{"type":"array","items":PLAN_ROW}}
CHECK = """Compare the source records together for evidence of the exact requested trip or
series of visits. Choose the ten most useful ORIGINAL sources. Rank verified contributions
by distinct part of the answer. A record that proves one requested city or date is useful even
if it does not prove an entire trip. Preserve sources documenting departure and return if
requested. Recorded location is evidence of actual presence on the capture date. A ticket,
confirmation, invitation, receipt, cancellation and on-site observation are different roles.
Travel planning may still establish an explicit route or scheduled date when qualified.
Use the exact source context to decide if a mention is the requested trip, an unrelated same
city, a transfer or a later event. For repeated visit or hotel questions, use separate observed
episodes and lodging names, and avoid inferring separate visits from nearby snapshots. Choose
records that best establish a distinct answer-bearing item or documentary date even if their
topic similarity is low. Return ten IDs, each with a short verbatim quote from its excerpt, a
role, and the specific fact. Do not give the answer itself or assume facts not shown."""
CHECK_FIELDS = {"ranked":{"type":"array","items":{"type":"object","additionalProperties":False,
    "required":["id","quote","fact","role","confidence"],
    "properties":{"id":{"type":"string"},"quote":{"type":"string"},"fact":{"type":"string"},
                  "role":{"type":"string","enum":list(ROLE)},
                  "confidence":{"type":"integer","enum":[1,2,3]}}}}}


def tokens(text):
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def exact_date_in(day, text):
    normalized = tokens(text)
    forms = [day.isoformat()]
    for month in (day.strftime("%B"), day.strftime("%b")):
        forms.extend((f"{month} {day.day} {day.year}", f"{day.day} {month} {day.year}"))
    return any(f" {tokens(value)} " in f" {normalized} " for value in forms)


def interleave(*lists):
    return list(dict.fromkeys(key for items in zip_longest(*lists) for key in items if key))


def date_window(request, query, shown):
    a, b = request["start"].strip(), request["end"].strip()
    safe = a and b
    source, quote = request["source_id"], request["quote"]
    if not safe:
        return None
    if source in shown and quote.strip() and quote in shown[source]:
        text = query + " " + quote
    else:
        text = query
    try:
        lo, hi = date.fromisoformat(a), date.fromisoformat(b)
    except ValueError:
        return None
    if lo > hi or not all(exact_date_in(day, text) for day in (lo, hi)):
        return None
    return lo-timedelta(days=1), hi+timedelta(days=1)


def pick_new(before, proposed, old_support, phase_matches):
    prior = set(before)
    ordered = [source for source, _ in proposed]
    pure = list(dict.fromkeys([*ordered, *before]))[:10]
    layers = defaultdict(float)
    corroboration = Counter()
    for rank, source in enumerate(before, 1):
        layers[source] += 1.12 - .038 * rank + .10 * old_support[source]
    for rank, (source, verdict) in enumerate(proposed, 1):
        layers[source] += .35 + .44 * verdict["confidence"] - .030 * rank
        corroboration[source] = len(phase_matches.get(source, []))
        layers[source] += .09 * log1p(corroboration[source])
    combined = sorted(layers, key=lambda k: (-layers[k], k))[:10]
    # Return a conservative variant that keeps cross-run evidence support before
    # admitting rediscovered sources absent from the original ten.
    candidates = [k for k in pure if k in prior or old_support[k] >= 1 or corroboration[k] >= 2]
    anchored = list(dict.fromkeys([*candidates, *before]))[:10]
    return pure, combined, anchored


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()), str(ROOT / "memory-design")]
    from cloud import TERRA, JUDGE
    from duorou.retrieval import BM25, terms, rrf
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    util = read_module("trip_utils", ROOT / "selector-panels/run.py")
    scorer = read_module("trip_metrics", ROOT / "selector-corpus/run.py")
    api = read_module("trip_api", ROOT / "event-graph-full/benchmark.py")
    src_module = read_module("trip_sources", ROOT / "event-links/links.py")
    geo_module = read_module("trip_geo", ROOT / "selector-geography/build.py")
    records, kinds, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("trip_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    sources = src_module.load_sources(args.data)
    all_geo = {}
    for kind in ("image","video"):
        for row in json.loads((args.data/f"{kind}_batch_results.json").read_text()):
            cap = geo_module.extract_capture(row, kind)
            if cap:
                all_geo[cap["id"]] = cap
    tags = util.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    event = util.keyed(ROOT/"event-graph-full/sources.jsonl")
    baseline = util.keyed(ROOT/"selector-lanes/episode-rankings.jsonl")
    witness = util.keyed(ROOT/"selector-witness/rankings.jsonl")
    review = util.keyed(ROOT/"selector-witness/review-rankings.jsonl")
    geo = util.keyed(ROOT/"selector-geography/pools.jsonl")
    prior = util.keyed(ROOT/"selector-scope/rankings.jsonl")
    old_panel = util.keyed(ROOT/"selector-panels/rankings.jsonl")
    old_candidates = util.keyed(ROOT/"iterative-search/rankings.jsonl")
    qs = splits["Hard"]

    full = BM25({key:r["text"] for key,r in records.items()})
    concise = {key:" ".join((r.city, r.location, str(r.day), *event.get(key,{}).get("index_terms",[]),
                             *(r.fields.get(f,"") for f in ("short_summary","short_caption","ocr_text"))))
               for key,r in sources.items()}
    brief = BM25(concise)
    frames = {}
    prepared = {}
    requests = []
    for q in qs:
        key, query = q["id"], q["question"]
        before = baseline[key]["episode_guard"]
        prepared[key] = {}, before
        if q["qtype"] == "list_recall" or not TRAVEL.search(query) or PRECISE_SOURCE.search(query):
            continue
        terms_q = set(terms(query)) - DIRECT_STOP
        requirements = plan({"question":query,"qtype":""})
        cards = prepared[key][0]
        for source in before:
            if source not in sources or source not in tags:
                cards[source] = util.clip(records[source]["text"], 1600)
                continue
            if source not in frames:
                frames[source] = frame(sources[source])
            cards[source] = project(sources[source],tags[source],frames[source],requirements,terms_q,1600)[0]
        requests.append(dict(id=key, question=query, baseline=[dict(id=src,text=cards[src]) for src in before]))
    print("Trip planning questions",len(requests),flush=True)
    if args.prepare_only:
        return
    plan_out, plan_usage = api.gather(args.cache,"hard-trip-phase-plan-v1",TERRA,PLAN,requests,
                                       PLAN_FIELDS,5,110000,args.workers)
    grade_jobs = []
    trace_parts = {}
    for q in qs:
        key, query = q["id"], q["question"]
        if key not in plan_out:
            continue
        cards, before = prepared[key]
        search = plan_out[key]["searches"][:6]
        multiple_ranks = []
        phase_matches = defaultdict(list)
        search_log = []
        for phase in search:
            search_value = phase["query"].strip()[:180]
            if not search_value:
                continue
            words = " ".join(dict.fromkeys([*terms(search_value), *terms(phase["place"]), *terms(ROLE_WORDS[phase["role"]])]))
            route = [full.search(search_value,14),brief.search(search_value,14),
                     full.search(words,15),brief.search(words,15)]
            fusion = rrf(route)
            ranked = sorted(fusion,key=lambda src:(-fusion[src],src))
            span = date_window(phase, query, cards)
            name = tokens(phase["place"])
            accepted = []
            if name:
                for source, coord in all_geo.items():
                    geography = tokens(coord["city"] + " " + coord["country"])
                    correct_place = f" {name} " in f" {geography} "
                    if correct_place and (not span or span[0] <= coord["day"] <= span[1]):
                        accepted.append(source)
                accepted.sort(key=lambda src:(src not in fusion,-fusion.get(src,0),
                                               not all_geo[src]["boundary"],src))
            # Keep distinct observed dates when searching broad location requests.
            by_day = defaultdict(list)
            for source in accepted:
                coord = all_geo[source]
                by_day[(coord["city"],coord["day"])].append(source)
            distinct = [group[0] for group in by_day.values()]
            pool = interleave(ranked[:10],distinct[:10])[:15]
            for src in pool:
                phase_matches[src].append(phase["role"])
            multiple_ranks.append(pool)
            search_log.append(dict(role=phase["role"],query=search_value,place=name,
                                   date_range=[str(day) for day in span] if span else [],retrieved=pool))
        claims = {**witness[key]["claims"],**review[key]["newly_quoted"]}
        previous_support = Counter()
        for group in (before,old_panel[key]["after"],list(claims),old_candidates[key]["pool"][:12]):
            previous_support.update(group)
        geo_known = geo[key]["extra"][:18]
        proof_order = sorted(claims,key=lambda source:(-previous_support[source],-max(
            item["strength"] for item in claims[source]),source))
        current = util.union(before,interleave(interleave(*multiple_ranks),proof_order,
                                               prior[key]["scope_cross"],geo_known))[:68]
        needed = plan({"question":query,"qtype":""})
        tks = set(terms(query)) - DIRECT_STOP
        for src in current:
            if src in cards:
                continue
            if src not in sources or src not in tags:
                cards[src] = util.clip(records[src]["text"],1450)
            else:
                if src not in frames:
                    frames[src] = frame(sources[src])
                cards[src] = project(sources[src],tags[src],frames[src],needed,tks,1450)[0]
        grade_jobs.append(dict(id=key,question=query,search_roles=[r["role"] for r in search_log],
                                sources=[dict(id=src,kind=kinds[src],text=cards[src],
                                              retrieval_roles=list(dict.fromkeys(phase_matches[src])))
                                         for src in current[::2]+current[1::2]]))
        trace_parts[key] = current, search_log, phase_matches, previous_support
    print("Trip ranking questions",len(grade_jobs),"sources",sum(len(x["sources"]) for x in grade_jobs),flush=True)
    ranks, rank_usage = api.gather(args.cache,"hard-trip-phase-ranking-v1",JUDGE,CHECK,grade_jobs,
                                    CHECK_FIELDS,2,155000,args.workers)
    traces = []
    packets = []
    eligible = []
    valid_quotes = 0
    for q in qs:
        key, query = q["id"],q["question"]
        cards,before=prepared[key]
        choices = []
        row=dict(id=key,previous=before)
        evidence={}
        if key in ranks:
            candidate, paths, phase_matches, old_support = trace_parts[key]
            seen=set()
            for result in ranks[key]["ranked"][:10]:
                src,text=result["id"],result["quote"]
                if src in candidate and src not in seen and len(text.strip())>=9 and text in cards[src] and result["fact"].strip():
                    choices.append((src,result))
                    seen.add(src)
            after = pick_new(before,choices,old_support,phase_matches)
            row.update(candidates=candidate,searches=paths,decisions=[proof for _,proof in choices])
            valid_quotes+=len(choices)
            gold=set(q["evidence_ids"])
            eligible.append((len(gold&set(candidate)),len(gold)))
        else:
            after=(before,before,before)
        arms=("phase","fused","corroborated")
        for arm,order in zip(arms,after,strict=True):
            assert len(order)==len(set(order))==10 and set(order)<=records.keys()
            row[arm]=order
            docs=[dict(id=src,text=cards.get(src) or util.clip(records[src]["text"],1700)) for src in order]
            assert sum(len(doc["text"].encode()) for doc in docs)<=BUDGET
            evidence[arm]=docs
        traces.append(row)
        packets.append(dict(id=key,question=query,evidence=evidence))
    args.output.mkdir(parents=True,exist_ok=True)
    api.write_rows(args.output/"rankings.jsonl",traces)
    api.write_rows(args.output/"reader-packets.jsonl",packets)
    report=dict(n=len(qs),data_sha256=hashes,evidence_budget_bytes=BUDGET,planned=len(plan_out),
                valid_model_quotes=valid_quotes,plan_usage=plan_usage,rank_usage=rank_usage,
                activated_candidate_gold=sum(n for n,d in eligible),
                activated_candidate_ceiling=mean(min(n,10)/d for n,d in eligible),
                metrics=scorer.measure(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","data","official","cache","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--workers",type=int,default=9)
    parser.add_argument("--prepare-only",action="store_true")
    main(parser.parse_args())
