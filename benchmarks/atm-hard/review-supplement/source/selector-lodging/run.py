"""Rank dated lodging versions, on-site media, and visit-entry observations."""
import argparse
from calendar import month_name
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import gzip
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import BUDGET, read_module
from run import load_inputs

HOTEL = re.compile(r"\b(?:hotel|hilton|novotel|sofitel|marriott|hyatt|resort|hostel|inn|suites|travelodge)\b", re.I)
STAY = re.compile(r"\b(?:stay|stayed|stays|staying|nights|visit\w*)\b", re.I)
ASK = re.compile(r"\b(?:which hotels?|how many nights)\b", re.I)
CANCELLED = re.compile(r"\b(?:cancellation notice|reservation cancellation|(?:was|has been|is) cancell?ed)\b", re.I)
CONFIRMED = re.compile(r"\b(?:reservation confirmation|confirmation of booking|reservation.*?confirmed|confirms? .*?reservation)\b", re.I)
COMPLETED = re.compile(r"\b(?:thank you for (?:your stay|staying)|receipt for recent stay|enjoyed your stay)\b", re.I)
PLAN = re.compile(r"\b(?:booking|reservation|check.in|check.out)\b", re.I)
ROOM = re.compile(r"\b(?:hotel room|guest room|room key|keycard|check.in desk|welcome card)\b", re.I)
ACTIVITY = re.compile(r"\b(?:breakfast|buffet|reception is underway|dining|meal)\b", re.I)
VISIT_COUNT = re.compile(r"\bhow many (?:times|visits|trips)\b", re.I)
TRANSIT = re.compile(r"\b(?:train|railway|railroad|station platform|terminal|boarding pass|arrivals?)\b", re.I)
MONTH = "|".join(month_name[1:])
SHORT_RANGE = re.compile(rf"\b({MONTH})\s+(\d{{1,2}})\s*[-–—]\s*(\d{{1,2}}),?\s+(20\d{{2}})\b", re.I)
LONG_RANGE = re.compile(rf"\b({MONTH})\s+(\d{{1,2}})(?:,?\s+(20\d{{2}}))?\s+(?:to|through|until)\s+({MONTH})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b", re.I)
ARMS = ("reference", "versions", "venue", "combined", "arrival", "combined_arrival")


def date_range(text):
    for pattern in (SHORT_RANGE, LONG_RANGE):
        match = pattern.search(text)
        if not match:
            continue
        if pattern is SHORT_RANGE:
            month, start_day, end_day, year = match.groups()
            start_month = end_month = month
            start_year = end_year = year
        else:
            start_month, start_day, start_year, end_month, end_day, end_year = match.groups()
            start_year = start_year or end_year
        try:
            start, end = [date(int(year), list(month_name).index(month.title()), int(day))
                          for year,month,day in ((start_year,start_month,start_day),(end_year,end_month,end_day))]
        except ValueError:
            continue
        if timedelta(0) < end - start <= timedelta(days=60):
            return start.isoformat(), end.isoformat(), match.group()
    return None


def build_catalogue(data, normalize):
    media, venues = {}, set()
    for kind in ("image", "video"):
        for value in json.loads((data / f"{kind}_batch_results.json").read_bytes()):
            location = (value.get("location_name") or "").split(",")[0].strip()
            if not HOTEL.search(location):
                continue
            place = normalize(location)
            try:
                when = datetime.fromisoformat(value["timestamp"]).replace(tzinfo=None)
            except (KeyError, TypeError, ValueError):
                continue
            caption = " ".join(value.get(field) or "" for field in ("short_caption","caption"))
            level = 3 if ROOM.search(caption) else 2 if ACTIVITY.search(caption) else 1
            key = Path(value[f"{kind}_path"]).stem
            media[key] = dict(venue=place, when=when, role=level, kind=kind)
            venues.add(place)
    ordered = sorted(venues, key=lambda place: (-len(place),place))
    names = re.compile(r"\b(?:" + "|".join(re.escape(place) for place in ordered) + r")\b")
    mail, transcripts = {}, {}
    for value in json.loads((data / "emails.json").read_bytes()):
        key = value["id"]
        fields = [value.get(field) or "" for field in ("detail", "short_summary")]
        text = "\n".join(fields)
        subjects = re.findall(r"(?im)^subject:\s*(.+)$",text)
        state = ("cancelled" if CANCELLED.search(" ".join(subjects)) or CANCELLED.search(text)
                 else "completed" if COMPLETED.search(text)
                 else "confirmed" if CONFIRMED.search(text) else "other")
        transcripts[key] = text
        if state == "other":
            continue
        matched = names.search(normalize(text))
        if not matched:
            continue
        stay_range = next((bounds for field in fields if (bounds:=date_range(field))), None)
        mail[key] = dict(venue=matched.group(), state=state, dates=stay_range,
                         stamp=value["timestamp"])
    return media, mail, transcripts


def restore_versions(prior, mail, media, transcripts, scores):
    selected = prior[:]
    traces = []
    by_booking = defaultdict(list)
    for key, row in mail.items():
        if row["dates"] and row["state"] == "confirmed":
            by_booking[row["venue"], *row["dates"][:2]].append(key)
    for key in prior:
        cancelled = mail.get(key)
        if not cancelled or cancelled["state"] != "cancelled" or not cancelled["dates"]:
            continue
        booking = (cancelled["venue"], *cancelled["dates"][:2])
        options = [doc for doc in by_booking[booking] if doc not in selected
                   and mail[doc]["stamp"] < cancelled["stamp"]]
        if not options:
            continue
        incoming = min(options,key=lambda doc:(-scores[doc], doc))
        # Keep all matched reservation, post-stay and on-site records. Other travel
        # artifacts cannot establish which version was in effect.
        other_media = Counter(media[doc]["venue"] for doc in selected if doc in media)
        outgoing = next((doc for doc in reversed(selected) if (
            (doc not in mail and doc not in media and not COMPLETED.search(transcripts.get(doc, ""))
             and not (HOTEL.search(transcripts.get(doc,"")) and PLAN.search(transcripts.get(doc,""))))
            or (doc in media and media[doc]["role"] == 1 and other_media[media[doc]["venue"]] > 1))), None)
        if outgoing is None:
            continue
        selected[selected.index(outgoing)] = incoming
        traces.append(dict(add=incoming, remove=outgoing, cancellation=key, venue=booking[0],
                           dates=list(booking[1:]), date_quote=mail[incoming]["dates"][2]))
    return selected, traces


def choose_venue_media(prior, media, mail):
    grouped = defaultdict(list)
    for key,row in media.items():
        grouped[row["venue"],row["kind"],row["when"].date()].append(key)
    confirmed_here = {mail[key]["venue"] for key in prior if key in mail and mail[key]["state"] == "completed"}
    selected = prior[:]
    traces = []
    for key in prior:
        row = media.get(key)
        if not row or row["venue"] not in confirmed_here or row["kind"] != "image":
            continue
        group = grouped[row["venue"],row["kind"],row["when"].date()]
        nearby = [doc for doc in group if abs(media[doc]["when"]-row["when"]) <= timedelta(minutes=5)]
        source = min(nearby,key=lambda doc:(-media[doc]["role"], media[doc]["when"], doc))
        if source in selected or media[source]["role"] < 2:
            continue
        selected[selected.index(key)] = source
        traces.append(dict(add=source,remove=key,venue=row["venue"],
                           capture_day=row["when"].date().isoformat(),
                           role=[row["role"],media[source]["role"]]))
    return selected, traces


def arrival_index(data, geo):
    captures = {}
    for record in json.loads((data / "image_batch_results.json").read_bytes()):
        candidate = geo.extract_capture(record, "image")
        if not candidate:
            continue
        try:
            when = datetime.fromisoformat(record["timestamp"]).replace(tzinfo=None)
        except (KeyError, TypeError, ValueError):
            continue
        text = " ".join(record.get(field) or "" for field in ("short_caption", "caption"))
        captures[candidate["id"]] = (candidate["city"], when, bool(TRANSIT.search(text)))
    return captures


def add_arrivals(query, prior, captures, normalize):
    target = f" {normalize(query)} "
    cities = Counter()
    for key in prior:
        if key in captures:
            city, _, _ = captures[key]
            if f" {city} " in target or f" {city.split()[0]} " in target:
                cities[city] += 1
    if not cities:
        return prior, []
    city = min(cities, key=lambda place: (-cities[place], place))
    city_images = sorted((key for key,(place,_,_) in captures.items() if place == city),
                         key=lambda key: (captures[key][1],key))
    visits = []
    for key in city_images:
        if not visits or captures[key][1] - captures[visits[-1][-1]][1] > timedelta(days=7):
            visits.append([])
        visits[-1].append(key)
    confirmed = [visit for visit in visits if set(visit) & set(prior)]
    if len(confirmed) < 2:
        return prior, []
    selected = prior[:]
    changes = []
    for visit in confirmed:
        first = visit[0]
        if first in selected or not captures[first][2]:
            continue
        remove = next((key for key in reversed(selected) if key in visit and not captures[key][2]
                       and captures[key][1] - captures[first][1] >= timedelta(minutes=30)), None)
        if remove is None:
            continue
        selected[selected.index(remove)] = first
        changes.append(dict(add=first, remove=remove, city=city,
                            first_observed=captures[first][1].isoformat()))
    return selected, changes


def main(args):
    sys.path[:0] = [str(args.release.resolve()), str(args.official.resolve()),
                    str(ROOT / "memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP, frame, plan, project

    records, _, splits, hashes = load_inputs(args.data)
    assert hashes == read_module("lodging_pins", ROOT / "iterative-search/search.py").PUBLIC_SHA256
    files = read_module("lodging_files", ROOT / "selector-source-ranking/run.py")
    evaluator = read_module("lodging_eval", ROOT / "selector-corpus/run.py")
    linked = read_module("lodging_sources", ROOT / "event-links/links.py")
    geo = read_module("lodging_geo", ROOT / "selector-geography/build.py")
    sources = linked.load_sources(args.data)
    tags = files.read_rows("event-hierarchy/source-tags.jsonl")
    base = files.read_rows("selector-weather-sequence/rankings.jsonl")
    media, mail, transcripts = build_catalogue(args.data,geo.normalize)
    captures = arrival_index(args.data,geo)
    requested = {q["id"] for q in splits["Hard"] if ASK.search(q["question"]) and STAY.search(q["question"])}
    scores = defaultdict(dict)
    with gzip.open(ROOT / "selector-proof/all-scores.jsonl.gz", "rt") as stream:
        for qid,key,_,_,direct,*_ in map(json.loads,stream):
            if qid in requested:
                scores[qid][key] = direct

    args.output.mkdir(parents=True,exist_ok=True)
    rows, packets, frames = [], [], {}
    for q in splits["Hard"]:
        qid, query = q["id"], q["question"]
        prior = base[qid]["day_only"]
        row = dict(id=qid, **{arm:prior for arm in ARMS})
        if qid in requested:
            row["versions"], row["version_swaps"] = restore_versions(prior, mail, media, transcripts, scores[qid])
            row["venue"], row["venue_swaps"] = choose_venue_media(prior, media, mail)
            row["combined"], _ = choose_venue_media(row["versions"], media, mail)
        if VISIT_COUNT.search(query) and STAY.search(query):
            row["arrival"], row["arrival_swaps"] = add_arrivals(query, prior, captures, geo.normalize)
        row["combined_arrival"] = row["combined"]
        if "arrival_swaps" in row:
            row["combined_arrival"], _ = add_arrivals(query, row["combined"], captures, geo.normalize)

        wanted = plan({"question":query,"qtype":""})
        tokens = set(terms(query)) - DIRECT_STOP
        evidence = {}
        for arm in ARMS:
            assert len(row[arm]) == len(set(row[arm])) == 10 and set(row[arm]) <= records.keys()
            docs = []
            for key in row[arm]:
                if key in sources and key in tags:
                    if key not in frames:
                        frames[key] = frame(sources[key])
                    body,spans = project(sources[key],tags[key],frames[key],wanted,tokens,3050)
                else:
                    body,spans = evaluator.utf8_prefix(records[key]["text"],3050), []
                docs.append(dict(id=key,text=body,source_spans=spans))
            assert sum(len(doc["text"].encode()) for doc in docs) <= BUDGET
            evidence[arm] = docs
        rows.append(row)
        packets.append(dict(id=qid,question=query,evidence=evidence))
    for name, data in (("rankings",rows),("reader-packets",packets)):
        with (args.output/f"{name}.jsonl").open("w") as stream:
            for row in data:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    report = dict(n=len(rows),data_sha256=hashes, budget=BUDGET, new_model_api_tokens=0,
                  lodging_emails=len(mail),venue_media=len(media), routed=len(requested),
                  changed={arm:sum(set(x[arm])!=set(x["reference"]) for x in rows) for arm in ARMS},
                  metrics=evaluator.measure(splits["Hard"],rows,ARMS,comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","output"):
        p.add_argument(name,type=Path)
    main(p.parse_args())
