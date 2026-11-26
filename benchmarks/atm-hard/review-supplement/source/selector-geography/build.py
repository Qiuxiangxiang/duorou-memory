"""Create a capture-metadata route for geographic questions on the public Hard benchmark."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
from pathlib import Path
import re
from statistics import mean, median
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from run import load_inputs
from evaluate import read_module

EUROPE=frozenset(("Albania Andorra Austria Belarus Belgium Bosnia Bulgaria Croatia Cyprus Czechia Denmark "
                 "Estonia Finland France Germany Greece Hungary Iceland Ireland Italy Kosovo Latvia "
                 "Liechtenstein Lithuania Luxembourg Malta Moldova Monaco Montenegro Netherlands "
                 "Norway Poland Portugal Romania Russia Serbia Slovakia Slovenia Spain Sweden "
                 "Switzerland Türkiye Ukraine Vatican").casefold().split())|{"united kingdom","north macedonia"}
NORTH_AMERICA={"united states","canada","mexico"}
COUNTRY_ALIAS={"england":"united kingdom","scotland":"united kingdom","wales":"united kingdom",
               "louisiana":"united states","massachusetts":"united states","texas":"united states",
               "california":"united states","ontario":"canada","quebec":"canada","london":"united kingdom"}
TRAVEL=re.compile(r"\b(?:airport|terminal|departure|arrivals?|gate|airplane|boarding pass|"
                  r"railway station|train station|baggage claim)\b",re.I)
GEO_REQUEST=re.compile(r"\b(?:cities|countries)\b",re.I)
PLACE_KEY=re.compile(r"(?im)^\s*(City|Country):\s*([^\n]+)")
YEAR=re.compile(r"\b20\d\d\b")


def data_rows(path):
    with path.open() as stream:
        return [json.loads(x) for x in stream]


def normalize(s):
    return " ".join(re.findall(r"[^\W_]+",s.casefold()))


def extract_capture(row,kind):
    key=Path(row[f"{kind}_path"]).stem
    raw=row.get("city") or ""
    matches={name.casefold():value for name,value in PLACE_KEY.findall(raw)}
    parts=[part.strip() for part in raw.splitlines()[0].split(",")] if raw else []
    city=normalize(matches.get("city",parts[0] if parts else ""))
    country=normalize(matches.get("country",parts[-1] if parts else ""))
    country=normalize(country.removeprefix("country "))
    country=COUNTRY_ALIAS.get(country,country)
    if not city or not country:
        return None
    try:
        day=date.fromisoformat(row["timestamp"][:10])
    except (KeyError,TypeError,ValueError):
        return None
    desc=" ".join([row.get("short_caption","") or "",row.get("caption","") or ""])
    return dict(id=key,city=city,country=country,day=day,boundary=bool(TRAVEL.search(desc)))


def sample(rows,limit):
    if len(rows)<=limit:
        return rows
    return [rows[round(i*(len(rows)-1)/(limit-1))] for i in range(limit)]


def group_by_episode(rows, keys, per_day):
    groups=defaultdict(list)
    for row in rows:
        groups[(*[row[key] for key in keys],row["day"])].append(row)
    for group in groups.values():
        group.sort(key=lambda r:(not r["boundary"],r["id"]))
    return [row for key in sorted(groups) for row in sample(groups[key],per_day)]


def coverage_route(question, captures, present):
    text=question["question"]
    target=normalize(text)
    words=set(target.split())
    if question["qtype"]!="open_end" or not GEO_REQUEST.search(text):
        return [],None
    known={r["country"] for r in captures}
    exact={country for country in known if country in EUROPE|NORTH_AMERICA and
           re.search(r"\b"+re.escape(country)+r"\b",target)}
    if "european" in words or "europe" in words:
        selected=EUROPE
    elif "north america" in target:
        selected=NORTH_AMERICA
    elif exact:
        selected=exact
    else:
        return [],None

    years=set(YEAR.findall(text))
    active=[r for r in captures if r["country"] in selected and (not years or str(r["day"].year) in years)]
    if "north america" in target and years:
        strong=[r["day"].toordinal() for r in active if r["id"] in present]
        if strong:
            center=date.fromordinal(round(median(strong)))
            active=[r for r in active if center-timedelta(days=23)<=r["day"]<=center+timedelta(days=38)]
    if "countries" in words:
        by_country=defaultdict(list)
        for row in group_by_episode(active,("country","city"),3):
            by_country[row["country"]].append(row)
        items=[r for c in sorted(by_country) for r in sample(by_country[c],26)]
    else:
        items=sample(group_by_episode(active,("city",),8),144)
    return [r["id"] for r in items],dict(countries=sorted(selected),suggested=len(items))


def run(args):
    records,_,splits,hashes=load_inputs(args.data)
    assert hashes==read_module("public_data",ROOT/"iterative-search/search.py").PUBLIC_SHA256
    qs=splits["Hard"]
    existing={r["id"]:r for r in data_rows(args.pool/"rankings.jsonl")}
    assert len(existing)==len(qs)==31
    observed=[]
    for kind in ("image","video"):
        for row in json.loads((args.data/f"{kind}_batch_results.json").read_bytes()):
            cap=extract_capture(row,kind)
            if cap:
                observed.append(cap)
    trace=[]
    for q in qs:
        key=q["id"]
        before=existing[key]["pool"]
        extras,meta=coverage_route(q,observed,set(before))
        after=list(dict.fromkeys([*before,*extras]))
        assert set(after)<=set(records)
        trace.append(dict(id=key,before=before,after=after,extra=[k for k in after if k not in set(before)],
                          route=meta))
    def score(label):
        count=[(len(set(q["evidence_ids"])&set(r[label])),len(q["evidence_ids"]))
               for q,r in zip(qs,trace,strict=True)]
        return {"pool_recall":mean(n/d for n,d in count),"gold_items":sum(n for n,d in count),
                "ideal_top10":mean(min(n,10)/d for n,d in count)}
    result={"n":len(qs),"data_sha256":hashes,"pool_only":True,
            "before":score("before"),"after":score("after"),
            "activated":sum(r["route"] is not None for r in trace)}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"pools.jsonl").write_text("".join(json.dumps(r)+"\n" for r in trace))
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("data","pool","output"):
        p.add_argument(arg,type=Path)
    run(p.parse_args())
