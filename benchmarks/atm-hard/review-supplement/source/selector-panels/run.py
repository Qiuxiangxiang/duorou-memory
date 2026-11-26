"""Review the complete frozen Hard pool in panels, then choose ten sources together."""
import argparse
from collections import Counter, defaultdict
import json
from math import ceil, log
from pathlib import Path
from statistics import mean
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from run import load_inputs
from cloud import TERRA
from evaluate import BUDGET,read_module

PANEL_SIZE=28
PANEL_PICKS=12
FINAL_LIMIT=68
PANEL_PROMPT="""Compare the supplied PUBLIC benchmark source previews with the question. Pick
up to twelve candidates that provide direct or complementary evidence, prioritizing the sources
that most directly answer any part of the question. Find multiple distinct requested items when
possible; for media ID lists keep each matching original. For travel distinguish actual visit
observations, planned itinerary and financial documents. For an expense honor transaction
identity and actual charge. For a requested time, venue, title or visual activity prefer text
that documents that fact, preserving time and place constraints from the question. A generated
description may be incomplete. Retain a plausible direct source when part of the visual or
temporal evidence may be missing from the preview rather than assuming no relationship.
Return a priority from 3 (strong direct evidence) to 1 (plausible complementary evidence),
an exact short contiguous quote and a concise contribution for each selected source.
Do not answer the question. Return at most twelve choices using only supplied IDs."""
FINAL_PROMPT="""Select exactly ten unique source IDs for the question when the supplied list
contains ten possible sources. Use the whole set to maximize coverage of distinct answer facts
and requested items. Source order, prior inclusion and panel nominations are hints only. Start
with the question's scope and exact entity, activity, place and time. For a media listing each
distinct matching original is an answer item even if it resembles another. For travel include
the actual observations and itinerary that establish distinct segments and dates; a generic
same-conference notice is weak. For receipts distinguish payment from quoted price and count
different transactions as separate. Prefer complete factual support across complementary
sources rather than ten copies of generic topic matches. Merge no sources automatically.
Quote exactly from each chosen source and say which fact or item it contributes. Do not
invent exclusions, requested time windows, references, or content. Use only supplied IDs;
do not answer the question."""
PANEL_FIELDS={"choices":{"type":"array","items":{"type":"object","additionalProperties":False,
    "required":["id","priority","quote","contribution"],"properties":{
        "id":{"type":"string"},"priority":{"type":"integer","enum":[1,2,3]},
        "quote":{"type":"string"},"contribution":{"type":"string"},
    }}}}


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def keyed(path):
    return {row["id"]:row for row in rows(path)}


def union(*groups):
    return list(dict.fromkeys(x for group in groups for x in group))


def clip(text,size):
    return text.encode()[:size].decode("utf-8",errors="ignore")


def make_card(source, tag, segments, weights, limit):
    lines=[f"Recorded {tag['recorded_time_kind']}: {tag['recorded_time'] or 'unknown'}; place: {source.city or 'unknown'}"]
    lead=source.fields.get("short_caption") or source.fields.get("short_summary")
    if lead:
        lines.append("Summary: "+clip(lead,270))
    buckets=defaultdict(list)
    for seg in segments:
        rank=sum(weights.get(w,0) for w in sorted(seg.words))
        if seg.field=="detail" and seg.text.startswith("Subject:"):
            rank+=2
        buckets[seg.field].append((rank,seg.start,seg.text))
    candidates=[]
    for bucket,items in buckets.items():
        order=sorted(items,key=lambda x:(-x[0],x[1]))
        for priority,_,value in order[:2]:
            candidates.append((priority,bucket,value))
    candidates.sort(key=lambda x:(-x[0],x[1],x[2]))
    for _,name,value in candidates:
        if any(value.casefold() in line.casefold() for line in lines):
            continue
        room=limit-len("\n".join(lines).encode())-len(name.encode())-3
        if room<70:
            break
        lines.append(name+": "+clip(value,min(room,330)))
    return "\n".join(lines)


def accept_panel(result, shown):
    chosen={}
    for row in result["choices"][:PANEL_PICKS]:
        key=row["id"]
        if (key in shown and key not in chosen and len(row["quote"].strip())>=9
                and row["quote"] in shown[key] and row["contribution"].strip()):
            chosen[key]=row
    return chosen


def run(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval as retrieval
    from memory import DIRECT_STOP,frame,plan,project

    args.output.mkdir(parents=True,exist_ok=True)
    records,_,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("public_pins",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Only pinned public ATM-Bench files may be used")
    qs=splits["Hard"]
    pool=keyed(args.pool/"rankings.jsonl")
    previous=keyed(args.previous/"rankings.jsonl")
    assert len(qs)==31 and set(pool)==set(previous)=={q["id"] for q in qs}
    scorer=read_module("whole_set",args.previous/"run.py")
    grader=read_module("model_batches",ROOT/"event-graph-full/benchmark.py")
    source=read_module("source_parser",ROOT/"event-links/links.py").load_sources(args.data)
    tags=keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    relevance=defaultdict(list)
    for row in rows(args.relevance/"scores.jsonl"):
        relevance[row["question_id"]].append(row["source_id"])
    frames={}
    prepared={}
    panel_input=[]
    panel_map={}
    for q in qs:
        key=q["id"]
        candidates=pool[key]["pool"]
        old=previous[key]["after"]
        assert set(candidates)==set(relevance[key]) and set(old)<=set(candidates)<=set(records)
        for identifier in candidates:
            if identifier not in frames:
                frames[identifier]=frame(source[identifier])
        tokens=set(terms(q["question"]))-DIRECT_STOP
        frequencies=Counter(w for i in candidates for w in
                            {word for s in frames[i] for word in s.words}&tokens)
        weights={w:log(1+len(candidates)/(2+frequencies[w])) for w in sorted(tokens)}
        n=ceil(len(candidates)/PANEL_SIZE)
        order=union(old,relevance[key],candidates)
        prepared[key]=(weights,order)
        for slot in range(n):
            group=order[slot::n]
            shown={i:make_card(source[i],tags[i],frames[i],weights,850) for i in group}
            ident=f"{key}:{slot+1}"
            panel_map[ident]=(key,shown)
            panel_input.append(dict(id=ident,question=q["question"],format=q["qtype"],
                                    candidates=[{"id":i,"text":shown[i]} for i in group]))
    print("Hard full pool",len(panel_input),"panels",sum(len(p["candidates"]) for p in panel_input),flush=True)
    if args.prepare_only:
        return
    nominated,panel_usage=grader.gather(args.cache,"hard-all-source-panels-v1",TERRA,PANEL_PROMPT,
        panel_input,PANEL_FIELDS,5,159000,args.workers)
    review=defaultdict(dict)
    for ident,(key,shown) in panel_map.items():
        review[key].update(accept_panel(nominated[ident],shown))
    finals=[]
    traces=[]
    models={}
    for q in qs:
        key=q["id"]
        weights,order=prepared[key]
        nominations=review[key]
        order_position={i:n for n,i in enumerate(order)}
        ordered=sorted(nominations,key=lambda i:(-nominations[i]["priority"],order_position[i],i))
        shortlist=union(ordered[:56],previous[key]["after"],ordered,order)[:FINAL_LIMIT]
        shown={i:make_card(source[i],tags[i],frames[i],weights,1600) for i in shortlist}
        # Keep reader text identical to the text that can be quoted to the selector.
        models[key]=(shortlist,shown)
        finals.append(dict(id=key,question=q["question"],format=q["qtype"],
                           candidates=[{"id":i,"text":shown[i]} for i in shortlist]))
        traces.append(dict(id=key,before=previous[key]["after"],nominations=list(nominations.values()),
                           shortlist=shortlist))
    grader.write_rows(args.output/"nominations.jsonl",traces)
    print("Hard finalist shortlists",sum(len(p["candidates"]) for p in finals),flush=True)
    outputs,final_usage=grader.gather(args.cache,"hard-all-source-final-v1",TERRA,FINAL_PROMPT,
                                      finals,scorer.FIELDS,2,150000,args.workers)
    packets=[]
    for q,row in zip(qs,traces,strict=True):
        key=q["id"]
        shortlist,shown=models[key]
        ranked,audit,size=scorer.apply_selection(shortlist,shown,shown,outputs[key])
        row.update(after=ranked,decisions=audit,evidence_bytes=size)
        assert len(ranked)==len(set(ranked))==10 and size<=BUDGET
        packets.append(dict(id=key,question=q["question"],
                            evidence=[{"id":i,"text":shown[i]} for i in ranked]))
    grader.write_rows(args.output/"rankings.jsonl",traces)
    grader.write_rows(args.output/"reader-packets.jsonl",packets)
    stats=Counter()
    for q,row in zip(qs,traces,strict=True):
        g=set(q["evidence_ids"])
        before,after=map(set,(row["before"],row["after"]))
        stats["added_gold"]+=len(g&(after-before))
        stats["lost_gold"]+=len(g&(before-after))
        stats["changed_questions"]+=before!=after
        stats["local_fills"]+=sum(d["role"]=="local_fill" for d in row["decisions"])
    ceiling=mean(min(10,len(set(q["evidence_ids"])&set(r["shortlist"])))/len(q["evidence_ids"])
                 for q,r in zip(qs,traces,strict=True))
    summary={"n":31,"dataset_sha256":hashes,"shortlist_ceiling":ceiling,"changes":dict(stats),
             "before":scorer.calculate(qs,traces,"before",retrieval),
             "after":scorer.calculate(qs,traces,"after",retrieval),
             "panel_usage":panel_usage,"final_usage":final_usage}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","cache","pool","previous","relevance","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--workers",type=int,default=10)
    p.add_argument("--prepare-only",action="store_true")
    run(p.parse_args())
