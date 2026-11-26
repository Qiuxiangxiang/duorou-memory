"""Assess every uncited finalist and test bounded replacements with exact proof."""
import argparse
from collections import Counter, defaultdict
import json
from math import log1p
from pathlib import Path
import re
import sys
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from evaluate import BUDGET,read_module
from run import load_inputs

ASSESS="""For every supplied original source, make a separate evidence decision for the
question. There is no quota on supporting sources. Return all supplied IDs. Use score 3
when it establishes a specific requested item or necessary exact fact, 2 when it establishes
a concrete but incomplete part, 1 for contextual or generic relevance, 0 when outside
scope. Include a literal contiguous quote for scores 2 or 3 and name what it establishes.
Record a canonical short item key using the exact place, event, endpoint, transaction,
date or media ID when it identifies a separate answer fact. Distinguish observed presence,
reservation, itinerary, transaction and capture timestamp. A nearby record is evidence
only when its own contents or recorded place/time support the question. For requested
original media collections, a matching source is its own item even if it looks like another.
Use a blank quote and item for scores 0 and 1. Do not invent facts or answer the question."""
FIELDS={"decisions":{"type":"array","items":{"type":"object","additionalProperties":False,
        "required":["id","score","item","quote","fact"],"properties":{
            "id":{"type":"string"},"score":{"type":"integer","enum":[0,1,2,3]},
            "item":{"type":"string"},"quote":{"type":"string"},"fact":{"type":"string"}}}}}
COMPARE="""Look only for justified one-for-one changes to a ten-source evidence list.
An alternative must add a requested specific fact, a distinct requested media original,
or a materially more direct source for the same fact. Compare the exact supporting excerpts.
Keep an incumbent when evidence for replacing it is uncertain. Remove a source only if
its scope is wrong, it supplies generic context alone, or every unique answer-relevant
fact it supplies remains proved by another retained source. For geography and chronology
the actual time and visited place matter; a plan or nearby record does not automatically
transfer proof. Preserve legitimate independent evidence for a distinct answer item.
Return zero to three changes. For each change quote the alternative exactly and explain the
added fact and why removal is safe. If removal is safe because another source supports
the same fact, give its ID and quote. Use empty retained fields only for an unrelated
or out-of-scope incumbent. Do not answer the question."""
SWAP_FIELDS={"changes":{"type":"array","items":{"type":"object","additionalProperties":False,
    "required":["add","remove","quote","fact","why_remove","retained","retained_quote"],
    "properties":{k:{"type":"string"} for k in (
        "add","remove","quote","fact","why_remove","retained","retained_quote")}}}}


def normalize(text):
    return " ".join(re.findall(r"[^\W_]+",text.casefold()))


def accept(row,source,body,exact_media):
    if row["id"]!=source:
        return None
    quote=row["quote"]
    if row["score"]>=2 and (len(quote.strip())<9 or quote not in body or not row["fact"].strip()
                            or not normalize(row["item"])):
        return None
    return {**row,"item":source if exact_media else normalize(row["item"])[:155]}


def apply_swaps(before,alternatives,visible,changes):
    current=list(before)
    preserved=set()
    approved=[]
    rejected=[]
    allowed=set(alternatives)
    for row in changes[:3]:
        new,old,keeper=row["add"],row["remove"],row["retained"]
        quote=row["quote"]
        if new not in allowed or new in current or old not in current or old not in before or old in preserved:
            reason="invalid source or protected proof"
        elif len(quote.strip())<9 or quote not in visible[new] or not row["fact"].strip() or not row["why_remove"].strip():
            reason="missing quoted proof or explanation"
        elif keeper and (keeper not in current or keeper==old or len(row["retained_quote"].strip())<9
                         or row["retained_quote"] not in visible[keeper]):
            reason="retained support unavailable"
        else:
            reason=""
        if reason:
            rejected.append({**row,"validation":reason})
        else:
            current[current.index(old)]=new
            if keeper:
                preserved.add(keeper)
            approved.append(row)
    return current,approved,rejected


def direct_rank(before,approved,older,positions):
    positive=[key for key,row in approved.items() if row["score"]>=2]
    observed={key:i for i,key in enumerate(before)}
    keys=list(dict.fromkeys([*before,*positive]))

    def score(key):
        previous=1.05-.04*observed[key] if key in observed else 0
        proof=older.get(key,0)
        review=approved.get(key,{}).get("score",0)
        return previous+.70*max(proof,review)+.16*log1p(min(proof,review))

    return sorted(keys,key=lambda key:(-score(key),positions[key],key))[:10]


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from cloud import TERRA,JUDGE
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP,frame,plan,project

    helper=read_module("recheck_files",ROOT/"selector-panels/run.py")
    runner=read_module("recheck_batches",ROOT/"event-graph-full/benchmark.py")
    metric=read_module("recheck_score",ROOT/"selector-corpus/run.py")
    records,kinds,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("recheck_pins",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected hash-verified public ATM-Bench data")
    adapter=read_module("recheck_source",ROOT/"event-links/links.py")
    sources=adapter.load_sources(args.data)
    tags=helper.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    latest=helper.keyed(ROOT/"selector-answer-trace/agreement-rankings.jsonl")
    old=helper.keyed(ROOT/"selector-answer-trace/rankings.jsonl")
    witness=helper.keyed(ROOT/"selector-witness/rankings.jsonl")
    revisit=helper.keyed(ROOT/"selector-witness/review-rankings.jsonl")
    qs=splits["Hard"]
    requests=[]
    jobs={}
    prepared={}
    frames={}
    for q in qs:
        key,query=q["id"],q["question"]
        trace=old[key]
        before=latest[key]["protected_agreement"]
        candidate=helper.union(before,trace["candidates"])
        omitted=[doc for doc in candidate if doc not in set(trace["shortlist"])|set(before)]
        tks=set(terms(query))-DIRECT_STOP
        spec=plan({"question":query,"qtype":""})
        shown={}
        for doc in candidate:
            if doc in sources and doc in tags:
                if doc not in frames:
                    frames[doc]=frame(sources[doc])
                body,_=project(sources[doc],tags[doc],frames[doc],spec,tks,2250)
            else:
                body=helper.clip(records[doc]["text"],2250)
            shown[doc]=body
        older=defaultdict(int)
        for key_, claims in {**witness[key]["claims"],**revisit[key]["newly_quoted"]}.items():
            older[key_]=max(c["strength"]-1 for c in claims)
        for claim in trace["partial_statements"]:
            key_=claim["id"]
            older[key_]=max(older[key_],{"direct":3,"partial":2,"context":1}[claim["role"]])
        prepared[key]=(before,candidate,shown,older,omitted)
        for index in range(0,len(omitted),14):
            group=omitted[index:index+14]
            row_id=f"{key}:{index//14+1}"
            jobs[row_id]=(key,group)
            requests.append(dict(id=row_id,question=query,format=q["qtype"],
                                 requirements=witness[key]["requirements"],
                                 sources=[dict(id=doc,kind=kinds[doc],text=shown[doc]) for doc in group]))
    print("Sources without first-round citations",sum(len(j["sources"]) for j in requests),
          "panels",len(requests),flush=True)
    if args.prepare_only:
        return
    results,use_assess=runner.gather(args.cache,"hard-uncited-fulltext-v1",TERRA,ASSESS,
                                    requests,FIELDS,4,152000,args.workers)
    actual=defaultdict(dict)
    by_question={q["id"]:q for q in qs}
    for row_id,value in results.items():
        key,group=jobs[row_id]
        displayed=prepared[key][2]
        group_set=set(group)
        for row in value["decisions"]:
            doc=row["id"]
            if doc in group_set and doc not in actual[key]:
                parsed=accept(row,doc,displayed[doc],by_question[key]["qtype"]=="list_recall")
                if parsed:
                    actual[key][doc]=parsed
    compare=[]
    compare_ids={}
    for q in qs:
        key=q["id"]
        before,candidate,shown,older,_=prepared[key]
        order={doc:i for i,doc in enumerate(candidate)}
        threshold = 2 if args.include_partial and q["qtype"] != "list_recall" else 3
        eligible=sorted((doc for doc,info in actual[key].items() if info["score"]>=threshold),
                        key=lambda doc:(-older.get(doc,0),-actual[key][doc]["score"],order[doc]))
        if args.include_partial:
            # Compare distinct proposed facts before repeated descriptions of the same item.
            representatives={}
            for doc in eligible:
                representatives.setdefault(actual[key][doc]["item"],doc)
            alternatives=helper.union(list(representatives.values()),eligible)[:12]
        else:
            alternatives=eligible[:8]
        if alternatives:
            compare_ids[key]=alternatives
            compare.append(dict(id=key,question=q["question"],format=q["qtype"],
                current=[dict(id=doc,text=shown[doc]) for doc in before],
                alternatives=[dict(id=doc,text=shown[doc],verified=actual[key][doc]) for doc in alternatives]))
    print("One-for-one comparisons",len(compare),flush=True)
    comparison_prompt=COMPARE
    if args.include_partial:
        comparison_prompt += (" An original that proves an exact missing sub-fact can be more valuable than a "
                              "complete description of a fact already covered. Consider actual capture dates, "
                              "distinct itinerary legs, venue identities and recorded presence as independent "
                              "partial facts if the question needs them.")
    replacements,use_compare=runner.gather(args.cache,"hard-uncited-replacements-v1",JUDGE,comparison_prompt,
                                          compare,SWAP_FIELDS,4,140000,args.workers)
    traces=[]
    packets=[]
    pools=defaultdict(list)
    changes=Counter()
    for q in qs:
        key=q["id"]
        before,candidate,shown,older,omitted=prepared[key]
        accepted=actual[key]
        positive={doc for doc,info in accepted.items() if info["score"]>=2}
        local=direct_rank(before,accepted,older,{doc:i for i,doc in enumerate(candidate)})
        after,edits,reject=apply_swaps(before,compare_ids.get(key,[]),shown,replacements.get(key,{"changes":[]})["changes"])
        arms=("fulltext","replacement")
        row=dict(id=key,previous=before,fulltext=local,replacement=after,candidates=candidate,
                 first_shortlist=old[key]["shortlist"],omitted=omitted,graded=list(accepted.values()),
                 accepted_swaps=edits,rejected_swaps=reject)
        evidence={}
        for field in arms:
            chosen=row[field]
            assert len(chosen)==len(set(chosen))==10 and set(chosen)<=set(candidate)
            docs=[dict(id=doc,text=shown[doc]) for doc in chosen]
            assert sum(len(item["text"].encode()) for item in docs)<=BUDGET
            evidence[field]=docs
            changes[field]+=set(chosen)!=set(before)
        gold=set(q["evidence_ids"])
        for label,ids in (("original_cited",set(old[key]["shortlist"])|set(before)),
                          ("after_fulltext",set(old[key]["shortlist"])|set(before)|positive),
                          ("all_candidates",set(candidate))):
            pools[label].append((len(gold&ids),len(gold)))
        traces.append(row)
        packets.append(dict(id=key,question=q["question"],evidence=evidence))
    args.output.mkdir(parents=True,exist_ok=True)
    runner.write_rows(args.output/"rankings.jsonl",traces)
    runner.write_rows(args.output/"reader-packets.jsonl",packets)
    report=dict(n=len(qs),data_sha256=hashes,evidence_budget_bytes=BUDGET,
        requested=sum(len(r["sources"]) for r in requests),
        valid_decisions=sum(map(len,actual.values())),
        accepted_positive=sum(row["score"]>=2 for g in actual.values() for row in g.values()),
        compare_questions=len(compare),allow_partial_replacements=args.include_partial,
        changed_questions=dict(changes),
        assess_usage=use_assess,compare_usage=use_compare,
        pools={name:dict(gold=sum(n for n,d in rows),ideal_recall_at_10=mean(min(n,10)/d for n,d in rows))
               for name,rows in pools.items()},
        metrics=metric.measure(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("release","data","official","cache","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--workers",type=int,default=11)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--include-partial",action="store_true")
    main(parser.parse_args())
