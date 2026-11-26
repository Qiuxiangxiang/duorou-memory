"""Compare low and high reasoning with identical broad evidence on ATM-Bench Hard."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
from statistics import mean
from subprocess import TimeoutExpired
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from cloud import JUDGE,call
from evaluate import BUDGET,read_module,usage
from run import load_inputs

PROMPT="""Select the ten original records that best enable an independent reader to answer
the exact question. First determine the distinct facts, time or place constraints and
evidence type required. When a question asks for a collection of source IDs, each exact
matching original is a separate answer item. When another task asks for a list, prefer
different verified answer items and complementary evidence. When it asks for a journey
duration or list of cities, look for the distinct beginnings, endings, observations, and
documented itinerary legs. Separate capture date, scheduled event date, transaction date,
and actual event evidence. A booking, invitation, payment and observed presence have
different significance; some are valuable corroboration. Ensure the city, time, trip and
event scope match. A very clear source for a single premise may be more useful than
several chronologically adjacent copies. Do not rely on earlier labels or hints as proof.
Choose exactly ten distinct IDs from the current list. For each, provide text quoted
exactly from its own excerpt and a concise description of its factual contribution.
Avoid inventing answer facts, replacing the task with a narrower question, or grouping
different original source IDs as one. Output only the ranked evidence IDs and justification."""
FIELDS={"ranked":{"type":"array","items":{"type":"object","additionalProperties":False,
    "required":["id","quote","contribution","priority"],
    "properties":{"id":{"type":"string"},"quote":{"type":"string"},"contribution":{"type":"string"},
                  "priority":{"type":"integer","enum":[1,2,3]}}}}}


def union(*groups):
    return list(dict.fromkeys(doc for group in groups for doc in group))


def diverse(*groups):
    return list(dict.fromkeys(doc for line in zip_longest(*groups) for doc in line if doc))


def choose_order(seed, ids):
    return sorted(ids,key=lambda i:sha256(f"{seed}:{i}".encode()).digest())


def evaluate_request(cache, row, effort):
    for attempt in range(3):
        try:
            return call(cache,f"hard-effort-controlled-{effort}-v1",JUDGE,PROMPT,[row],FIELDS,effort=effort)
        except (RuntimeError,TimeoutExpired,json.JSONDecodeError):
            if attempt==2:
                raise


def read_rank(result, shown, baseline):
    approved={}
    for item in result["ranked"][:10]:
        doc,quote=item["id"],item["quote"]
        if doc in shown and doc not in approved and len(quote.strip())>=9 and quote in shown[doc] and item["contribution"].strip():
            approved[doc]=item
    return union(list(approved),baseline)[:10],approved


def agreement_rank(before,low,high,lproof,hproof):
    fusion={}
    for ranked,enc in ((before,3.1),(low,1.0),(high,1.9)):
        for pos,key in enumerate(ranked,1):
            fusion[key]=fusion.get(key,0)+enc*16/(16+pos)
    candidates=union(before,[doc for doc in high if doc in lproof and doc in hproof and hproof[doc]["priority"]>=2])
    return sorted(candidates,key=lambda key:(-fusion[key],key))[:10]


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    from memory import DIRECT_STOP,frame,plan,project
    helper=read_module("effort_tools",ROOT/"selector-panels/run.py")
    metrics=read_module("effort_score",ROOT/"selector-corpus/run.py")
    records,kinds,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("effort_data",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected hash-pinned public ATM-Bench archive")
    sources=read_module("effort_sources",ROOT/"event-links/links.py").load_sources(args.data)
    tags=helper.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    ranks=helper.keyed(ROOT/"selector-answer-trace/agreement-rankings.jsonl")
    previous=helper.keyed(ROOT/"selector-answer-trace/rankings.jsonl")
    recheck=helper.keyed(ROOT/"selector-recheck/partial/rankings.jsonl")
    scope=helper.keyed(ROOT/"selector-scope/rankings.jsonl")
    geo=helper.keyed(ROOT/"selector-geography/pools.jsonl")
    trip=helper.keyed(ROOT/"selector-trip/rankings.jsonl")
    panels=helper.keyed(ROOT/"selector-panels/rankings.jsonl")
    qs=splits["Hard"]
    frames={}
    requests=[]
    data={}
    for q in qs:
        key,query=q["id"],q["question"]
        before=ranks[key]["protected_agreement"]
        grades=sorted(recheck[key]["graded"],key=lambda v:(-v["score"],v["id"]))
        positive=[doc["id"] for doc in grades if doc["score"]>=2]
        raw=diverse(previous[key]["candidates"],positive,geo[key]["extra"],panels[key]["shortlist"],
                    trip[key].get("candidates",[]),scope[key]["scope_cross"])
        ids=union(before,previous[key]["shortlist"],raw)[:120]
        assert set(ids)<=records.keys()
        query_terms=set(terms(query))-DIRECT_STOP
        desired=plan({"question":query,"qtype":""})
        visible={}
        for doc in ids:
            if doc not in sources or doc not in tags:
                visible[doc]=helper.clip(records[doc]["text"],1450)
            else:
                if doc not in frames:
                    frames[doc]=frame(sources[doc])
                body,_=project(sources[doc],tags[doc],frames[doc],desired,query_terms,1450)
                visible[doc]=body
        order=choose_order(key,ids)
        request=dict(id=key,question=query,format=q["qtype"],
                     sources=[dict(id=doc,medium=kinds[doc],text=visible[doc]) for doc in order])
        requests.append(request)
        data[key]=(before,ids,visible)
    print("Paired effort inputs",len(requests),"Hard questions,",sum(len(r["sources"]) for r in requests),
          "question/source records",flush=True)
    if args.prepare_only:
        return
    outputs={"low":{},"high":{}}
    use={"low":[],"high":[]}
    jobs=[(effort,row) for row in requests for effort in ("low","high")]
    progress=Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as threads:
        futures={threads.submit(evaluate_request,args.cache,row,effort):(effort,row["id"]) for effort,row in jobs}
        for future in as_completed(futures):
            effort,key=futures[future]
            result=future.result()
            outputs[effort][key]=result["items"][0]
            use[effort].append(result["usage"])
            progress[effort]+=1
            if progress[effort]%5==0 or progress[effort]==len(qs):
                print(effort,"Hard",progress[effort],"/",len(qs),flush=True)
    traces=[]
    packets=[]
    candidates=[]
    valid=Counter()
    arms=("low","high","high_guarded")
    for q in qs:
        key,query=q["id"],q["question"]
        before,pool,shown=data[key]
        low,lproof=read_rank(outputs["low"][key],shown,before)
        high,hproof=read_rank(outputs["high"][key],shown,before)
        agreed=agreement_rank(before,low,high,lproof,hproof)
        valid["low"]+=len(lproof)
        valid["high"]+=len(hproof)
        r=dict(id=key,previous=before,low=low,high=high,high_guarded=agreed,candidates=pool,
               low_model_evidence=list(lproof.values()),high_model_evidence=list(hproof.values()))
        evidence={}
        for arm in arms:
            selected=r[arm]
            assert len(selected)==len(set(selected))==10 and set(selected)<=set(pool)
            items=[dict(id=doc,text=shown[doc]) for doc in selected]
            assert sum(len(item["text"].encode()) for item in items)<=BUDGET
            evidence[arm]=items
        traces.append(r)
        packets.append(dict(id=key,question=query,evidence=evidence))
        gold=set(q["evidence_ids"])
        candidates.append((len(gold&set(pool)),len(gold)))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,items in (("rankings",traces),("reader-packets",packets)):
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
    report=dict(n=len(qs),data_sha256=hashes,evidence_budget_bytes=BUDGET,model=JUDGE,
        usage={key:usage(bills) for key,bills in use.items()},valid_quotes=dict(valid),
        candidate_sets=sum(len(r["candidates"]) for r in traces),
        candidate_gold=sum(n for n,d in candidates),
        candidate_ceiling=mean(min(n,10)/d for n,d in candidates),
        selection_changes={arm:sum(set(r[arm])!=set(r["previous"]) for r in traces) for arm in arms},
        metrics=metrics.measure(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","data","official","cache","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--prepare-only",action="store_true")
    main(p.parse_args())
