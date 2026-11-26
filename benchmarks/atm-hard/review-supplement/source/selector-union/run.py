"""Evaluate a compact union of earlier top-ten results and its source-level reranking."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from subprocess import TimeoutExpired
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from cloud import TERRA,call
from evaluate import BUDGET,read_module,usage
from run import load_inputs

ROUTES=(
    ("history","selector-panels/session-followup/rankings.jsonl",("session_20",)),
    ("history","selector-panels/rankings.jsonl",("after",)),
    ("history","selector-coverage/consensus-rankings.jsonl",("contextual",)),
    ("history","selector-lanes/episode-rankings.jsonl",("lanes",)),
    ("geography","selector-coverage/rankings.jsonl",("coverage",)),
    ("candidate-reading","selector-answer-trace/rankings.jsonl",("merged",)),
    ("candidate-reading","selector-effort/rankings.jsonl",("low","high")),
    ("candidate-reading","selector-structure/rankings.jsonl",("structured",)),
    ("proof","selector-witness/review-rankings.jsonl",("review",)),
    ("proof","selector-recheck/partial/rankings.jsonl",("fulltext","replacement")),
    ("event","selector-propagation/rankings.jsonl",("propagation","fused")),
    ("event","selector-trip/rankings.jsonl",("fused",)),
    ("semantic","selector-scope/rankings.jsonl",("scope_cross",)),
    ("semantic","selector-pretrained/rankings.jsonl",("mixedbread","rewritten")),
    ("semantic","selector-distill/rankings.jsonl",("tree","pairwise")),
    ("visual","selector-media/rankings.jsonl",("vision",)),
)
PROMPT="""Choose exactly ten original source IDs from the supplied compact slate. Each original
can prove a different small part of the requested answer. Resolve the question's scope and
then select for distinct evidence coverage across the whole ten. A source proving a precise
date, city, route leg, amount, exact title or requested captured scene deserves attention even
if it uses different wording from the question. Check whether observations describe the
requested trip and whether dates are capture timestamps, itinerary dates, actual visits or
later document dates. Distinguish confirmed payment from booking. For repeated visits use
separate supported occurrences. For a list of matching original media IDs, retain each exact
matching source as its own item. Wherever the question has multiple required facts, cover
different useful facts before extra corroboration of one fact. Do not infer the answers from
nearby records or choose multiple copies of generic context. Output the ten ranked IDs and
for each an exact verbatim quote from its own excerpt and the concrete fact it supplies.
Use only the given sources. Do not output a final answer."""
FIELDS={"ranked":{"type":"array","items":{"type":"object","additionalProperties":False,
    "required":["id","quote","fact","priority"],
    "properties":{"id":{"type":"string"},"quote":{"type":"string"},"fact":{"type":"string"},
                  "priority":{"type":"integer","enum":[1,2,3]}}}}}


def union(*groups):
    return list(dict.fromkeys(item for group in groups for item in group))


def compose(variants,qid,before):
    groups=defaultdict(dict)
    for name,index,columns in variants:
        for column in columns:
            for rank,doc in enumerate(index[qid][column],1):
                value=12/(12+rank)
                groups[name][doc]=max(value,groups[name].get(doc,0))
    pool=union(before,*(list(group) for group in groups.values()))
    return pool,groups


def ordinal(pool,groups,before,retain):
    summary=defaultdict(float)
    weights={"history":1.3,"proof":1.24,"candidate-reading":1.1,"event":.83,
             "geography":.55,"semantic":.72,"visual":.65}
    for group,values in groups.items():
        for doc,rank in values.items():
            summary[doc]+=weights[group]*rank
    for i,doc in enumerate(before,1):
        summary[doc]+=retain*12/(12+i)
    return sorted(pool,key=lambda k:(-summary[k],k))


def select(result,visible,before):
    proof={}
    for row in result["ranked"][:10]:
        key,quote=row["id"],row["quote"]
        if key in visible and key not in proof and len(quote.strip())>=9 and quote in visible[key] and row["fact"].strip():
            proof[key]=row
    return union(list(proof),before)[:10],proof


def combine(before,local,llm,proof,groups):
    scores=defaultdict(float)
    for ids,weight in [(before,2.0),(local,1.1),(llm,1.0)]:
        for i,key in enumerate(ids,1):
            scores[key]+=weight*16/(16+i)
    candidate=[key for key in union(before,local,llm) if key in before or
               (key in proof and proof[key]["priority"]>=2 and sum(key in docs for docs in groups.values())>=2)]
    return sorted(candidate,key=lambda key:(-scores[key],key))[:10]


def request(cache,row):
    for attempt in range(3):
        try:
            return call(cache,"hard-compact-route-union-v1",TERRA,PROMPT,[row],FIELDS,effort="high")
        except (TimeoutExpired,RuntimeError,json.JSONDecodeError):
            if attempt==2:
                raise


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import terms
    from memory import DIRECT_STOP,frame,plan,project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    tools=read_module("route_union_files",ROOT/"selector-panels/run.py")
    metric=read_module("route_union_metrics",ROOT/"selector-corpus/run.py")
    records,kinds,splits,hashes=load_inputs(args.data)
    assert hashes==read_module("route_union_data",ROOT/"iterative-search/search.py").PUBLIC_SHA256
    baseline=tools.keyed(ROOT/"selector-answer-trace/agreement-rankings.jsonl")
    sources=read_module("route_union_sources",ROOT/"event-links/links.py").load_sources(args.data)
    tags=tools.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    variants=[(name,tools.keyed(ROOT/path),columns) for name,path,columns in ROUTES]
    frames={}
    data={}
    jobs=[]
    qs=splits["Hard"]
    for q in qs:
        key,query=q["id"],q["question"]
        before=baseline[key]["protected_agreement"]
        pool,groups=compose(variants,key,before)
        assert set(pool)<=records.keys()
        spec=plan({"question":query,"qtype":""})
        words=set(terms(query))-DIRECT_STOP
        shown={}
        for doc in pool:
            if doc in sources and doc in tags:
                if doc not in frames:
                    frames[doc]=frame(sources[doc])
                shown[doc]=project(sources[doc],tags[doc],frames[doc],spec,words,1850)[0]
            else:
                shown[doc]=tools.clip(records[doc]["text"],1850)
        order=sorted(pool,key=lambda doc:sha256(f"{key}+{doc}".encode()).digest())
        jobs.append(dict(id=key,question=query,format=q["qtype"],
                         sources=[dict(id=doc,medium=kinds[doc],text=shown[doc]) for doc in order]))
        data[key]=(before,pool,groups,shown)
    print("Top-ten union",len(qs),"questions",sum(len(row["sources"]) for row in jobs),"source pairs",flush=True)
    if args.prepare_only:
        return
    answers={}
    if not args.local_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(request,args.cache,row):row["id"] for row in jobs}
            for index,future in enumerate(as_completed(pending),1):
                answers[pending[future]]=future.result()
                if index%5==0 or index==len(jobs):
                    print("Compact union",index,"/",len(jobs),flush=True)
    traces=[];packets=[];bounds=[];accepted=0
    arms=("unweighted_baseline_retain","no_baseline_retain")
    if answers:
        arms+=("model","guarded")
    for q in qs:
        key,query=q["id"],q["question"]
        before,options,groups,shown=data[key]
        local=ordinal(options,groups,before,1.5)[:10]
        no_prior=ordinal(options,groups,before,0)[:10]
        choices=[local,no_prior]
        proof={}
        if answers:
            new,proof=select(answers[key]["items"][0],shown,before)
            choices.extend((new,combine(before,local,new,proof,groups)))
        row=dict(id=key,previous=before,candidates=options,verified_quotes=list(proof.values()))
        gold=set(q["evidence_ids"])
        bounds.append((len(gold&set(options)),len(gold),q["qtype"]))
        accepted+=len(proof)
        evidence={}
        for arm,rank in zip(arms,choices,strict=True):
            assert len(rank)==len(set(rank))==10 and set(rank)<=set(options)
            docs=[dict(id=doc,text=shown[doc]) for doc in rank]
            assert sum(len(item["text"].encode()) for item in docs)<=BUDGET
            row[arm]=rank
            evidence[arm]=docs
        traces.append(row)
        packets.append(dict(id=key,question=query,evidence=evidence))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in (("rankings",traces),("reader-packets",packets)):
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    report=dict(n=len(qs),data_sha256=hashes,evidence_budget_bytes=BUDGET,
        candidates=sum(len(row["candidates"]) for row in traces),candidate_gold=sum(n for n,d,k in bounds),
        candidate_ceiling=mean(min(n,10)/d for n,d,k in bounds),
        candidate_ceiling_by_qtype={t:mean(min(n,10)/d for n,d,k in bounds if k==t)
                                   for t in sorted({k for n,d,k in bounds})},accepted_quotes=accepted,
        usage=usage([row["usage"] for row in answers.values()]),
        metrics=metric.measure(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","cache","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--workers",type=int,default=11)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--local-only",action="store_true")
    main(p.parse_args())
