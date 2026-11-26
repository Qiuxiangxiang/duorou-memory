"""Choose every Top-10 slot together on the frozen public Hard candidate pool."""
import argparse
from collections import Counter
import json
from math import log
from pathlib import Path
from statistics import mean
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from run import load_inputs
from cloud import TERRA
from evaluate import BUDGET,read_module

SHORTLIST_LIMIT=36
LIMIT=10
PROMPT="""Select the best complete evidence set for each question. Every output position is open.
Rank exactly ten different supplied source IDs when ten useful sources exist. The order is by
direct answer support and complementary contribution. First identify what the question asks.
Across related records, prefer distinct, answer-relevant dates, entities, titles, observations,
transactions or media IDs. A difference alone is not evidence of relevance. For records giving
the same fact or event stage, choose the strongest direct source. Preserve material conflicts
when they could change the answer, citing their sources. For list questions requesting source IDs,
two distinct matching media records can both be useful even if they look alike.
Use original source quotes to verify support: identify the particular information each source
contributes, and quote one short exact contiguous phrase from that source. A general meeting
notice can be context but cannot replace paper titles, personal attendance evidence, or actual
transactions. Treat pair differences as hints, not identities. Do not answer the question or
introduce IDs outside the shortlist."""
FIELDS={"ranked":{"type":"array","items":{"type":"object","additionalProperties":False,
    "required":["id","role","claim","quote"],"properties":{
       "id":{"type":"string"},
       "role":{"type":"string","enum":["direct","complement","context","conflict"]},
       "claim":{"type":"string"},
       "quote":{"type":"string"},
    }}}}
DIFFERENCE_FIELDS={"amount_roles","route_wording","printed_dates","printed_clock_ranges",
                   "observed_location_text","activities"}


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def keyed(path):
    return {row["id"]:row for row in rows(path)}


def union(*sequences):
    return list(dict.fromkeys(i for s in sequences for i in s))


def pick_pair_hints(ranking, pairs_by_id):
    candidates=set(ranking)
    pairs={}
    for identifier in ranking:
        for row in pairs_by_id.get(identifier,[]):
            if row["a"] not in candidates or row["b"] not in candidates:
                continue
            values={key:value for key,value in row["differences"].items() if key in DIFFERENCE_FIELDS}
            if not values:
                continue
            key=(row["a"],row["b"])
            pairs[key]={"source_ids":list(key),"candidate_relation":row["candidate_relations"],
                        "differences":values}
    return sorted(pairs.values(),key=lambda p:(-len(p["differences"]),p["source_ids"]))[:8]


def apply_selection(order, shown, texts, model):
    short=set(order)
    selected=[]
    trace=[]
    used=0
    for row in model["ranked"]:
        key=row["id"]
        observed=row["quote"]
        issue=None
        if key not in short or key in selected:
            issue="outside shortlist or duplicate source"
        elif not row["claim"].strip() or len(observed.strip())<9 or observed not in shown[key] or observed not in texts[key]:
            issue="missing claim or unverified quote"
        elif len(selected)>=LIMIT or used+len(texts[key].encode())>BUDGET:
            issue="evidence budget"
        else:
            selected.append(key)
            used+=len(texts[key].encode())
        trace.append({**row,"accepted":issue is None,"reason":issue})
    for key in order:
        if len(selected)==LIMIT:
            break
        if key in selected or used+len(texts[key].encode())>BUDGET:
            continue
        selected.append(key)
        used+=len(texts[key].encode())
        trace.append({"id":key,"accepted":True,"role":"local_fill","reason":"unfilled model slot"})
    assert len(selected)==len(set(selected))<=10 and used<=BUDGET
    return selected,trace,used


def calculate(questions, records, field, retrieval):
    pairs=list(zip(questions,records,strict=True))
    scored=retrieval.compute_metrics(
         [dict(gt_evidence_ids=q["evidence_ids"],retrieval_ids=r[field]) for q,r in pairs],[10]
    )["metrics"]["recall@10"]
    counts=[(len(set(q["evidence_ids"])&set(r[field])),len(q["evidence_ids"])) for q,r in pairs]
    assert abs(scored-mean(n/d for n,d in counts))<1e-12
    return {"recall_at_10":scored,"gold_items":sum(n for n,_ in counts),
            "all_hit":sum(n==d for n,d in counts),"zero_hit":sum(not n for n,d in counts)}


def run(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import BM25,terms
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval as retrieval
    from memory import DIRECT_STOP,frame,plan,project

    args.output.mkdir(parents=True,exist_ok=True)
    records,_,splits,digest=load_inputs(args.data)
    public=read_module("pinned_public_data",ROOT/"iterative-search/search.py")
    if digest!=public.PUBLIC_SHA256:
        raise ValueError("Expected the pinned public ATM-Bench files")
    qs=splits["Hard"]
    base=keyed(args.previous/"rankings.jsonl")
    assert len(qs)==31 and set(base)=={q["id"] for q in qs}
    tags=keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    sources=read_module("source_loader",ROOT/"event-links/links.py").load_sources(args.data)
    gaps=read_module("gap_helpers",ROOT/"selector-gaps/run.py")
    judge=read_module("benchmark_grader",ROOT/"event-graph-full/benchmark.py")
    encoder=read_module("local_reranker",ROOT/"rerank/run.py").CrossEncoder(args.model)
    by_source={}
    for item in rows(ROOT/"unified-replay/candidate-differences.jsonl"):
        for source in (item["a"],item["b"]):
            by_source.setdefault(source,[]).append(item)

    frames={}
    pending=[]
    saved={}
    for n,q in enumerate(qs,1):
        key=q["id"]
        seed=base[key]
        pool=seed["pool"]
        previous=seed["after"]
        assert len(pool)==len(set(pool)) and set(previous)<=set(pool)<=set(records)
        lexical=set(terms(q["question"]))-DIRECT_STOP
        desired=plan(q)
        roles=gaps.roles_for(q)
        texts={}
        for src in pool:
            if src not in frames:
                frames[src]=frame(sources[src])
            texts[src]=project(sources[src],tags[src],frames[src],desired,lexical,3050)[0]
        frequency=Counter(word for value in texts.values() for word in set(terms(value)) if word in lexical)
        weights={word:log(1+len(pool)/(2+frequency[word])) for word in sorted(lexical) if len(word)>2}
        features={i:gaps.candidate_features(texts[i],roles,weights,terms,n) for n,i in enumerate(pool)}
        local_index=BM25(texts)
        focused=gaps.shortlist(pool,features,roles,set(),30)
        first=union(previous,local_index.search(q["question"],48),focused,pool[:12])
        previews={i:gaps.evidence_excerpt(i,features,1040) for i in first}
        values=encoder.score(q["question"],[previews[i] for i in first])
        by_score=dict(zip(first,values,strict=True))
        direct=sorted(first,key=lambda i:(-by_score[i],i))
        # The prior reader set is eligible, with no assigned output positions.
        working=union(direct[:16],focused[:10],previous)
        more=[]
        for src in direct[:12]:
            for edge in by_source.get(src,[]):
                other=edge["b"] if edge["a"]==src else edge["a"]
                if other in features and other not in working and DIFFERENCE_FIELDS&edge["differences"].keys():
                    if features[other][0]["direct"][0]>0:
                        more.append(other)
        shortlist=union(working,more,direct,focused)[:SHORTLIST_LIMIT]
        shown={i:gaps.evidence_excerpt(i,features,1040) for i in shortlist}
        pending.append(dict(id=key,question=q["question"],format=q["qtype"],
                       relevant_pair_hints=pick_pair_hints(shortlist,by_source),
                       candidates=[{"id":i,"text":shown[i]} for i in shortlist]))
        saved[key]=(shortlist,shown,texts,previous,pool)
        print("Hard shortlist",n,"/",len(qs),flush=True)

    result,usage=judge.gather(args.cache,"hard-whole-set-v1",TERRA,PROMPT,pending,FIELDS,5,168000,args.workers)
    traces=[]
    for q,packet in zip(qs,pending,strict=True):
        key=q["id"]
        shortlist,shown,texts,previous,pool=saved[key]
        selected,audit,used=apply_selection(shortlist,shown,texts,result[key])
        traces.append(dict(id=key,before=previous,after=selected,shortlist=shortlist,
                           differences=packet["relevant_pair_hints"],decisions=audit,evidence_bytes=used))
    judge.write_rows(args.output/"rankings.jsonl",traces)
    changes=Counter()
    by_type={}
    for q,r in zip(qs,traces,strict=True):
        old,new=map(set,(r["before"],r["after"]))
        gold=set(q["evidence_ids"])
        type_data=by_type.setdefault(q["qtype"],{"before":[],"after":[]})
        type_data["before"].append(len(gold&old)/len(gold))
        type_data["after"].append(len(gold&new)/len(gold))
        changes["gained_gold"]+=len((new-old)&gold)
        changes["lost_gold"]+=len((old-new)&gold)
        changes["more_gold_questions"]+=len(new&gold)>len(old&gold)
        changes["less_gold_questions"]+=len(new&gold)<len(old&gold)
        changes["first_source_changed"]+=r["after"][:1]!=r["before"][:1]
        changes["questions_changed"]+=old!=new
        changes["local_fill"]+=sum(d["role"]=="local_fill" for d in r["decisions"])
    summary={"n":len(qs),"data_sha256":digest,"mode":"retrieval only",
             "before":calculate(qs,traces,"before",retrieval),"after":calculate(qs,traces,"after",retrieval),
             "shortlist_ceiling":mean(min(10,len(set(r["shortlist"])&set(q["evidence_ids"])))/
                                      len(q["evidence_ids"]) for q,r in zip(qs,traces,strict=True)),
             "changes":dict(changes),"by_qtype":{kind:{k:mean(v) for k,v in data.items()}
                                              for kind,data in by_type.items()},
             "model_usage":usage,"shortlist_max":SHORTLIST_LIMIT}
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("release","data","official","model","cache","previous","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--workers",type=int,default=5)
    run(parser.parse_args())
