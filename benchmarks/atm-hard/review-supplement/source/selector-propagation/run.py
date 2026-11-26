"""Evaluate typed evidence propagation over the pinned ATM-Bench source graph."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from itertools import combinations
from math import sqrt
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from evaluate import BUDGET,read_module
from run import load_inputs

RELATION_WEIGHTS={"mentions_named_edition":1.4,"mentions_identifier":2.4,
                  "same_recorded_place_and_day":.50,"next_nearby_observation":1.0}
MEDIA=re.compile(r"\b(?:photo\w*|video\w*|imag(?:e|es)|media|slides|ids?)\b",re.I)
IMAGE=re.compile(r"\b(?:photo\w*|images?)\b",re.I)
VIDEO=re.compile(r"\b(?:videos?|timelapse|slow.motion)\b",re.I)
FEE=re.compile(r"\b(?:pay|paid|cost|spend|spent|reimburs\w*)\b",re.I)


def rows(path):
    with path.open() as inp:
        return [json.loads(line) for line in inp]


def keyed(path):
    return {row["id"]:row for row in rows(path)}


def relations(edges,ids):
    groups=defaultdict(list)
    pairs={}
    for row in edges:
        label=row["relation"]
        if label not in RELATION_WEIGHTS:
            continue
        left,right=(row[n].removeprefix("observation:") for n in ("from","to"))
        if left not in ids:
            continue
        if label=="next_nearby_observation":
            if right not in ids or left==right or row["seconds"]>5400:
                continue
            multiplier=1/(1+row["seconds"]/900)
            pairs[(min(left,right),max(left,right),label)]=RELATION_WEIGHTS[label]*multiplier
        else:
            groups[(label,right)].append(left)
    for (label,_),entries in groups.items():
        unique=sorted(set(entries))
        if not 2<=len(unique)<=65:
            continue
        strength=RELATION_WEIGHTS[label]/sqrt(len(unique)-1)
        for a,b in combinations(unique,2):
            pairs[a,b,label]=max(strength,pairs.get((a,b,label),0))
    return pairs


def sparse_paths(pairs,positions,allowed,kinds,route):
    from scipy.sparse import csr_matrix, diags

    left=[]
    right=[]
    values=[]
    direct=defaultdict(list)
    family={}
    for (a,b,label),weight in pairs.items():
        if not allowed[a] or not allowed[b]:
            continue
        if route=="media" and (kinds[a]=="email" or kinds[b]=="email"):
            continue
        if route=="payment" and label=="same_recorded_place_and_day":
            # A nearby record is a weak cue for payments without a transaction reference.
            weight*=.35
        ia,ib=positions[a],positions[b]
        left.extend((ia,ib))
        right.extend((ib,ia))
        values.extend((weight,weight))
        direct[a].append((b,label,weight))
        direct[b].append((a,label,weight))
        family[label]=family.get(label,0)+1
    size=len(positions)
    shared=csr_matrix((values,(left,right)),shape=(size,size))
    normalizer=np.maximum(np.asarray(shared.sum(axis=0)).ravel(),1e-12)
    return shared@diags(1/normalizer),direct,family


def ranks(order):
    return {key:16/(16+rank) for rank,key in enumerate(order,1)}


def readable_category(text,qtype):
    if FEE.search(text):
        return "payment"
    if qtype=="list_recall" and MEDIA.search(text):
        return "media"
    return "general"


def grades(recheck,first,followup,partial):
    proof={row["id"]:row["score"] for row in recheck}
    for doc,facts in {**first,**followup}.items():
        proof[doc]=max(proof.get(doc,0),max(f["strength"]-1 for f in facts))
    for fact in partial:
        doc=fact["id"]
        proof[doc]=max(proof.get(doc,0),{"direct":3,"partial":2,"context":1}[fact["role"]])
    return proof


def get_corpus_rank(path):
    result=defaultdict(dict)
    with gzip.open(path,"rt") as f:
        for line in f:
            key,order,doc,topic,scoped,logit=json.loads(line)
            result[key][doc]=(order,topic,scoped)
    return result


def graph_scores(vector,transition):
    result=vector.copy()
    for _ in range(14):
        result=.57*vector+.43*(transition@result)
    return result


def lead_neighbors(doc,before,direct):
    return [dict(id=other,relation=label,weight=round(weight,6))
            for other,label,weight in direct.get(doc,[]) if other in before]


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import terms
    from memory import DIRECT_STOP,frame,plan,project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    metric=read_module("path_metrics",ROOT/"selector-calibration/run.py")
    records,kinds,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("graph_pins",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected fixed public ATM-Bench archive")
    graph=ROOT/"event-graph-full"
    assert json.loads((graph/"graph-summary.json").read_text())["data_sha256"]==hashes
    ids=sorted(records)
    positions={doc:i for i,doc in enumerate(ids)}
    pairs=relations(rows(graph/"edges.jsonl"),positions)
    baseline=keyed(ROOT/"selector-answer-trace/agreement-rankings.jsonl")
    trace=keyed(ROOT/"selector-answer-trace/rankings.jsonl")
    recheck=keyed(ROOT/"selector-recheck/partial/rankings.jsonl")
    witness=keyed(ROOT/"selector-witness/rankings.jsonl")
    followup=keyed(ROOT/"selector-witness/review-rankings.jsonl")
    pretrained=keyed(ROOT/"selector-pretrained/rankings.jsonl")
    tags=keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    adapter=read_module("graph_sources",ROOT/"event-links/links.py")
    source_records=adapter.load_sources(args.data)
    lexical=get_corpus_rank(ROOT/"selector-scope/all-scores.jsonl.gz")
    qs=splits["Hard"]
    frames={}
    results=[]
    packets=[]
    activate=Counter()
    arms=("propagation","fused","guarded")
    for q in qs:
        key,query=q["id"],q["question"]
        before=baseline[key]["protected_agreement"]
        prior=set(before)
        old_rank=ranks(before)
        proof=grades(recheck[key]["graded"],witness[key]["claims"],followup[key]["newly_quoted"],
                     trace[key]["partial_statements"])
        semantic=ranks(pretrained[key]["rewritten"])
        old_lead={doc for doc in before[:6] if proof.get(doc,0)>=2}
        old_lead=old_lead or set(before[:4])
        scope=readable_category(query,q["qtype"])
        activate[scope]+=1
        image=scope=="media" and IMAGE.search(query) and not VIDEO.search(query)
        video=scope=="media" and VIDEO.search(query) and not IMAGE.search(query)
        allowed={doc:((not image or kinds[doc]=="image") and (not video or kinds[doc]=="video"))
                    for doc in ids}
        assert len(lexical[key])==len(ids)
        semantic_seed=np.zeros(len(ids))
        original_seed=np.zeros(len(ids))
        for doc in ids:
            pos,_,_=lexical[key][doc]
            score=16/(16+pos)
            semantic_seed[positions[doc]]=(.48*score+.26*semantic.get(doc,0)
                                            +.14*proof.get(doc,0)/3+.12*old_rank.get(doc,0))
            original_seed[positions[doc]]=(semantic_seed[positions[doc]] if pos<=170 else 0.0) + (
                .95 if doc in old_lead else .48*old_rank.get(doc,0))
        path,direct,stats=sparse_paths(pairs,positions,allowed,kinds,scope)
        propagation=graph_scores(original_seed,path)
        qualified=pd_threshold(propagation)
        graph_order=sorted(ids,key=lambda doc:(-propagation[positions[doc]],doc))
        fused=sorted(ids,key=lambda doc:(-(.38*qualified[positions[doc]]+.44*semantic_seed[positions[doc]]
                                            +.55*old_rank.get(doc,0)),doc))
        # Guard against a cluster overwhelming the ranked set. A new record requires
        # an independent excerpt and at least two corroborating links to existing sources.
        pool=[doc for doc in list(dict.fromkeys([*before,*graph_order[:80],*fused[:80]]))
              if doc in prior or (proof.get(doc,0)>=2 and allowed[doc] and
                  len({r["id"] for r in lead_neighbors(doc,prior,direct)})>=2)]
        protected=sorted(pool,key=lambda doc:(-(.35*qualified[positions[doc]]+.51*semantic_seed[positions[doc]]
                                               +.69*old_rank.get(doc,0)),doc))
        row=dict(id=key,previous=before,scope=scope,edges=stats,seed_ids=sorted(old_lead),added_links={})
        evidence={}
        anchor=plan({"question":query,"qtype":""})
        question_terms=set(terms(query))-DIRECT_STOP
        for arm,order in zip(arms,(graph_order,fused,protected),strict=True):
            selected=order[:10]
            assert len(selected)==len(set(selected))==10 and set(selected)<=records.keys()
            row[arm]=selected
            row["added_links"][arm]={doc:lead_neighbors(doc,prior,direct) for doc in selected if doc not in prior}
            docs=[]
            for doc in selected:
                if doc in source_records and doc in tags:
                    if doc not in frames:
                        frames[doc]=frame(source_records[doc])
                    text,spans=project(source_records[doc],tags[doc],frames[doc],anchor,question_terms,3050)
                else:
                    text,spans=records[doc]["text"].encode()[:3050].decode(errors="ignore"),[]
                docs.append(dict(id=doc,text=text,spans=spans))
            assert sum(len(d["text"].encode()) for d in docs)<=BUDGET
            evidence[arm]=docs
        results.append(row)
        packets.append(dict(id=key,question=query,evidence=evidence))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,out in (("rankings",results),("reader-packets",packets)):
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out))
    summary=dict(n=len(qs),data_sha256=hashes,evidence_budget=BUDGET,new_api_tokens=0,
                 graph_edges=len(pairs),scored_pairs=len(qs)*len(ids),routing=dict(activate),
                 changes={arm:sum(set(row[arm])!=set(row["previous"]) for row in results) for arm in arms},
                 metrics=metric.record_metrics(qs,results,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


def pd_threshold(vector):
    scale=np.quantile(vector,.998)
    return np.minimum(vector/max(float(scale),1e-9),1.35)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","output"):
        p.add_argument(name,type=Path)
    main(p.parse_args())
