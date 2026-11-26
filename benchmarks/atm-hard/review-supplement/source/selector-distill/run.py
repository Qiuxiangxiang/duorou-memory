"""Distill existing grader judgments with each evaluation question held out."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from math import log1p
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from evaluate import BUDGET,read_module
from run import load_inputs

ROLE={"context":.20,"partial":.66,"direct":1.0}


def keyed(path):
    with path.open() as stream:
        return {r["id"]:r for r in map(json.loads,stream)}


def grouped_scores(path, fields, limit):
    result=defaultdict(dict)
    with gzip.open(path,"rt") as stream:
        for row in map(json.loads,stream):
            if row[1]<=limit:
                result[row[0]][row[2]]=tuple(row[i] for i in fields)
    return result


def list_rank(ranked):
    return {source: 12/(12+n) for n,source in enumerate(ranked,1)}


def strongest(claims):
    return {key:max((item["strength"]-1)/3 for item in rows)
            for key,rows in claims.items()}


def collect_teacher(key, trace, witness, review, scope, first, lanes, trip):
    votes=defaultdict(list)
    for group in (
        {v["source_id"]:v["score"]/4 for v in scope[key]["ratings"]},
        {v["id"]:v["score"]/3 for v in first[key]["graded"]},
        strongest({**witness[key]["claims"],**review[key]["newly_quoted"]}),
    ):
        for src,prob in group.items():
            votes[src].append(prob)
    independently=defaultdict(float)
    for claim in trace[key]["partial_statements"]:
        source=claim["id"]
        independently[source]=max(ROLE[claim["role"]],independently[source])
    for src,prob in independently.items():
        votes[src].append(prob)
    ranked=[trace[key]["merged"],review[key]["review"],lanes[key]["covered"],scope[key]["independent"]]
    if "decisions" in trip[key]:
        ranked.append(trip[key]["phase"])
    counts=Counter(source for ranking in ranked for source in ranking)
    return {src:.72*float(np.mean(votes[src]))+.28*counts[src]/len(ranked)
            for src in set(votes)|set(counts) if votes[src]}


def fit_predict(groups, test, method):
    assert test not in groups
    if method=="tree":
        x=np.vstack([g[0] for g in groups.values()])
        y=np.concatenate([g[1] for g in groups.values()])
        weights=np.concatenate([np.full(len(g[1]),1/len(g[1])) for g in groups.values()])
        model=HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=15,
                         min_samples_leaf=18,learning_rate=.065,l2_regularization=.30,random_state=42)
        model.fit(x,y,sample_weight=weights)
        return model.predict
    x=[]
    w=[]
    for features,scores in groups.values():
        high=np.flatnonzero(scores>=.55)
        low=np.flatnonzero(scores<=.40)
        if not high.size or not low.size:
            continue
        if high.size*low.size>240:
            high=high[::max(1,high.size//13)][:13]
            low=low[::max(1,low.size//17)][:17]
        n=len(high)*len(low)
        # Compare within question, so the model learns ordering rather than cohorts.
        for better in high:
            for worse in low:
                x.append(features[better]-features[worse])
                w.append((scores[better]-scores[worse])/n)
    data=np.asarray(x)
    weights=np.asarray(w)
    model=LogisticRegression(C=.75,fit_intercept=False,max_iter=300,random_state=42)
    model.fit(np.vstack([data,-data]),np.r_[np.ones(len(data)),np.zeros(len(data))],
              sample_weight=np.r_[weights,weights]*20)
    return lambda values: model.predict_proba(values)[:,1]


def percentile_ranks(scores,ids):
    return {key:12/(12+n) for n,key in enumerate(sorted(ids,key=lambda s:(-scores[s],s)),1)}


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import BM25,terms
    from memory import DIRECT_STOP,frame,plan,project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    tools=read_module("distill_features",ROOT/"selector-calibration/run.py")
    records,kinds,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("distill_pins",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected pinned public ATM-Bench files")
    names={
        "trace":"selector-answer-trace/rankings.jsonl",
        "final":"selector-answer-trace/agreement-rankings.jsonl",
        "witness":"selector-witness/rankings.jsonl",
        "review":"selector-witness/review-rankings.jsonl",
        "challenge":"selector-scope/challenge-rankings.jsonl",
        "recheck":"selector-recheck/partial/rankings.jsonl",
        "trip":"selector-trip/rankings.jsonl",
        "lanes":"selector-lanes/rankings.jsonl",
        "panels":"selector-panels/rankings.jsonl",
        "contribution":"selector-contribution/rankings.jsonl",
        "scope":"selector-scope/rankings.jsonl",
        "geography":"selector-geography/pools.jsonl",
    }
    saved={name:keyed(ROOT/path) for name,path in names.items()}
    tags=keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    events=keyed(ROOT/"event-graph-full/sources.jsonl")
    sources=read_module("distill_source",ROOT/"event-links/links.py").load_sources(args.data)
    cross_scores=grouped_scores(ROOT/"selector-scope/all-scores.jsonl.gz",(3,4,5),160)
    feature_index=tools.FeatureIndex(records,kinds,events,BM25,terms)
    qs=splits["Hard"]
    data={}
    group_rows={}
    for q in qs:
        key,query=q["id"],q["question"]
        packs={name:rows[key] for name,rows in saved.items()}
        before=packs["final"]["protected_agreement"]
        paths=(before,packs["trace"]["candidates"],packs["witness"]["candidate_ids"],
               packs["geography"]["extra"],packs["contribution"]["eligible"],
               packs["panels"]["shortlist"],packs["trip"].get("candidates",[]))
        ids,lists,raw=feature_index.candidates(query,paths)
        raw_features=feature_index.encode(query,ids,lists,raw)
        features=[]
        supports=[
            list_rank(packs["final"]["previous"]),
            list_rank(packs["panels"]["after"]),
            list_rank(packs["scope"]["scope_cross"]),
            list_rank(packs["trip"]["fused"]),
            list_rank(packs["contribution"]["model"]),
        ]
        grades=defaultdict(float)
        for row in packs["challenge"]["ratings"]:
            grades[row["source_id"]]=row["score"]/4
        for src in ids:
            local=[m.get(src,0) for m in supports]
            primary,scoped,cross=cross_scores[key].get(src,(0,0,None))
            covered=grades.get(src,0)
            # Source identity and evaluation labels never enter this matrix.
            features.append([*local,mean_nonzero(local),sum(v>0 for v in local),
                             primary,scoped,np.tanh(cross/6) if cross is not None else 0,covered,
                             float(src in packs["trace"]["shortlist"]),
                             float(src in packs["panels"]["shortlist"])])
        matrix=np.c_[raw_features,np.array(features)]
        targets=collect_teacher(key,saved["trace"],saved["witness"],saved["review"],saved["challenge"],
                                saved["recheck"],saved["lanes"],saved["trip"])
        observed=[n for n,src in enumerate(ids) if src in targets]
        group_rows[key]=(matrix[observed],np.asarray([targets[ids[n]] for n in observed]))
        data[key]=(before,ids,matrix,targets,supports,packs)
    packets=[]
    traces=[]
    frames={}
    arms=("tree","pairwise","blended","retained")
    for index,q in enumerate(qs,1):
        qid,query=q["id"],q["question"]
        before,ids,matrix,targets,supports,packs=data[qid]
        train={key:group for key,group in group_rows.items() if key!=qid}
        assert len(train)==len(qs)-1 and train.keys().isdisjoint({qid})
        output={}
        for flavor in ("tree","pairwise"):
            pred=fit_predict(train,qid,flavor)(matrix)
            output[flavor]=dict(zip(ids,map(float,pred),strict=True))
        a=percentile_ranks(output["tree"],ids)
        b=percentile_ranks(output["pairwise"],ids)
        eligible=set(before)|set(packs["contribution"]["eligible"])|set(packs["trace"]["shortlist"])
        eligible|={r["id"] for r in packs["recheck"]["graded"] if r["score"]>=2}
        prior=list_rank(before)
        blend={}
        contextual={}
        for key in eligible:
            score=.42*a[key]+.37*b[key]+.21*mean_nonzero([path.get(key,0) for path in supports])
            blend[key]=score+.52*prior.get(key,0)
            contextual[key]=score+.82*prior.get(key,0)
        variations=(output["tree"],output["pairwise"],blend,contextual)
        trace={"id":qid,"previous":before,"train_questions":len(train),"eligible":len(eligible),
               "sources_scored":len(ids)}
        evidence={}
        q_terms=set(terms(query))-DIRECT_STOP
        wanted=plan({"question":query,"qtype":""})
        for arm,scores in zip(arms,variations,strict=True):
            top=sorted(scores,key=lambda key:(-scores[key],key))[:10]
            assert len(top)==len(set(top))==10 and set(top)<=records.keys()
            trace[arm]=top
            texts=[]
            for key in top:
                if key in sources and key in tags:
                    if key not in frames:
                        frames[key]=frame(sources[key])
                    text,spans=project(sources[key],tags[key],frames[key],wanted,q_terms,3050)
                else:
                    text,spans=records[key]["text"].encode()[:3050].decode(errors="ignore"),[]
                texts.append(dict(id=key,text=text,spans=spans))
            assert sum(len(x["text"].encode()) for x in texts)<=BUDGET
            evidence[arm]=texts
        traces.append(trace)
        packets.append(dict(id=qid,question=query,evidence=evidence))
        if index%5==0 or index==len(qs):
            print("Held-out question",index,"/",len(qs),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in (("rankings",traces),("reader-packets",packets)):
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
    report=dict(n=len(qs),data_sha256=hashes,budget=BUDGET,new_api_tokens=0,
                transfer_groups=len(group_rows),teacher_examples=sum(len(y) for _,y in group_rows.values()),
                scored_pairs=sum(row["sources_scored"] for row in traces),
                metrics=tools.record_metrics(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


def mean_nonzero(values):
    positive=[x for x in values if x>0]
    return sum(positive)/len(positive) if positive else 0.


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","data","official","output"):
        p.add_argument(arg,type=Path)
    main(p.parse_args())
