"""Fuse evidence ranks across distinct retrieval families and benchmark at ten."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from evaluate import BUDGET,read_module
from run import load_inputs


def score(runs):
    result=defaultdict(float)
    votes=Counter()
    for ranking, weight in runs:
        for position,key in enumerate(ranking,1):
            result[key]+=weight*12/(12+position)
            votes[key]+=1
    return result,votes


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import terms
    from memory import DIRECT_STOP,frame,plan,project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    helper=read_module("agreement_lookup",ROOT/"selector-panels/run.py")
    metric=read_module("agreement_metrics",ROOT/"selector-corpus/run.py")
    links=read_module("agreement_origins",ROOT/"event-links/links.py")
    records,_,splits,hashes=load_inputs(args.data)
    assert hashes==read_module("agreement_public",ROOT/"iterative-search/search.py").PUBLIC_SHA256
    sources=links.load_sources(args.data)
    tags=helper.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    history=helper.keyed(ROOT/"selector-lanes/episode-rankings.jsonl")
    initial=helper.keyed(ROOT/"selector-panels/session-followup/rankings.jsonl")
    witness=helper.keyed(ROOT/"selector-witness/review-rankings.jsonl")
    contribution=helper.keyed(ROOT/"selector-contribution/rankings.jsonl")
    trip=helper.keyed(ROOT/"selector-trip/rankings.jsonl")
    scope=helper.keyed(ROOT/"selector-scope/rankings.jsonl")
    media=helper.keyed(ROOT/"selector-media/rankings.jsonl")
    extra=helper.keyed(args.trace) if args.trace else {}
    frames={}
    traces=[]
    packets=[]
    for q in splits["Hard"]:
        key,query=q["id"],q["question"]
        baseline=history[key]["episode_guard"]
        sources_by_family=[
            (initial[key]["session_20"],1.05),
            (witness[key]["review"],1.05),
            (contribution[key]["model"],0.90),
            (scope[key]["scope_cross"],0.75),
        ]
        if "candidates" in trip[key]:
            sources_by_family.append((trip[key]["fused"],0.85))
        if media[key]["ratings"]:
            sources_by_family.append((media[key]["vision"],0.80))
        if key in extra:
            sources_by_family.append((extra[key]["merged"],1.05))
        fused,votes=score(sources_by_family)
        baseline_positions={doc:i for i,doc in enumerate(baseline)}
        baseline_ids=set(baseline)
        row=dict(id=key,previous=baseline)
        arms=("broad_agreement","protected_agreement")
        for arm,weight in zip(arms,(2.05,3.4),strict=True):
            candidates=[doc for doc in fused if votes[doc]>=2 or doc in baseline_ids]
            candidates=helper.union(candidates,baseline)
            ranked=sorted(candidates,key=lambda doc:(-(fused[doc]+(
                   weight*12/(13+baseline_positions[doc]) if doc in baseline_positions else 0)),doc))
            row[arm]=ranked[:10]
        row["votes"]={doc:votes[doc] for doc in set(row["broad_agreement"])|set(row["protected_agreement"])}
        evidence={}
        question_plan=plan({"question":query,"qtype":""})
        query_tokens=set(terms(query))-DIRECT_STOP
        for arm in arms:
            chosen=row[arm]
            assert len(chosen)==len(set(chosen))==10 and set(chosen)<=records.keys()
            docs=[]
            for doc in chosen:
                if doc in sources and doc in tags:
                    if doc not in frames:
                        frames[doc]=frame(sources[doc])
                    body,spans=project(sources[doc],tags[doc],frames[doc],question_plan,query_tokens,3050)
                else:
                    body,spans=helper.clip(records[doc]["text"],3050),[]
                docs.append(dict(id=doc,text=body,spans=spans))
            assert sum(len(x["text"].encode()) for x in docs)<=BUDGET
            evidence[arm]=docs
        traces.append(row)
        packets.append(dict(id=key,question=query,evidence=evidence))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,items in (("agreement-rankings",traces),("agreement-reader-packets",packets)):
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
    report=dict(n=len(traces),data_sha256=hashes,budget=BUDGET,model_api_tokens=0,
        uses_new_answer_ranking=bool(extra),
        set_changes={arm:sum(set(x[arm])!=set(x["previous"]) for x in traces) for arm in arms},
        metrics=metric.measure(splits["Hard"],traces,("previous",*arms),comprehensive_eval))
    (args.output/"agreement-summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","data","official","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--trace",type=Path)
    main(p.parse_args())
