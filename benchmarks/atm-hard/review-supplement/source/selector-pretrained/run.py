"""Compare pretrained cross-encoders on identical Hard candidate evidence."""
import argparse
from collections import Counter
import json
from math import log1p
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from evaluate import BUDGET,read_module
from run import load_inputs

REVISION="4df024b67f8d3432f18a4149f2ae5a7e35f1ec27"
SHA256={"model.onnx":"15ef19a6de90be7d52b627f2c784107bd806e64826450f41fb75fa4f0179ab30",
        "tokenizer.json":"305674b4d785287feecfb5f73f24aa75e9b57c87c579cfe24fbd207987d4b4c4"}


def ranked(ids, values):
    return sorted(ids,key=lambda doc:(-values[doc],doc))


def reciprocal(order):
    return {doc:20/(20+i) for i,doc in enumerate(order,1)}


def proof_scores(row,older,second,claims):
    result={a["id"]:a["score"] for a in row}
    combined={**older,**second}
    for doc,proof in combined.items():
        result[doc]=max(result.get(doc,0),max(a["strength"]-1 for a in proof))
    for a in claims:
        doc=a["id"]
        result[doc]=max(result.get(doc,0),{"direct":3,"partial":2,"context":1}[a["role"]])
    return result


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import terms
    from memory import DIRECT_STOP,frame,plan,project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    metrics=read_module("external_metrics",ROOT/"selector-corpus/run.py")
    engine=read_module("external_pair",ROOT/"rerank/run.py")
    helper=read_module("external_files",ROOT/"selector-panels/run.py")
    records,_,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("external_pins",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Required fixed public Hard split")
    mixer=engine.CrossEncoder(args.mixedbread)
    if mixer.hashes!=SHA256:
        raise ValueError("Invalid Mixedbread checkpoint or tokenizer hash")
    legacy=engine.CrossEncoder(args.minilm)
    assert mixer.score("Where did the workshop take place?",[
        "The workshop was held in Oslo.", "A sleeping dog lay on the sofa."])[0]>-3
    src=read_module("external_source",ROOT/"event-links/links.py").load_sources(args.data)
    tags=helper.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    final=helper.keyed(ROOT/"selector-answer-trace/agreement-rankings.jsonl")
    pools=helper.keyed(ROOT/"selector-answer-trace/rankings.jsonl")
    recheck=helper.keyed(ROOT/"selector-recheck/partial/rankings.jsonl")
    witness=helper.keyed(ROOT/"selector-witness/rankings.jsonl")
    review=helper.keyed(ROOT/"selector-witness/review-rankings.jsonl")
    scopes=helper.keyed(ROOT/"selector-scope/rankings.jsonl")
    rewrites_raw=json.loads(args.rewrites.read_text())
    rewrites={row["id"]:row["queries"] for row in rewrites_raw["items"]}
    qs=splits["Hard"]
    traces=[]
    packets=[]
    time_taken=[]
    frames={}
    arms=("minilm","mixedbread","rewritten","fused","corroborated")
    for i,q in enumerate(qs,1):
        start=perf_counter()
        key,query=q["id"],q["question"]
        before=final[key]["protected_agreement"]
        ids=helper.union(before,pools[key]["candidates"])
        shape=plan({"question":query,"qtype":""})
        keywords=set(terms(query))-DIRECT_STOP
        excerpts={}
        for doc in ids:
            if doc in src and doc in tags:
                if doc not in frames:
                    frames[doc]=frame(src[doc])
                evidence=project(src[doc],tags[doc],frames[doc],shape,keywords,1450)[0]
            else:
                evidence=helper.clip(records[doc]["text"],1450)
            excerpts[doc]=evidence
        texts=[excerpts[doc] for doc in ids]
        mini=dict(zip(ids,legacy.score(query,texts),strict=True))
        mixed=dict(zip(ids,mixer.score(query,texts),strict=True))
        alternate=next((s for s in rewrites[key] if s.strip()),query)
        alt=dict(zip(ids,mixer.score(alternate,texts),strict=True))
        a,b,c=(ranked(ids,vals) for vals in (mini,mixed,alt))
        mini_rank,mixed_rank,alt_rank,old_rank,scope_rank=(
            reciprocal(order) for order in (a,b,c,before,scopes[key]["scope_cross"]))
        claims=proof_scores(recheck[key]["graded"],witness[key]["claims"],
                           review[key]["newly_quoted"],pools[key]["partial_statements"])
        rewritten=ranked(ids,{doc:.65*mixed_rank[doc]+.35*alt_rank[doc] for doc in ids})
        fused=ranked(ids,{doc:.56*mixed_rank[doc]+.28*alt_rank[doc]+.16*mini_rank[doc]
                         +.24*claims.get(doc,0)+.76*old_rank.get(doc,0)+.15*scope_rank.get(doc,0)
                         for doc in ids})
        votes=Counter(a[:15]+b[:15]+c[:15]+scopes[key]["scope_cross"][:8])
        competitive=[doc for doc in fused if doc in before or
                     (votes[doc]>=2 and claims.get(doc,0)>=2)]
        safeguarded=ranked(competitive,{
            doc:.42*mixed_rank[doc]+.30*alt_rank[doc]+.20*mini_rank[doc]+.24*claims.get(doc,0)
                + .95*old_rank.get(doc,0)+.06*log1p(votes[doc]) for doc in competitive})
        trace=dict(id=key,previous=before,scope_pool=ids,query_variant=alternate,
                   scores=[dict(id=doc,minilm=mini[doc],mixedbread=mixed[doc],alternate=alt[doc]) for doc in ids])
        packet={}
        for arm,order in zip(arms,(a,b,rewritten,fused,safeguarded),strict=True):
            chosen=order[:10]
            assert len(chosen)==len(set(chosen))==10 and set(chosen)<=set(ids)
            evidence=[dict(id=doc,text=excerpts[doc]) for doc in chosen]
            assert sum(len(item["text"].encode()) for item in evidence)<=BUDGET
            trace[arm]=chosen
            packet[arm]=evidence
        traces.append(trace)
        packets.append(dict(id=key,question=query,evidence=packet))
        time_taken.append(perf_counter()-start)
        print("Paired Hard rankings",i,"/",len(qs),round(time_taken[-1],2),"s",flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in (("rankings",traces),("reader-packets",packets)):
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    stats=dict(n=len(qs),data_sha256=hashes,evidence_budget_bytes=BUDGET,new_api_tokens=0,
               model_revision=REVISION,model_sha256=mixer.hashes,minilm_sha256=legacy.hashes,
               scored_pairs=sum(len(row["scope_pool"]) for row in traces),engine_runs_per_pair=3,
               compute_seconds=sum(time_taken),metrics=metrics.measure(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps(stats),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("release","data","official","mixedbread","minilm","rewrites","output"):
        parser.add_argument(n,type=Path)
    main(parser.parse_args())
