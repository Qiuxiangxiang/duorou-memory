"""Score two existing Hard rankings with an identical reader, judge and source projection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import ANSWER, BUDGET, STRING, read_module
from run import load_inputs

JUDGE_PROMPT = "Apply each supplied official grading prompt independently. Return a boolean accuracy per ID."


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from cloud import JUDGE, TERRA
    from duorou.retrieval import terms
    from memory import DIRECT_STOP, frame, plan, project
    from memqa.utils.evaluator import evaluate_qa as qa
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval

    helper=read_module("paired_helper",ROOT/"selector-panels/run.py")
    grader=read_module("paired_runner",ROOT/"event-graph-full/benchmark.py")
    metrics=read_module("paired_metrics",ROOT/"selector-corpus/run.py")
    records,_,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("paired_data",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected pinned public ATM-Bench inputs")
    sources=read_module("paired_sources",ROOT/"event-links/links.py").load_sources(args.data)
    tags=helper.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    rankings=helper.keyed(args.ranks)
    qs=splits["Hard"]
    base_frames={}
    samples={}
    arms=("previous",args.arm)
    for question in qs:
        qid,query=question["id"],question["question"]
        source_ids=helper.union(*(rankings[qid][arm] for arm in arms))
        assert set(source_ids)<=records.keys()
        q_tokens=set(terms(query))-DIRECT_STOP
        wanted=plan({"question":query,"qtype":""})
        bodies={}
        for key in source_ids:
            if key in sources and key in tags:
                if key not in base_frames:
                    base_frames[key]=frame(sources[key])
                body,_=project(sources[key],tags[key],base_frames[key],wanted,q_tokens,3050)
            else:
                body=helper.clip(records[key]["text"],3050)
            bodies[key]=body
        samples[qid]=bodies
    args.output.mkdir(parents=True,exist_ok=True)

    def run_arm(arm):
        inputs=[]
        for q in qs:
            key=q["id"]
            order=rankings[key][arm]
            assert len(order)==len(set(order))==10
            docs=[dict(id=i,text=samples[key][i]) for i in order]
            assert sum(len(item["text"].encode()) for item in docs)<=BUDGET
            inputs.append(dict(id=key,question=q["question"],format=q["qtype"],evidence=docs))
        reader_answers, reader_usage=grader.gather(args.cache,f"{args.stage_prefix}-qa-{arm}",TERRA,ANSWER,
                inputs,{"answer":STRING},6,150000,args.workers)
        results=[]
        scored=[]
        for q in qs:
            key,kind=q["id"],q["qtype"]
            prediction=reader_answers[key]["answer"]
            row=dict(id=key,qtype=kind,prediction=prediction,ranked=rankings[key][arm])
            if kind=="number":
                row["accuracy"]=qa.deterministic_accuracy(str(q["answer"]),prediction,q["question"])
            elif kind=="list_recall":
                row["accuracy"]=qa.list_jaccard_score(str(q["answer"]),prediction)
            else:
                scored.append(dict(id=key,instructions=qa.build_judge_prompt(q["question"],str(q["answer"]),prediction)))
            results.append(row)
        grades,judge_usage=grader.gather(args.cache,f"{args.stage_prefix}-judge-{arm}",JUDGE,JUDGE_PROMPT,scored,
                {"accuracy":{"type":"boolean"}},13,87000,args.workers)
        for row in results:
            if row["id"] in grades:
                row["accuracy"]=grades[row["id"]]["accuracy"]
        grader.write_rows(args.output/f"qa-{arm}.jsonl",results)
        answer_metrics=qa.summarize_atm(results)
        retrieval=metrics.measure(qs,list(rankings[q["id"]] for q in qs),(arm,),comprehensive_eval)[arm]
        return dict(qs=answer_metrics["accuracy"],qs_by_qtype=answer_metrics["by_qtype"],
                    recall_at_10=retrieval["recall_at_10"],
                    reader_usage=reader_usage,judge_usage=judge_usage)

    with ThreadPoolExecutor(max_workers=2) as pool:
        evaluated=dict(zip(arms,pool.map(run_arm,arms),strict=True))
    output=dict(n=len(qs),models=dict(reader=TERRA,judge=JUDGE),data_sha256=hashes,
                budget_bytes=BUDGET,metrics=evaluated)
    (args.output/"qa-summary.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","data","official","cache","ranks","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--arm",choices=("phase","fused","corroborated","guarded"),required=True)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--stage-prefix",default="hard-trip")
    main(parser.parse_args())
