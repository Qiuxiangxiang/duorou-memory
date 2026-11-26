"""Run both complete ATM-Bench splits with source tags and named-event candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import hashlib
import json
from pathlib import Path
from statistics import mean
from subprocess import TimeoutExpired
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"p0-full")]
from run import load_inputs
from cloud import JUDGE,TERRA,batches,call
from evaluate import ANSWER,BUDGET,STRING,batch_usage,read_module,select,usage

ANSWER_PROMPT = ANSWER + (
    " If evidence includes activity tags or event names, use those as search hints and verify against "
    "the actual evidence. Distinguish plans from completed actions, quoted prices from paid amounts, "
    "and recorded media times from actual event times. A dated receipt or email can substantiate an "
    "expense when its transaction and scope match. Use the evidence to determine the requested total."
)
JUDGE_PROMPT = "Apply each item's supplied grading instructions independently. Return one boolean accuracy per id."


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, items):
    with path.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item,ensure_ascii=False)+"\n")


def rewrites(cache, questions):
    wanted={q["id"] for q in questions}
    found={}
    used=[]
    for path in sorted(cache.glob("rewrite-*.json")):
        packet=json.loads(path.read_bytes())
        inside={item["id"]:item["queries"] for item in packet["items"]}
        if not inside.keys()<=wanted:
            continue
        assert not found.keys()&inside.keys()
        found.update(inside)
        used.append(packet["usage"])
    assert set(found)==wanted, f"missing rewrites for {len(wanted-set(found))}"
    return found,usage(used)


def event_candidates(question, names, terms):
    qwords=set(terms(question))
    for title,ids in names.items():
        words=set(terms(title))
        if len(words)>=2 and words<=qwords:
            yield ids[:50]


def get_rankings(args, split, questions, records, tags, editions, changes, rewrites_by_id):
    from duorou import Memory
    from duorou.retrieval import rrf,terms

    target=args.output/f"{split}-rankings.jsonl"
    if target.exists():
        got=rows(target)
        if [r["id"] for r in got]==[q["id"] for q in questions]:
            return got
    output=[]
    with tempfile.TemporaryDirectory() as directory:
        with closing(Memory(Path(directory)/"events.sqlite","benchmark")) as memory:
            for scope in ("source","tagged"):
                memory.provision(scope)
            memory.ingest_many("source",records.values())
            memory.ingest_many("tagged",({**r,"tags":tags[k]["index_terms"]} for k,r in records.items()))
            for n,q in enumerate(questions,1):
                key=q["id"]
                queries=list(dict.fromkeys([q["question"],*rewrites_by_id[key][:2]]))
                queries=[text for text in queries if text.strip() and len(text)<=4096]
                out={"id":key,"qtype":q["qtype"]}
                for scope in ("source","tagged"):
                    paths=[[r["id"] for r in memory.search(scope,phrase,top_k=50,budget=2_000_000)]
                           for phrase in queries]
                    named=list(event_candidates(q["question"],editions,terms)) if scope=="tagged" else []
                    weights=rrf(paths+named)
                    ordered=sorted(weights,key=lambda i:(-weights[i],i))
                    out[scope]=select(ordered,changes)
                    out[scope+"_pool_size"]=len(ordered)
                    assert len(set(out[scope]))==len(out[scope])<=10
                output.append(out)
                if n%100==0 or n==len(questions):
                    print(split,"retrieval",n,"/",len(questions),flush=True)
    write_rows(target,output)
    return output


def gather(cache,stage,model,prompt,items,fields,max_rows,max_bytes,workers):
    groups=list(batches(items,max_rows,max_bytes))
    if not groups:
        return {},usage([])

    def process(group):
        for attempt in range(3):
            try:
                return call(cache,stage,model,prompt,group,fields)
            except (RuntimeError,TimeoutExpired,json.JSONDecodeError) as e:
                if attempt==2:
                    raise
                print(stage,"retry",type(e).__name__,flush=True)

    completed=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(process,g) for g in groups]
        for n,job in enumerate(as_completed(jobs),1):
            completed.append(job.result())
            if n%5==0 or n==len(jobs):
                print(stage,n,"/",len(jobs),flush=True)
    output={row["id"]:row for packet in completed for row in packet["items"]}
    assert set(output)=={row["id"] for row in items}
    return output,usage(batch_usage(completed))


def evidence_items(ids, sources, tags):
    result=[{"id":key,"text":sources[key]["text"]} for key in ids]
    remaining=BUDGET-sum(len(row["text"].encode()) for row in result)
    assert remaining>=0
    for row in result:
        labels=tags[row["id"]]["index_terms"]
        amount=len(json.dumps({"activity_hints":labels},ensure_ascii=False).encode())
        if labels and amount<=remaining:
            row["activity_hints"]=labels
            remaining-=amount
    return result


def score_split(args,split,questions,ranks,clean,tags,rewrite_usage,qa,retrieval):
    print(split,"building answer requests",flush=True)
    ranking={row["id"]:row for row in ranks}
    jobs=[dict(id=q["id"],question=q["question"],format=q["qtype"],
               evidence=evidence_items(ranking[q["id"]]["tagged"],clean,tags)) for q in questions]
    answers,answer_use=gather(args.cache,f"event-full-answer-{split}",TERRA,ANSWER_PROMPT,jobs,
                              {"answer":STRING},10,230000,args.workers)
    outputs=[]
    judge_jobs=[]
    for q in questions:
        key,kind=q["id"],q["qtype"]
        prediction=answers[key]["answer"]
        item=dict(id=key,qtype=kind,prediction=prediction,ranked=ranking[key]["tagged"])
        if kind=="number":
            item["accuracy"]=qa.deterministic_accuracy(str(q["answer"]),prediction,q["question"])
        elif kind=="list_recall":
            item["accuracy"]=qa.list_jaccard_score(str(q["answer"]),prediction)
        else:
            judge_jobs.append(dict(id=key,instructions=qa.build_judge_prompt(
                q["question"],str(q["answer"]),prediction)))
        outputs.append(item)
    judgments,judge_use=gather(args.cache,f"event-full-judge-{split}",JUDGE,JUDGE_PROMPT,judge_jobs,
                               {"accuracy":{"type":"boolean"}},24,87000,args.workers)
    for item in outputs:
        if item["id"] in judgments:
            item["accuracy"]=judgments[item["id"]]["accuracy"]
    qa_summary=qa.summarize_atm(outputs)
    write_rows(args.output/f"{split}-answers.jsonl",outputs)
    report=json.loads((args.output/f"{split}-retrieval.json").read_bytes())
    report.update(qs=qa_summary["accuracy"],by_qtype=qa_summary["by_qtype"],
                  total_tokens=rewrite_usage["total_tokens"]+answer_use["total_tokens"],
                  stages={"frozen_rewrite":rewrite_usage,"answer":answer_use,"judge":judge_use})
    (args.output/f"{split}-summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(split,"complete",report["qs"],report["recall_at_10"],report["total_tokens"],flush=True)
    return report


def score_retrieval(questions, rankings, retrieval):
    aligned=zip(questions,rankings,strict=True)
    pairs=list(aligned)
    assert all(q["id"]==r["id"] for q,r in pairs)
    out={}
    for name in ("source","tagged"):
        dataset=[dict(gt_evidence_ids=q["evidence_ids"],retrieval_ids=r[name]) for q,r in pairs]
        recall=retrieval.compute_metrics(dataset,[10])["metrics"]["recall@10"]
        hits=[(len(set(q["evidence_ids"])&set(r[name])),len(set(q["evidence_ids"]))) for q,r in pairs]
        assert abs(recall-mean(n/d for n,d in hits))<1e-12
        out[name]=dict(recall_at_10=recall,all_hit=sum(n==d for n,d in hits),
                       zero_hit=sum(n==0 for n,d in hits))
    return dict(n=len(questions),recall_at_10=out["tagged"]["recall_at_10"],retrieval=out,
                changed_source_sets=sum(set(r["source"])!=set(r["tagged"]) for _,r in pairs))


def run(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve())]
    from memqa.utils.evaluator import evaluate_qa as qa
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval as retrieval

    args.output.mkdir(parents=True,exist_ok=True)
    records,_,splits,hashes=load_inputs(args.data)
    graph=json.loads((args.graph/"graph-summary.json").read_bytes())
    assert graph["data_sha256"]==hashes
    tags={row["id"]:row for row in rows(args.graph/"sources.jsonl")}
    assert set(tags)==set(records)
    editions=json.loads((args.graph/"edition-sources.json").read_bytes())
    clean=records.copy()
    view=read_module("ocr_cleanup",ROOT/"data-processing/run.py")
    for key,text in view.prepare_views(args.data,records).items():
        clean[key]={**records[key],"text":text}
    status={}
    for split in ("Hard","Main"):
        qs=splits[split]
        plans,rewritten=rewrites(args.cache,qs)
        ranks=get_rankings(args,split,qs,records,tags,editions,clean,plans)
        metrics=score_retrieval(qs,ranks,retrieval)
        (args.output/f"{split}-retrieval.json").write_text(json.dumps(metrics,indent=2)+"\n")
        print(split,"retrieval result",metrics,flush=True)
        status[split]=(qs,ranks,rewritten)
    if args.retrieval_only:
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={name:pool.submit(score_split,args,name,qs,ranks,clean,tags,rew
                               ,qa,retrieval) for name,(qs,ranks,rew) in status.items()}
        results={name:job.result() for name,job in jobs.items()}
    n=sum(r["n"] for r in results.values())
    combined={"n":n,"qs":sum(r["qs"]*r["n"] for r in results.values())/n,
              "recall_at_10":sum(r["recall_at_10"]*r["n"] for r in results.values())/n,
              "total_tokens":sum(r["total_tokens"] for r in results.values()),
              "judge_tokens":sum(r["stages"]["judge"]["total_tokens"] for r in results.values())}
    summary={"graph":graph,"splits":results,"combined":combined,
             "config_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/"benchmark-summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","cache","graph","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--retrieval-only",action="store_true")
    run(parser.parse_args())
