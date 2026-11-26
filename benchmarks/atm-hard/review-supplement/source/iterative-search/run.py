"""Hard-only public benchmark for local pool expansion and three-round retrieval."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full"),str(ROOT/"money-facts")]
from run import load_inputs
from cloud import TERRA
from evaluate import BUDGET,read_module,usage
from facts import load_facts
from search import (MAX_ROUNDS,PUBLIC_SHA256,ROUND_FIELDS,ROUND_PROMPT,accept_findings,
                    execute,neighborhood,neighbors_of,prefix,read_rows,round_robin,slots_from,union)

REVIEW_ADD="""The recorded findings have verbatim source quotes. Keep them available and focus
on adding directly useful missing facts. Newly found sources are candidates, including nearby
observations. Verify that any recommendation is in the requested context. Keep distinct media
when IDs are requested. Do not replace unique already-supported facts or treat similar charges
as the same transaction from location and time alone."""


def keyed(path):
    return {row["id"]:row for row in read_rows(path)}


def metrics(questions,traces,key,cap=None):
    hits=[]
    for q,row in zip(questions,traces,strict=True):
        gold=set(q["evidence_ids"])
        n=len(gold&set(row[key]))
        hits.append((min(n,cap) if cap else n,len(gold)))
    return {"recall":mean(n/d for n,d in hits),"gold_items":sum(n for n,_ in hits),
            "all_hit":sum(n==d for n,d in hits)}


def run(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import BM25,terms
    from memqa.utils.evaluator import evaluate_qa as qa
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval as retrieval
    from memory import DIRECT_STOP,frame,plan,project,source_views

    args.output.mkdir(parents=True,exist_ok=True)
    records,_,splits,hashes=load_inputs(args.data)
    if hashes!=PUBLIC_SHA256:
        raise ValueError("Cloud inputs must match the pinned public ATM-Bench files")
    qs=splits["Hard"]
    prior=keyed(args.baseline/"rankings.jsonl")
    existing=keyed(args.pool/"rankings.jsonl")
    assert len(qs)==31 and set(prior)==set(existing)=={q["id"] for q in qs}
    tags=keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    index_tags=keyed(args.graph/"sources.jsonl")
    assert set(tags)==set(index_tags)==set(records)
    graph_report=json.loads((args.graph/"graph-summary.json").read_bytes())
    assert graph_report["data_sha256"]==hashes
    sources=read_module("source_loader",ROOT/"event-links/links.py").load_sources(args.data)
    cleanup=read_module("ocr_cleanup",ROOT/"data-processing/run.py")
    grader=read_module("benchmark",args.graph/"benchmark.py")
    gaps=read_module("gap_validator",ROOT/"selector-gaps/run.py")
    tail=read_module("tail_validator",ROOT/"selector-review/run.py")
    unified=read_module("unified_reader",ROOT/"unified-replay/run.py")
    payment=read_module("payment_reader",ROOT/"payment-confirmation/run.py")
    differences=read_module("differences",ROOT/"unified-replay/comparisons.py")
    diff_rows={(x["a"],x["b"]):x for x in differences.read_rows(ROOT/"unified-replay/candidate-differences.jsonl")}
    views=source_views(sources,index_tags,load_facts(args.data),cleanup.clean_ocr)
    clean=cleanup.prepare_views(args.data,records)
    views["full"]={i:clean.get(i,r["text"]) for i,r in records.items()}
    indices={k:BM25(v) for k,v in views.items() if v}
    edges=neighborhood(args.graph)

    frames={}
    states={}
    keys={q["id"]:q for q in qs}
    words={q["id"]:set(terms(q["question"]))-DIRECT_STOP for q in qs}
    goals={q["id"]:plan(q) for q in qs}

    def get_text(key,src):
        state=states[key]
        cache=state["texts"]
        if src not in cache:
            if src not in frames:
                frames[src]=frame(sources[src])
            cache[src]=project(sources[src],tags[src],frames[src],goals[key],words[key],3050)[0]
        return cache[src]

    for q in qs:
        key=q["id"]
        before=prior[key]["adaptive"]
        base=existing[key]["expanded_pool"]
        assert set(before)<=set(base)<=set(records)
        phrases=union([q["question"]],prior[key]["queries"][:3])
        lexical=round_robin([indices[field].search(query,18)
                             for query in phrases for field in indices
                             if field!="amount" or "payment_material" in goals[key]],base,24)
        near_paths=[neighbors_of(src,edges) for src in before]
        nearby=round_robin(near_paths,base,20)
        slots=slots_from(prior[key]["plan"],q)
        shown=union(before,lexical[:4],nearby[:4])
        assert set(shown)<=set(records)
        states[key]=dict(before=before,base=base,local_field=lexical,local_near=nearby,
                         shown=set(shown),fresh=shown,open=slots.copy(),ledger=[],history=set(),rounds=[],texts={})

    offline=[]
    for q in qs:
        key=q["id"]
        st=states[key]
        offline.append(dict(id=key,base=st["base"],field=union(st["base"],st["local_field"]),
                            neighbors=union(st["base"],st["local_near"]),
                            combined=union(st["base"],st["local_field"],st["local_near"])))
    local={k:metrics(qs,offline,k) for k in ("base","field","neighbors","combined")}
    grader.write_rows(args.output/"local-pools.jsonl",offline)
    print("Public Hard local pools",local,flush=True)

    search_usage=[]
    for number in range(1,MAX_ROUNDS+1):
        prompts=[]
        shown={}
        for q in qs:
            key=q["id"]
            st=states[key]
            if not st["fresh"] or not st["open"]:
                continue
            shown[key]={src:prefix(get_text(key,src),1050) for src in st["fresh"]}
            # Locked facts are summarized without resending their quoted document.
            prompts.append(dict(id=key,question=q["question"],open=list(st["open"].values()),
                locked=[{k:r[k] for k in ("slot_id","fact","source_id")} for r in st["ledger"][:8]],
                new_sources=[{"id":s,"text":t} for s,t in shown[key].items()]))
        answered,used=grader.gather(args.cache,f"hard-public-iterative-{number}",TERRA,ROUND_PROMPT,
                                   prompts,ROUND_FIELDS,6,165000,args.workers)
        search_usage.append(used)
        for key,row in answered.items():
            st=states[key]
            accepted=accept_findings(row["findings"],st,shown[key],tail.has_quote)
            found,actions=execute(row["searches"],st,shown[key],indices,edges,tail.has_quote)
            assert set(found)<=set(records)
            st["rounds"].append(dict(round=number,findings=accepted,retrieved=found,actions=actions,
                                     unresolved=list(st["open"])))
            st["shown"].update(found)
            st["fresh"]=found
        print("Hard round",number,len(prompts),"active questions",sum(len(
              st["rounds"][-1]["retrieved"]) for st in states.values()
              if st["rounds"] and st["rounds"][-1]["round"]==number),flush=True)

    traces=[]
    review_prompts=[]
    review_context={}
    for q,original in zip(qs,offline,strict=True):
        key=q["id"]
        st=states[key]
        returned=union(*(r["retrieved"] for r in st["rounds"]))
        evidence_pool=union(original["combined"],returned)
        saved=union([item["source_id"] for item in st["ledger"]])
        choices=[src for src in union(saved,returned,st["local_field"],st["local_near"])
                 if src not in st["before"]]
        wanted=gaps.roles_for(q)
        features={src:gaps.candidate_features(get_text(key,src),wanted,
                    {w:1.0 for w in sorted(words[key]) if len(w)>2},terms,i)
                  for i,src in enumerate(st["before"]+choices)}
        selected=[src for src in saved if src in choices][:4]
        selected+=gaps.shortlist([src for src in choices if src not in selected],features,
                                 wanted,set(),8-len(selected))
        shown={src:gaps.evidence_excerpt(src,features,950) for src in st["before"]+selected}
        review_context[key]=(selected,features,shown)
        if selected:
            review_prompts.append(dict(id=key,question=q["question"],format=q["qtype"],
                   protected_ids=st["before"][:3],remaining=list(st["open"].values()),
                   saved_findings=st["ledger"][:12],
                   current=[{"id":src,"text":shown[src]} for src in st["before"]],
                   shortlist=[{"id":src,"text":shown[src]} for src in selected]))
        traces.append(dict(**original,before=st["before"],pool=evidence_pool,
                           locked=st["ledger"],rounds=st["rounds"],shortlist=selected))
    grader.write_rows(args.output/"search-trace.jsonl",traces)
    selected,selector_cost=grader.gather(args.cache,"hard-public-iterative-tail",TERRA,
         tail.REVIEW_PROMPT+"\n"+gaps.MORE_RULES+"\n"+REVIEW_ADD,review_prompts,tail.FIELDS,7,170000,args.workers)

    packets=[]
    for q,row in zip(qs,traces,strict=True):
        key=q["id"]
        st=states[key]
        ids,features,shown=review_context[key]
        response=selected.get(key,{"unfilled":[],"replacements":[]})
        after,checks=gaps.apply(q,st["before"],ids,st["texts"],shown,features,response,tail)
        row["after"]=after
        row["selection"]=checks
        packet={"id":key,"question":q["question"],"format":q["qtype"],
                "evidence":[{"id":src,"text":get_text(key,src)} for src in after]}
        expense=bool(q["qtype"]=="number" and unified.PAY_REQUEST.search(q["question"]))
        cards=differences.difference_cards(packet,diff_rows,expense,unified)
        if cards:
            packet["related_source_differences"]=cards
        size=sum(len(item["text"].encode()) for item in packet["evidence"])+(
            unified.json_bytes({"related_source_differences":cards}) if cards else 0)
        assert size<=BUDGET and len(after)<=10 and set(after)<=set(row["pool"])
        packets.append((packet,expense))
    grader.write_rows(args.output/"rankings.jsonl",traces)
    stage_results=[]
    rolling=[dict(row,stage=row["combined"]) for row in traces]
    for number in range(1,MAX_ROUNDS+1):
        for row in rolling:
            row["stage"]=union(row["stage"],*(r["retrieved"] for r in row["rounds"] if r["round"]==number))
        stage_results.append(metrics(qs,rolling,"stage"))
    moves=Counter()
    for q,row in zip(qs,traces,strict=True):
        gold=set(q["evidence_ids"])
        before,after=set(row["before"]),set(row["after"])
        moves.update({
            "top10_gold_added":len(gold&(after-before)),"top10_gold_lost":len(gold&(before-after)),
            "top10_gold_from_new_pool":len((after-set(row["base"]))&gold),
            "saved_findings":len(row["locked"]),"closed_slots":sum(x["closed"] for x in row["locked"]),
        })
    summary=dict(n=len(qs),data_sha256=hashes,offline=local,round_pools=stage_results,
                 bound_before=metrics(qs,traces,"base",10),bound_after=metrics(qs,traces,"pool",10),
                 top10_before=metrics(qs,traces,"before"),top10_after=metrics(qs,traces,"after"),
                 changes=dict(moves),phase_tokens=search_usage,
                 search_tokens=usage(search_usage),selection_tokens=selector_cost)
    (args.output/"retrieval-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Hard final retrieval",summary["bound_after"],summary["top10_after"],flush=True)
    if not args.retrieval_only:
        prev=json.loads((args.pool/"summary.json").read_bytes())["results"]["multifield"]
        reader=unified.score(args,"Hard-iterative",qs,packets,prev["stages"]["frozen_planning"],
                             None,None,grader,payment,qa,retrieval)
        assert abs(reader["recall_at_10"]-summary["top10_after"]["recall"])<1e-12
        summary["reader"]=reader
        summary["total_tokens"]=reader["total_tokens"]+summary["search_tokens"]["total_tokens"]+selector_cost["total_tokens"]
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","cache","pool","baseline","graph","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--retrieval-only",action="store_true")
    run(parser.parse_args())
