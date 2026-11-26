"""Create partial answers with source citations and rank their source-level evidence."""
import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
from math import ceil, log1p
from pathlib import Path
import re
import sys
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from evaluate import BUDGET,read_module
from run import load_inputs

READ="""Answer as much of this question as the supplied original records establish. Split the
partial answer into distinct answer items or concrete necessary premises. Include exact named
items, dates or transaction facts when given. Explicitly distinguish reported from scheduled
dates and actual observed visits from plans, transfers and document timestamps. A media-ID
question treats each original that directly matches the requested scene as a separate item.
Do not assume a source proves other evidence in the group. An exact origin or return date may
be a useful premise even if it cannot answer the whole question. For each grounded statement
give its answer item, and 1–2 original IDs with an exact short excerpt from each. Use
direct when an item is directly established, partial for an exact incomplete premise, and
context when useful to disambiguate. Do not infer missing evidence. Produce up to 15 concrete
statements. Use an empty list when the group has no grounded evidence."""
CITATION={"type":"object","additionalProperties":False,
          "required":["id","quote"],"properties":{"id":{"type":"string"},"quote":{"type":"string"}}}
READ_FIELDS={"statements":{"type":"array","items":{"type":"object","additionalProperties":False,
             "required":["statement","item","role","citations"],"properties":{
                "statement":{"type":"string"},"item":{"type":"string"},
                "role":{"type":"string","enum":["direct","partial","context"]},
                "citations":{"type":"array","items":CITATION}}}}}
MERGE="""Reconstruct a single partial answer from the evidence statements and ORIGINAL
source excerpts. Their previous interpretation is provisional: compare each to the original
excerpt and question. Use only IDs from the supplied sources. Then choose exactly ten
original IDs that together would best let an independent reader verify the answer.
Prefer different answer-bearing facts and, where needed, distinct cities, dates, payments,
observations or named items. For a list of matching source IDs, every genuinely matching
original is an answer item. Keep complementary sources that establish a distinct fact
even if another record is more obviously on topic. A booking or document date cannot by
itself establish actual visits; separate observed presence, scheduled route, and underlying
travel endpoints. For payment distinguish an actual distinct transaction from many views
of the same one. Avoid treating nearby or similarly named records as evidence without
visible support. Return the evidence ranking, original quote and exact contribution per
ID. Include weak context only if there are not ten better originals. Do not output the answer."""
MERGE_FIELDS={"ranked":{"type":"array","items":{"type":"object","additionalProperties":False,
              "required":["id","quote","contribution","priority"],"properties":{
                "id":{"type":"string"},"quote":{"type":"string"},"contribution":{"type":"string"},
                "priority":{"type":"integer","enum":[1,2,3]}}}}}
ROLE={"direct":3,"partial":2,"context":1}


def canonical(text):
    return " ".join(re.findall(r"[^\W_]+",text.casefold()))


def join(*streams):
    return list(dict.fromkeys(x for row in zip_longest(*streams) for x in row if x))


def accepted(output, shown, exact_ids):
    out=[]
    for group in output["statements"][:15]:
        item=canonical(group["item"])[:155]
        if not item or not group["statement"].strip():
            continue
        for link in group["citations"][:2]:
            key,quote=link["id"],link["quote"]
            if key in shown and len(quote.strip())>=9 and quote in shown[key]:
                out.append(dict(id=key,item=key if exact_ids else item,statement=group["statement"],
                                quote=quote,role=group["role"]))
    return out


def score_order(before,pool,statements,prior,retain):
    by_id=defaultdict(dict)
    for claim in statements:
        key,item=claim["id"],claim["item"]
        current=by_id[key].get(item,0)
        by_id[key][item]=max(current,ROLE[claim["role"]])
    available=list(dict.fromkeys([*before,*by_id]))
    old={key:n for n,key in enumerate(before)}
    pos={key:n for n,key in enumerate(pool)}
    covered={}
    out=[]
    while available and len(out)<10:
        def utility(key):
            facts=by_id[key]
            direct=max(facts.values(),default=0)
            gains=sum(max(0,value-covered.get(item,0)) for item,value in facts.items())
            p=retain-.030*old[key] if key in old else 0
            return .69*direct + .25*gains + .16*log1p(prior[key]) + p

        chosen=min(available,key=lambda key:(-utility(key),pos[key],key))
        out.append(chosen)
        available.remove(chosen)
        for item,value in by_id[chosen].items():
            covered[item]=max(value,covered.get(item,0))
    return out


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from cloud import TERRA,JUDGE
    from duorou.retrieval import terms
    from memory import DIRECT_STOP,frame,plan,project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval

    helper=read_module("trace_helper",ROOT/"selector-panels/run.py")
    grader=read_module("trace_grader",ROOT/"event-graph-full/benchmark.py")
    metric=read_module("trace_metrics",ROOT/"selector-corpus/run.py")
    src=read_module("trace_sources",ROOT/"event-links/links.py").load_sources(args.data)
    records,kind,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("trace_data",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected fixed public ATM-Bench data")
    qs=splits["Hard"]
    state=helper.keyed(ROOT/"selector-lanes/episode-rankings.jsonl")
    witness=helper.keyed(ROOT/"selector-witness/rankings.jsonl")
    review=helper.keyed(ROOT/"selector-witness/review-rankings.jsonl")
    panel=helper.keyed(ROOT/"selector-panels/rankings.jsonl")
    trip=helper.keyed(ROOT/"selector-trip/rankings.jsonl")
    geography=helper.keyed(ROOT/"selector-geography/pools.jsonl")
    scope=helper.keyed(ROOT/"selector-scope/rankings.jsonl")
    tags=helper.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    frames={}
    requests=[]
    setups={}
    mapping={}
    for q in qs:
        key,query=q["id"],q["question"]
        first=witness[key]
        before=state[key]["episode_guard"]
        proofs={**first["claims"],**review[key]["newly_quoted"]}
        proof_ids=sorted(proofs,key=lambda k:(-max(ROLE.get(r.get("role"),r["strength"]-1)
                                                    for r in proofs[k]),k))
        other=trip[key].get("candidates",[])
        broad=join(panel[key]["shortlist"],proof_ids,other,geography[key]["extra"],scope[key]["scope_cross"])
        ids=helper.union(before,broad)[:100]
        assert set(ids)<=records.keys()
        votes=Counter()
        for order in (before,first["guarded"],review[key]["review"],panel[key]["after"],trip[key]["fused"]):
            votes.update(order)
        words=set(terms(query))-DIRECT_STOP
        required=plan({"question":query,"qtype":""})
        shown={}
        for source in ids:
            if source not in src or source not in tags:
                shown[source]=helper.clip(records[source]["text"],900)
            else:
                if source not in frames:
                    frames[source]=frame(src[source])
                shown[source]=project(src[source],tags[source],frames[source],required,words,900)[0]
        setups[key]=(before,ids,shown,votes)
        n=ceil(len(ids)/35)
        for i in range(n):
            group=ids[i::n]
            label=f"{key}:{i+1}"
            mapping[label]=(key,{doc:shown[doc] for doc in group})
            requests.append(dict(id=label,question=query,format=q["qtype"],
                                 sources=[dict(id=doc,kind=kind[doc],text=shown[doc]) for doc in group]))
    print("Partial answer groups",len(requests),"sources",sum(len(x["sources"]) for x in requests),flush=True)
    if args.prepare_only:
        return
    partial,use_read=grader.gather(args.cache,"hard-answer-trace-partial-v1",TERRA,READ,requests,
                                    READ_FIELDS,4,147000,args.workers)
    q_by_id={q["id"]:q for q in qs}
    claims=defaultdict(list)
    for ident,result in partial.items():
        qid, shown=mapping[ident]
        claims[qid].extend(accepted(result,shown,q_by_id[qid]["qtype"]=="list_recall"))
    drafts=[]
    shortlists={}
    for q in qs:
        qid,query=q["id"],q["question"]
        before,ids,shown,votes=setups[qid]
        cited={x["id"] for x in claims[qid]}
        provisional=sorted(cited,key=lambda key:(-max(ROLE[x["role"]] for x in claims[qid] if x["id"]==key),
                                                -votes[key],ids.index(key)))
        shortlist=helper.union(before,provisional)[:54]
        shortlists[qid]=shortlist
        if len(shortlist)>10:
            drafts.append(dict(id=qid,question=query,format=q["qtype"],
                               statements=[x for x in claims[qid] if x["id"] in shortlist],
                               sources=[dict(id=x,kind=kind[x],text=shown[x])
                                        for x in shortlist[::2]+shortlist[1::2]]))
    print("Merged answer groups",len(drafts),"sources",sum(len(x["sources"]) for x in drafts),flush=True)
    merged,use_merge=grader.gather(args.cache,"hard-answer-trace-merge-v1",JUDGE,MERGE,drafts,
                                    MERGE_FIELDS,3,150000,args.workers)

    traces=[]
    packets=[]
    pool_scores=defaultdict(list)
    arms=("claims","merged","guarded")
    for q in qs:
        key,query=q["id"],q["question"]
        before,ids,shown,votes=setups[key]
        proofs=[]
        for row in merged.get(key,{"ranked":[]})["ranked"][:10]:
            source,quote=row["id"],row["quote"]
            if source in shortlists[key] and source not in {p["id"] for p in proofs} and len(quote.strip())>=9 and quote in shown[source]:
                proofs.append(row)
        proposal=[r["id"] for r in proofs]
        output_model=helper.union(proposal,before)[:10]
        independent=score_order(before,ids,claims[key],votes,1.90)
        trusted={x["id"] for x in claims[key] if x["role"]=="direct"}
        consensus=helper.union([x for x in before if x in proposal],
                               [r["id"] for r in proofs if r["priority"]>=2 and r["id"] in trusted],
                               before)[:10]
        trace=dict(id=key,previous=before,claims=independent,merged=output_model,guarded=consensus,
                   candidates=ids,shortlist=shortlists[key],partial_statements=claims[key],final_quotes=proofs)
        snippets={}
        for arm in arms:
            order=trace[arm]
            assert len(order)==len(set(order))==10 and set(order)<=set(ids)
            docs=[dict(id=i,text=shown[i]) for i in order]
            assert sum(len(x["text"].encode()) for x in docs)<=BUDGET
            snippets[arm]=docs
        gold=set(q["evidence_ids"])
        for label,pool in (("candidates",ids),("cited_or_prior",shortlists[key])):
            pool_scores[label].append((len(gold&set(pool)),len(gold)))
        traces.append(trace)
        packets.append(dict(id=key,question=query,evidence=snippets))
    args.output.mkdir(parents=True,exist_ok=True)
    grader.write_rows(args.output/"rankings.jsonl",traces)
    grader.write_rows(args.output/"reader-packets.jsonl",packets)
    stat=dict(n=len(qs),data_sha256=hashes,evidence_budget_bytes=BUDGET,
              first_group_count=len(requests),merged_group_count=len(drafts),
              confirmed_partial_statements=sum(map(len,claims.values())),
              read_usage=use_read,merge_usage=use_merge,
              diag={name:dict(gold=sum(n for n,d in scores),max_recall_at_10=mean(min(n,10)/d for n,d in scores))
                    for name,scores in pool_scores.items()},
              metrics=metric.measure(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(stat,indent=2)+"\n")
    print(json.dumps(stat),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","data","official","cache","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--workers",type=int,default=10)
    parser.add_argument("--prepare-only",action="store_true")
    main(parser.parse_args())
