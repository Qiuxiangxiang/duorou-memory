"""Index what each original source can prove before seeing any evaluation question."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from evaluate import BUDGET,read_module
from run import load_inputs

FIELD_BUDGETS={"timestamp":115,"location_name":215,"city":100,"short_summary":470,
               "short_caption":370,"ocr_text":850,"detail":1050,"caption":620,
               "duration_formatted":60}
FIELD=re.compile(r"(?m)^(timestamp|location_name|city|short_summary|short_caption|ocr_text|detail|caption|duration_formatted|tags|entities): ")
PROMPT="""Independently index each supplied original archive record. You have no target query.
Identify up to three specific facts or narrow lookups this source itself can answer, such
as a named place and date, observed activity, hotel, transport endpoint, speaker, title or
stated transaction fact. Write a short natural lookup phrase or question and an exact
contiguous quote from this record supporting it. Name the fact without adding details.
Distinguish a capture timestamp from a documented date and a booking from payment or an
observed stay. Do not infer a different event or entity, or assume who appears in an image.
If there is no meaningful fact, use an empty list. Produce one index entry per original ID."""
FIELDS={"facts":{"type":"array","items":{"type":"object","additionalProperties":False,
    "required":["lookup","fact","quote","role"],"properties":{
        "lookup":{"type":"string"},"fact":{"type":"string"},"quote":{"type":"string"},
        "role":{"type":"string","enum":["observed","documented","scheduled","paid","identifying","other"]}}}}}


def prefix(text,limit):
    return text.encode()[:limit].decode(errors="ignore")


def blind_excerpt(text):
    segments=list(FIELD.finditer(text))
    rows=[text.splitlines()[0]]
    for i,m in enumerate(segments):
        key=m[1]
        if key not in FIELD_BUDGETS:
            continue
        end=segments[i+1].start() if i+1<len(segments) else len(text)
        value=prefix(text[m.end():end].strip(),FIELD_BUDGETS[key])
        if value:
            rows.append(f"{key}: {value}")
    return "\n".join(rows)


def normalize(s):
    return " ".join(re.findall(r"[^\W_]+",s.casefold()))


def validate(content,items):
    facts=[]
    used=set()
    for row in items[:3]:
        quote=row["quote"].strip()
        lookup=normalize(row["lookup"])
        if len(quote)<9 or quote not in content or not lookup or lookup in used or not row["fact"].strip():
            continue
        used.add(lookup)
        facts.append({**row,"quote":quote})
    return facts


def order(ids,score):
    return sorted(ids,key=lambda key:(-score[key],key))


def rank_score(ids,scores):
    return {k:16/(16+n) for n,k in enumerate(order(ids,scores),1)}


def confirmed_grades(older,more,check,answers):
    result={}
    for key,rows in {**older,**more}.items():
        result[key]=max(row["strength"]-1 for row in rows)
    for row in check:
        key=row["id"]
        result[key]=max(result.get(key,0),row["score"])
    for row in answers:
        key=row["id"]
        result[key]=max(result.get(key,0),{"direct":3,"partial":2,"context":1}[row["role"]])
    return result


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from cloud import TERRA
    from duorou.retrieval import BM25,terms
    from memory import DIRECT_STOP,frame,plan,project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    host=read_module("inverse_helpers",ROOT/"selector-panels/run.py")
    runner=read_module("inverse_index_model",ROOT/"event-graph-full/benchmark.py")
    metrics=read_module("inverse_metrics",ROOT/"selector-corpus/run.py")
    pretrained=read_module("inverse_model_pin",ROOT/"selector-pretrained/run.py")
    reranker=read_module("inverse_reranker",ROOT/"rerank/run.py")
    records,kinds,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("inverse_pins",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected hash-pinned public ATM-Bench inputs")
    compact=host.keyed(ROOT/"selector-union/rankings.jsonl")
    witness=host.keyed(ROOT/"selector-witness/rankings.jsonl")
    review=host.keyed(ROOT/"selector-witness/review-rankings.jsonl")
    recheck=host.keyed(ROOT/"selector-recheck/partial/rankings.jsonl")
    answers=host.keyed(ROOT/"selector-answer-trace/rankings.jsonl")
    earlier=host.keyed(ROOT/"selector-pretrained/rankings.jsonl")
    tags=host.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    src=read_module("inverse_sources",ROOT/"event-links/links.py").load_sources(args.data)
    qs=splits["Hard"]
    data={}
    collection=set()
    for q in qs:
        key=q["id"]
        before=compact[key]["previous"]
        grades=confirmed_grades(witness[key]["claims"],review[key]["newly_quoted"],
                                 recheck[key]["graded"],answers[key]["partial_statements"])
        extra=sorted((doc for doc in grades if grades[doc]>=2 and doc not in set(compact[key]["candidates"])),
                     key=lambda doc:(-grades[doc],doc))
        pool=host.union(before,compact[key]["candidates"],extra)
        assert set(pool)<=records.keys()
        data[key]=before,pool,grades
        collection.update(pool)
    # Send source-only records. The target question, answer and relevance labels are absent.
    cards={key:blind_excerpt(records[key]["text"]) for key in sorted(collection)}
    inputs=[dict(id=key,source_type=kinds[key],evidence=raw) for key,raw in cards.items()]
    print("Source-only fact index",len(inputs),"original records",flush=True)
    if args.prepare_only:
        return
    response,bill=runner.gather(args.cache,"hard-inverse-source-blind-v1",TERRA,PROMPT,inputs,
                                FIELDS,args.batch_size,165000,args.workers)
    facts={doc:validate(cards[doc],hit["facts"]) for doc,hit in response.items()}
    args.output.mkdir(parents=True,exist_ok=True)
    runner.write_rows(args.output/"source-index.jsonl",
                      [dict(id=doc,facts=facts[doc]) for doc in sorted(facts)])
    print("Sources with literal facts",sum(bool(value) for value in facts.values()),flush=True)
    model=reranker.CrossEncoder(args.model)
    if model.hashes!=pretrained.SHA256:
        raise ValueError("Unexpected local scoring model files")
    frames={}
    traces=[]
    packets=[]
    arms=("literal","inverse","lexical","mixed","guarded")
    start=perf_counter()
    for i,q in enumerate(qs,1):
        key,query=q["id"],q["question"]
        before,ids,grade=data[key]
        short={}
        qterms=set(terms(query))-DIRECT_STOP
        requirements=plan({"question":query,"qtype":""})
        for doc in ids:
            if doc in src and doc in tags:
                if doc not in frames:
                    frames[doc]=frame(src[doc])
                short[doc]=project(src[doc],tags[doc],frames[doc],requirements,qterms,1450)[0]
            else:
                short[doc]=prefix(records[doc]["text"],1450)
        view={doc:"\n".join(f"{row['lookup']} Evidence: {row['fact']}. {row['quote']}"
                           for row in facts[doc]) or short[doc] for doc in ids}
        base={p["id"]:p["mixedbread"] for p in earlier[key]["scores"]}
        new=[doc for doc in ids if doc not in base]
        if new:
            base.update(dict(zip(new,map(float,model.score(query,[short[doc] for doc in new])),strict=True)))
        inverse=dict(zip(ids,map(float,model.score(query,[view[doc] for doc in ids])),strict=True))
        rawlex=BM25(view).search(query,len(ids))
        lex=host.union(rawlex,ids)
        a,b=rank_score(ids,base),rank_score(ids,inverse)
        la={k:16/(16+j) for j,k in enumerate(lex,1)}
        mixed={doc:.40*a[doc]+.42*b[doc]+.18*la[doc] for doc in ids}
        old={doc:18/(18+j) for j,doc in enumerate(before,1)}
        allowed=[doc for doc in ids if doc in old or
                 (grade.get(doc,0)>=2 and b[doc]>=16/35 and a[doc]>=16/45)]
        safe={doc:.70*old.get(doc,0)+.31*a[doc]+.51*b[doc]+.10*la[doc]+.22*grade.get(doc,0)
              for doc in allowed}
        row=dict(id=key,previous=before,candidates=ids,
                 scores=[dict(id=doc,literal=base[doc],inverse=inverse[doc]) for doc in ids])
        packet={}
        for arm,choices in zip(arms,(order(ids,base),order(ids,inverse),lex,order(ids,mixed),
                                   order(allowed,safe)),strict=True):
            top=choices[:10]
            assert len(top)==len(set(top))==10 and set(top)<=set(ids)
            row[arm]=top
            docs=[dict(id=doc,text=short[doc]) for doc in top]
            assert sum(len(d["text"].encode()) for d in docs)<=BUDGET
            packet[arm]=docs
        traces.append(row)
        packets.append(dict(id=key,question=query,evidence=packet))
        if i%5==0 or i==len(qs):
            print("Indexed Hard",i,"/",len(qs),flush=True)
    for name,rows in (("rankings",traces),("reader-packets",packets)):
        runner.write_rows(args.output/f"{name}.jsonl",rows)
    stat=dict(n=len(qs),data_sha256=hashes,model_sha256=model.hashes,evidence_budget_bytes=BUDGET,
        initialization_usage=bill,question_time_model_api_tokens=0,
        original_sources=len(inputs),source_facts=sum(map(len,facts.values())),
        source_entries_with_facts=sum(bool(v) for v in facts.values()),
        scored_pairs=sum(len(r["candidates"]) for r in traces),
        scoring_seconds=perf_counter()-start,
        metrics=metrics.measure(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(stat,indent=2)+"\n")
    print(json.dumps(stat),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("release","data","official","cache","model","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=20)
    p.add_argument("--prepare-only",action="store_true")
    main(p.parse_args())
