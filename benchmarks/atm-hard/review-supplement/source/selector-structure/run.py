"""Rearrange verified source text into role-aware cards under the same byte limit."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import re
from subprocess import TimeoutExpired
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"p0-full")]
from cloud import JUDGE,call
from evaluate import BUDGET,read_module,usage
from run import load_inputs

CAP=1450
ROLE={"direct":3,"partial":2,"context":1}
FIELD={"short_caption":"reported description","short_summary":"reported summary",
       "caption":"reported description","detail":"document text","ocr_text":"visible document text"}


def norm(text):
    return " ".join(re.findall(r"[^\W_]+",text.casefold()))


def quote_sources(current,extra,rechecked):
    accepted=defaultdict(dict)
    for doc,claims in extra.items():
        for claim in claims:
            text=claim["quote"]
            accepted[doc][text]=max(accepted[doc].get(text,0),claim["strength"]-1)
    for line in current:
        doc,text=line["id"],line["quote"]
        accepted[doc][text]=max(accepted[doc].get(text,0),ROLE[line["role"]])
    for line in rechecked:
        doc,text=line["id"],line["quote"]
        if line["score"]>=2:
            accepted[doc][text]=max(accepted[doc].get(text,0),line["score"])
    return accepted


def new_card(src,tag,segments,doc_facts,hints,roles,keywords):
    metadata=f"Recorded time ({tag['recorded_time_kind']}): {tag['recorded_time'] or 'unknown'}\n"
    if src.city:
        metadata+=f"Recorded place: {src.city}\n"
    rows=[]
    roles=set(roles)
    timing=tag.get("other_time_text",{})
    printed=tuple(x for x in timing.get("printed_dates",[])+timing.get("printed_clock_ranges",[]) if x)
    for seg in segments:
        literal=seg.text.strip()
        related=len(seg.words&keywords)
        if not literal:
            continue
        verdict=max((score for quote,score in hints.items() if quote in literal), default=0)
        label=FIELD[seg.field]
        value=min(6,related)*1.05+1.35*verdict
        if seg.field in ("short_summary","short_caption"):
            value+=1.1
        if "timeline" in roles and (any(x in literal for x in printed) or "timeline" in seg.roles):
            label="printed date/time wording"
            value+=1.8
        if "payment_material" in roles:
            payment=[f for f in doc_facts if f["field"]==seg.field and f["quote"] in literal
                     and f["role"]!="unclassified"]
            if payment:
                role=payment[0]["role"]
                label="amount wording: "+role
                value+=3.2 if "paid" in role or "reimbursement" in role else 1.4
            elif "process" in seg.roles:
                label="booking or itinerary wording"
        if seg.field=="detail" and literal.startswith("Subject:"):
            value+=2.5
        if verdict>=2:
            # Surface an exact confirmed quote when the original sentence is long.
            strong=sorted((q for q,s in hints.items() if s==verdict and q in literal),
                          key=lambda q:(-len(q),q))
            value+=.35*bool(strong)
            if strong and len(literal.encode())>400:
                literal=strong[0]
        if len(literal.encode())>505:
            literal=literal.encode()[:505].decode(errors="ignore").rstrip()
        rows.append((value,seg.field,seg.start,label,literal))
    chosen=[]
    allowed=[line for line in metadata.splitlines() if line]
    used=len(metadata.encode())
    seen=[]
    fields=Counter()
    for _,field,offset,label,text in sorted(rows,key=lambda x:(-x[0],x[1],x[2])):
        normalized=norm(text)
        if not normalized or any(normalized in prev or prev in normalized for prev in seen):
            continue
        if fields[field]>=5:
            continue
        line=f"{label}: {text}\n"
        if used+len(line.encode())>CAP:
            continue
        chosen.append(line)
        fields[field]+=1
        used+=len(line.encode())
        allowed.append(text)
        seen.append(normalized)
    return metadata+"".join(chosen),allowed


def compare(cache,row,prompt,fields):
    for attempt in range(3):
        try:
            return call(cache,"hard-role-structured-v1",JUDGE,prompt,[row],fields)
        except (RuntimeError,TimeoutExpired,json.JSONDecodeError):
            if attempt==2:
                raise


def valid_rank(output,allowed,seen,before):
    verdict={}
    for item in output["ranked"][:10]:
        code,quote=item["id"],item["quote"].strip()
        if code in seen and code not in verdict and len(quote)>=9 and quote in seen[code] and (
                any(quote in fragment for fragment in allowed[code]) and item["contribution"].strip()):
            verdict[code]={**item,"quote":quote}
    return list(dict.fromkeys([*verdict,*before]))[:10],verdict


def extra_order(before,ranked,control,proof,narrow,exclusive):
    old=set(before)
    added=[doc for doc in ranked if doc not in old and doc in proof and proof[doc]["priority"]>=2
           and (not exclusive or proof[doc]["quote"] not in narrow[doc])]
    removed=[doc for doc in reversed(before[3:]) if doc not in ranked and doc not in control]
    swaps=dict(zip(removed[:2],added[:2]))
    return [swaps.get(doc,doc) for doc in before],swaps


def main(args):
    sys.path[:0]=[str(args.release.resolve()),str(args.official.resolve()),str(ROOT/"memory-design")]
    from duorou.retrieval import terms
    from memory import DIRECT_STOP,frame,plan,project
    from memqa.utils.evaluator.evaluate_retrieval import comprehensive_eval
    helper=read_module("structure_files",ROOT/"selector-panels/run.py")
    engine=read_module("structure_control",ROOT/"selector-effort/run.py")
    metric=read_module("structure_metrics",ROOT/"selector-corpus/run.py")
    records,kinds,splits,hashes=load_inputs(args.data)
    if hashes!=read_module("structure_pins",ROOT/"iterative-search/search.py").PUBLIC_SHA256:
        raise ValueError("Expected fixed public ATM-Bench data")
    src=read_module("structure_source",ROOT/"event-links/links.py").load_sources(args.data)
    tags=helper.keyed(ROOT/"event-hierarchy/source-tags.jsonl")
    money=helper.keyed(ROOT/"money-facts/source-facts.jsonl")
    earlier=helper.keyed(args.control/"rankings.jsonl")
    occurrences=helper.keyed(ROOT/"selector-answer-trace/rankings.jsonl")
    first=helper.keyed(ROOT/"selector-witness/rankings.jsonl")
    review=helper.keyed(ROOT/"selector-witness/review-rankings.jsonl")
    recheck=helper.keyed(ROOT/"selector-recheck/partial/rankings.jsonl")
    baseline=json.loads((args.control/"summary.json").read_text())
    assert baseline["data_sha256"]==hashes and baseline["model"]==JUDGE
    qs=splits["Hard"]
    frames={};requests=[];data={}
    altered=0
    for q in qs:
        key,query=q["id"],q["question"]
        row=earlier[key];base=row["previous"];ids=row["candidates"]
        assert set(ids)<=records.keys()
        hints=quote_sources(occurrences[key]["partial_statements"],
                         {**first[key]["claims"],**review[key]["newly_quoted"]},recheck[key]["graded"])
        words=set(terms(query))-DIRECT_STOP
        control_plan=plan({"question":query,"qtype":""})
        actual_plan=plan({"question":query,"qtype":q["qtype"]})
        short={};cards={};literals={};flat_literals={}
        for doc in ids:
            if doc not in src or doc not in tags:
                text=helper.clip(records[doc]["text"],CAP)
                short[doc]=text;cards[doc]=text;literals[doc]=[text];flat_literals[doc]=[text]
            else:
                source=src[doc]
                tag=tags[doc]
                if doc not in frames:
                    frames[doc]=frame(source)
                chunks=frames[doc]
                short[doc],spans=project(source,tag,chunks,control_plan,words,CAP)
                flat_literals[doc]=[line for line in short[doc].splitlines()
                                   if line.startswith(("Recorded time","Recorded place"))]
                flat_literals[doc].extend(span["quote"] for span in spans)
                facts=money.get(doc,{}).get("facts",[])
                facts=[f for f in facts if f["field"] in source.fields and
                       source.fields[f["field"]][f["start"]:f["end"]]==f["quote"]]
                cards[doc],literals[doc]=new_card(source,tag,chunks,facts,hints[doc],actual_plan,words)
            assert len(cards[doc].encode())<=CAP and len(short[doc].encode())<=CAP
            altered+=cards[doc]!=short[doc]
        order=engine.choose_order(key,ids)
        control=dict(id=key,question=query,format=q["qtype"],
                     sources=[dict(id=doc,medium=kinds[doc],text=short[doc]) for doc in order])
        cached=call(args.cache,"hard-effort-controlled-low-v1",JUDGE,engine.PROMPT,
                    [control],engine.FIELDS,cached_only=True)
        short_rank,short_claims=engine.read_rank(cached["items"][0],short,base)
        assert short_rank==row["low"]
        assert list(short_claims.values())==row["low_model_evidence"]
        matched_flat,matched_proof=valid_rank(cached["items"][0],flat_literals,short,base)
        requests.append(dict(id=key,question=query,format=q["qtype"],
                     sources=[dict(id=doc,medium=kinds[doc],text=cards[doc]) for doc in order]))
        data[key]=base,ids,short,cards,literals,short_rank,short_claims,matched_flat,matched_proof
    print("Source cards",len(qs),"questions",altered,"restructured question/source records",flush=True)
    if args.prepare_only:
        return
    replies={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(compare,args.cache,row,engine.PROMPT,engine.FIELDS):row["id"] for row in requests}
        for index,future in enumerate(as_completed(futures),1):
            replies[futures[future]]=future.result()
            if index%5==0 or index==len(futures):
                print("Structured Hard",index,"/",len(futures),flush=True)
    traces=[];packets=[];stat=Counter()
    arms=("flat","flat_literal","structured_legacy","structured","agreement","new_quote","any_quote")
    for q in qs:
        key,query=q["id"],q["question"]
        before,ids,short,cards,literals,flat_rank,flat_claims,matched_flat,matched_proof=data[key]
        raw=replies[key]["items"][0]
        structured,proof=valid_rank(raw,literals,cards,before)
        legacy,legacy_proof=engine.read_rank(raw,cards,before)
        agreed=engine.agreement_rank(before,flat_rank,structured,flat_claims,proof)
        new,edits=extra_order(before,structured,flat_rank,proof,short,True)
        known,relax=extra_order(before,structured,flat_rank,proof,short,False)
        new_quotes=[doc for doc,tag in proof.items() if tag["quote"] not in short[doc]]
        gold=set(q["evidence_ids"])
        stat.update(flat_source_bytes=sum(len(source.encode()) for source in short.values()),
                    structured_source_bytes=sum(len(source.encode()) for source in cards.values()),
                    valid_flat_literal_quotes=len(matched_proof),valid_structured_legacy_quotes=len(legacy_proof),
                    valid_quotes=len(proof),novel_quotes=len(new_quotes),
                    novel_gold_quotes=len(set(new_quotes)&gold))
        row=dict(id=key,previous=before,candidates=ids,structured_quotes=list(proof.values()),
                 flat_literal_quotes=list(matched_proof.values()),
                 newly_visible=new_quotes,strict_swaps=[dict(old=a,new=b) for a,b in edits.items()],
                 relaxed_swaps=[dict(old=a,new=b) for a,b in relax.items()])
        evidence={}
        for arm,chosen in zip(arms,(flat_rank,matched_flat,legacy,structured,agreed,new,known),strict=True):
            assert len(chosen)==len(set(chosen))==10 and set(chosen)<=set(ids)
            row[arm]=chosen
            body=short if arm in ("flat","flat_literal") else cards
            excerpts=[dict(id=doc,text=body[doc]) for doc in chosen]
            assert sum(len(x["text"].encode()) for x in excerpts)<=BUDGET
            evidence[arm]=excerpts
        traces.append(row)
        packets.append(dict(id=key,question=query,evidence=evidence))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in (("rankings",traces),("reader-packets",packets)):
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    report=dict(n=len(qs),data_sha256=hashes,evidence_budget_bytes=BUDGET,
        source_max_bytes=CAP,model=JUDGE,restructured_candidates=altered,
        counts=dict(stat),historical_flat_usage=baseline["usage"]["low"],
        new_usage=usage([r["usage"] for r in replies.values()]),
        metrics=metric.measure(qs,traces,("previous",*arms),comprehensive_eval))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","data","official","cache","control","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--workers",type=int,default=11)
    p.add_argument("--prepare-only",action="store_true")
    main(p.parse_args())
