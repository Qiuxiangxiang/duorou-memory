"""Uniform evidence benchmark: exact source IDs, scope, budgets and actual returned text.

Public inputs: vendor converters' source-only documents and separate queries.
Synthetic fixture is explicitly not a public benchmark. Timing is in-process.
"""
import argparse
from collections import defaultdict
import hashlib
import math
import json
from pathlib import Path
import platform
import random
import re
import statistics
import tempfile
import time

from duorou import Memory


def jsonl(path):
    with open(path,encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path, rows):
    with open(path,'w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def percentile(values,p):
    values=sorted(values)
    return values[min(len(values)-1,max(0,math.ceil(len(values)*p)-1))] if values else None


def summarize(rows):
    positive=[r for r in rows if r['recall'] is not None]
    negative=[r for r in rows if r['recall'] is None]
    times=[r['latency_ms'] for r in rows]
    return {'queries':len(rows),'positive_queries':len(positive),'negative_queries':len(negative),
            'errors':sum(bool(r['error']) for r in rows),
            'recall':statistics.mean(r['recall'] for r in positive) if positive else None,
            'all_hit':statistics.mean(r['all_hit'] for r in positive) if positive else None,
            'precision':statistics.mean(r['precision'] for r in positive) if positive else None,
            'empty_correct':statistics.mean(not r['ranked_ids'] and not r['error'] for r in negative) if negative else None,
            'latency_ms':{f'p{int(p*100)}':percentile(times,p) for p in (.5,.95,.99)},
            'mean_evidence_cost':statistics.mean(r['cost'] for r in rows) if rows else None}


def score(case,ids,corpus,cost,latency,error=None):
    gold=set(case['expected_ids']); found=gold.intersection(ids)
    if len(ids)!=len(set(ids)) or any(i not in corpus or corpus[i]['scope']!=case['scope'] for i in ids):
        raise ValueError('invalid or cross-scope result')
    return {'query_id':case['id'],'scope':case['scope'],'category':case.get('category','unspecified'),
            'ranked_ids':ids,'gold_ids':sorted(gold),'recall':len(found)/len(gold) if gold else None,
            'all_hit':float(found==gold and not error) if gold else None,
            'precision':len(found)/len(ids) if ids else 0.0,
            'cost':sum(cost(corpus[i]['text']) for i in ids),'latency_ms':latency,'error':error}


def budget_rank(ids,corpus,k,budget,cost):
    selected=[]; remaining=budget
    for i in ids:
        if len(selected)==k: break
        size=cost(corpus[i]['text'])
        if size<=remaining: selected.append(i);remaining-=size
    return selected


def validate(documents,cases):
    corpus={d['id']:d for d in documents}
    if len(corpus)!=len(documents) or any(set(d)!={'id','scope','text'} or not all(isinstance(x,str) for x in d.values()) for d in documents):
        raise ValueError('documents must be unique, source-only id/scope/text records')
    if len({q['id'] for q in cases})!=len(cases): raise ValueError('duplicate query ID')
    for q in cases:
        if any(i not in corpus or corpus[i]['scope']!=q['scope'] for i in q['expected_ids']): raise ValueError('bad gold mapping')
    return corpus


def source_session(d):
    # Source-derived only. Never infer session from gold/question metadata.
    if d['scope'].startswith('locomo:') and re.search(r':D\d+:\d+$',d['id']):
        return d['id'].rsplit(':',1)[0]
    return None


def run(documents,cases,destination,*,k=20,budget=32768,cost=None,unit='utf8_evidence_bytes',embed=None,kind='public',reference=True):
    cost=cost or (lambda t:len(t.encode('utf-8')))
    corpus=validate(documents,cases);destination.mkdir(parents=True,exist_ok=True)
    docfile=destination/'documents.jsonl';dump(docfile,documents)
    byscope=defaultdict(list); queries=defaultdict(list)
    for d in documents: byscope[d['scope']].append(d)
    for q in cases: queries[q['scope']].append(q)
    methods=['bm25','enriched','hybrid']+(['dense'] if embed else [])
    if reference:
        from vendor.public_memory_bench.retrieval import ScopedRetriever
        start=time.perf_counter();baseline=ScopedRetriever(docfile,'bm25');base_build=time.perf_counter()-start
        methods.insert(0,'reference_bm25')
    results={}; timings={}
    with tempfile.TemporaryDirectory() as tmp:
        m=Memory(Path(tmp)/'memory.db','bench',embed=embed,token_count=cost,cache_scopes=2)
        start=time.perf_counter()
        for scope,docs in byscope.items():
            m.provision(scope)
            m.ingest_many(scope,[dict(source_id=d['id'],text=d['text'],session=source_session(d)) for d in docs])
        ingest=time.perf_counter()-start;index_build=0
        allrows={method:[] for method in methods}
        for scope,qs in queries.items():
            start=time.perf_counter()
            with m.tx(): m._allow(scope);m._index(scope)
            index_build+=time.perf_counter()-start
            for q in qs:
                for method in methods:
                    start=time.perf_counter(); error=None; ids=[]
                    try:
                        if method=='reference_bm25':
                            candidates=baseline.search(q['query'],max(80,k*5),scope)
                            ids=budget_rank([r['id'] for r in candidates],corpus,k,budget,cost)
                        else:
                            hits=m.search(scope,q['query'],k,budget,method)
                            ids=[r['id'] for r in hits]
                            if any(r['text']!=corpus[r['id']]['text'] for r in hits): raise ValueError('source text mismatch')
                        row=score(q,ids,corpus,cost,(time.perf_counter()-start)*1000)
                        if len(ids)>k or row['cost']>budget: raise ValueError('budget violation')
                    except Exception as exc:
                        row=score(q,[],corpus,cost,(time.perf_counter()-start)*1000,type(exc).__name__+': '+str(exc))
                    allrows[method].append(row)
        for method,rows in allrows.items():
            dump(destination/(method+'.predictions.jsonl'),rows)
            categories=defaultdict(list)
            for row in rows: categories[row['category']].append(row)
            results[method]={'overall':summarize(rows),'categories':{key:summarize(rs) for key,rs in categories.items()}}
        m.close()
    # Do not ship original licensed corpus in reports; input hash identifies it.
    digest=hashlib.sha256(docfile.read_bytes()).hexdigest();docfile.unlink()
    report={'dataset_kind':kind,'documents':len(documents),'queries':len(cases),'scopes':len(byscope),
            'documents_projection_sha256':digest,'top_k':k,'budget':budget,'budget_unit':unit,
            'platform':platform.platform(),'python':platform.python_version(),'protocol':'in-process, warmed scope index, no MCP transport',
            'ingest_seconds':ingest,'duorou_index_build_seconds':index_build,'reference_index_build_seconds':base_build if reference else None,
            'semantic_encoder_used':bool(embed),'results':results}
    (destination/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def compare_external(documents,cases,predictions,*,k=20,budget=32768,cost=None):
    """Score an external memory backend with identical evidence IDs and budget.

    IDs must correspond to actually delivered source texts; callers must ensure
    the backend did not merely return pointers to unreturned children. This
    helper checks IDs/duplicates/scope/size, not external runtime authenticity.
    """
    cost=cost or (lambda t:len(t.encode('utf-8')))
    corpus=validate(documents,cases)
    pred={p['query_id']:p for p in predictions}
    if len(pred)!=len(predictions) or set(pred)!={q['id'] for q in cases}:
        raise ValueError('external predictions require exactly one record per query')
    rows=[]
    for q in cases:
        item=pred[q['id']]; ids=item['ranked_ids']
        latency=item.get('latency_ms')
        if not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency<0:
            raise ValueError('external predictions must include observed nonnegative latency_ms')
        row=score(q,ids,corpus,cost,latency,item.get('error'))
        if len(ids)>k or row['cost']>budget: raise ValueError('external budget violation')
        rows.append(row)
    return summarize(rows),rows


def fixture():
    """Generated stress corpus, not a research benchmark or held-out real data."""
    rng=random.Random(92); docs=[]; qs=[]
    for j in range(8):
        scope=f'locomo:fixture-{j}'
        for s in range(24):
            ident=f'K{j:02d}X{s:02d}'; part=rng.choice(['controller','sensor','valve','motor'])
            code=f'ZX{rng.randrange(100000,999999)}'; deadline=rng.choice(['Tuesday','Wednesday','Friday'])
            texts=[f'Project {ident} selected a {part} for the pilot.',f'The part shipment has tracking code {code}.',
                   f'The installation meeting is {deadline}.', 'The team reviewed an unrelated maintenance checklist.']
            ids=[f'{scope}:D{s+1}:{n+1}' for n in range(len(texts))]
            docs.extend({'id':key,'scope':scope,'text':txt} for key,txt in zip(ids,texts))
            qs.extend([{'id':ident+'-1','scope':scope,'query':f'Which part was selected for {ident}?','expected_ids':[ids[0]],'category':'single'},
                       {'id':ident+'-2','scope':scope,'query':f'For project {ident}, what is its shipment tracking code?','expected_ids':[ids[0],ids[1]],'category':'local_context'}])
        qs.append({'id':scope+'-empty','scope':scope,'query':'qzxmnonexistentidentifier','expected_ids':[],'category':'empty'})
    return docs,qs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--documents',type=Path);p.add_argument('--queries',type=Path)
    p.add_argument('--metadata',type=Path);p.add_argument('--fixture',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--top-k',type=int,default=20);p.add_argument('--budget',type=int,default=32768)
    p.add_argument('--embedding-model');p.add_argument('--tokenizer',help='local HuggingFace tokenizer directory; budget counts each returned evidence text, not prompt framing')
    p.add_argument('--no-reference',action='store_true');p.add_argument('--external-rankings',type=Path);a=p.parse_args()
    if a.fixture:
        docs,qs=fixture();kind='synthetic fixture'
    elif a.documents and a.queries:
        docs,qs=jsonl(a.documents),jsonl(a.queries);kind='provided corpus'
    else: p.error('use --fixture or both --documents and --queries')
    if a.metadata:
        meta=json.loads(a.metadata.read_text())
        for q in qs: q['category']=meta.get(q['id'],{}).get('category_name',meta.get(q['id'],{}).get('category','unspecified'))
    embed=cost=None;unit='utf8_evidence_bytes'
    if a.embedding_model:
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(a.embedding_model,local_files_only=True)
        embed=lambda texts:model.encode(texts,batch_size=32,show_progress_bar=False)
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
        cost=lambda t:len(tok.encode(t,add_special_tokens=False));unit='sum_of_evidence_text_tokens'
    if a.external_rankings:
        summary,rows=compare_external(docs,qs,jsonl(a.external_rankings),k=a.top_k,budget=a.budget,cost=cost)
        a.output.mkdir(parents=True,exist_ok=True);dump(a.output/'external.predictions.jsonl',rows)
        (a.output/'summary.json').write_text(json.dumps({'protocol':'external reported timings','budget':a.budget,'budget_unit':unit,'top_k':a.top_k,'overall':summary},indent=2)+'\n')
        print(json.dumps(summary,indent=2));return
    report=run(docs,qs,a.output,k=a.top_k,budget=a.budget,cost=cost,unit=unit,embed=embed,kind=kind,reference=not a.no_reference)
    print(json.dumps({k:v['overall'] for k,v in report['results'].items()},indent=2))

if __name__=='__main__':main()
