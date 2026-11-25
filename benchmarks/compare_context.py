"""Direct, model-free LoCoMo test of the context index (not MCP latency)."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median, quantiles
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from examples.sqlite_demo import connect_database, import_locomo_records, initialize_schema, LOCOMO_SHA256
from vendor.public_memory_bench.locomo import convert_locomo
from .retriever import DuorouRetriever, source_documents

METHODS = ('bm25_enriched', 'fts_enriched', 'fts_enriched_neighbors', 'fts_enriched_context')


def compare(records, methods=METHODS):
    data = convert_locomo(records, include_captions=True)
    with TemporaryDirectory() as tmp:
        db = Path(tmp) / 'memory.sqlite'
        docs = Path(tmp) / 'documents.jsonl'
        conn = connect_database(db)
        try:
            initialize_schema(conn)
            with conn:
                scopes = import_locomo_records(conn, records, recorded_at_us=1)
                conn.execute("INSERT INTO principals VALUES ('benchmark-reader','service')")
                conn.executemany("INSERT INTO scope_grants VALUES (?,'benchmark-reader','read')", [(s,) for s in scopes])
            docs.write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in source_documents(conn, enriched=True)), encoding='utf-8')
        finally:
            conn.close()
        results, rankings = {}, []
        for method in methods:
            target = DuorouRetriever(db, method, docs if method.startswith('bm25') else None)
            scores = defaultdict(list)
            times = []
            try:
                for q in data['queries']:
                    start = perf_counter_ns()
                    ids = [h['id'] for h in target.search(q['query'],20,q['scope'])]
                    times.append((perf_counter_ns()-start)/1e6)
                    gold = set(q['expected_ids'])
                    category = data['metadata'][q['id']]['category_name']
                    for key in ('overall', category):
                        scores[key].append((len(gold & set(ids[:10]))/len(gold), len(gold & set(ids))/len(gold), int(gold <= set(ids))))
                    rankings.append({'method': method, 'query_id':q['id'], 'ranked_ids':ids, 'latency_ms':times[-1]})
                results[method] = {'metrics': {k: {'n':len(v),'recall10':mean(r[0] for r in v),'recall20':mean(r[1] for r in v),'allhit20':mean(r[2] for r in v)} for k,v in scores.items()},
                    'direct_latency_ms': {'p50':median(times), 'p95':quantiles(times,n=100,method='inclusive')[94] if len(times)>1 else times[0]}, 'index':target.build_stats}
            finally:
                target.close()
        return results, rankings


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locomo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = args.locomo.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCOMO_SHA256:
        p.error('LoCoMo revision hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    scores, rankings = compare(json.loads(raw))
    (args.output/'report.json').write_text(json.dumps({'dataset_sha256': LOCOMO_SHA256, 'mode':'in-process, serial, IDs only', 'methods':scores},indent=2))
    (args.output/'rankings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rankings))
