"""Local full-text retrieval load smoke test; synthetic, closed loop, no MCP."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import tempfile
import time
import statistics
import resource
from duorou import Memory
from .v1_eval import percentile


def run(n=5000,requests=1000):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'memory.db';m=Memory(path,'bench');m.provision('demo')
        start=time.perf_counter();m.ingest_many('demo',[{'source_id':str(i),'text':f'Equipment UID{i:06d} has inspection result {i%71} and scheduled maintenance.','session':f'batch-{i//10}'} for i in range(n)])
        ingest=time.perf_counter()-start
        start=time.perf_counter();m.search('demo','UID000001',top_k=5);cold=(time.perf_counter()-start)*1000
        runs=[]
        for workers in (1,8):
            for repeat in range(3):
                def query(i):
                    ident=(i*7919+repeat*7)%n;start=time.perf_counter()
                    hits=m.search('demo',f'UID{ident:06d}',top_k=5)
                    return (time.perf_counter()-start)*1000,str(ident) in {x['id'] for x in hits}
                start=time.perf_counter()
                with ThreadPoolExecutor(max_workers=workers) as pool: rows=list(pool.map(query,range(requests)))
                duration=time.perf_counter()-start;lat=[r[0] for r in rows]
                runs.append({'concurrency':workers,'repeat':repeat+1,'requests':requests,'qps':requests/duration,'hit_rate':statistics.mean(r[1] for r in rows),'p50_ms':percentile(lat,.5),'p95_ms':percentile(lat,.95),'p99_ms':percentile(lat,.99)})
        start=time.perf_counter();m.ingest('demo','new','Unprecedented visibility marker uniquenewbeacon');write_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();visible=any(r['id']=='new' for r in m.search('demo','uniquenewbeacon'));first_read_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();m.delete('demo','new');delete_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();gone=not any(r['id']=='new' for r in m.search('demo','uniquenewbeacon'));first_after_delete_ms=(time.perf_counter()-start)*1000
        m.close()
        return {'kind':'synthetic closed-loop in-process test (not saturation or MCP)', 'documents':n,'ingest_seconds':ingest,'cold_first_search_ms':cold,'db_bytes_after_close':path.stat().st_size,'runs':runs,'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'visibility':{'write_ms':write_ms,'first_query_after_write_ms':first_read_ms,'found':visible,'delete_ms':delete_ms,'first_query_after_delete_ms':first_after_delete_ms,'absent':gone}}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--documents',type=int,default=5000);p.add_argument('--requests',type=int,default=1000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.documents<2 or a.requests<1:p.error('positive documents/requests required')
    result=run(a.documents,a.requests);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
