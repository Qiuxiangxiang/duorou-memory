"""Replay fixed archived public-data rankings; not a fresh retrieval test."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean
from .fusion import fuse

ARMS = ('bm25_enriched', 'fts_enriched', 'fts_enriched_neighbors')


def session(doc):
    # Only for LoCoMo archival IDs, never for production identity inference.
    return re.sub(r':\d+$', '', doc)


def replay(root):
    runs = {arm: {r['query_id']: r for r in map(json.loads, (root / f'{arm}.predictions.jsonl').read_text().splitlines())} for arm in ARMS}
    ids = set(runs[ARMS[0]])
    if not all(set(run) == ids for run in runs.values()):
        raise ValueError('mismatched queries')
    metrics = defaultdict(list)
    for qid in sorted(ids):
        rows = [runs[arm][qid] for arm in ARMS]
        gold = set(rows[0]['gold_ids'])
        scope = rows[0]['scope']
        if not gold or any(r['status'] != 'success' or set(r['gold_ids']) != gold or r['scope'] != scope for r in rows):
            raise ValueError('mismatched ground truth or failed request')
        ranks = [r['ranked_ids'][:20] for r in rows]
        if any(not d.startswith(scope + ':') for rank in ranks for d in rank):
            raise ValueError('unexpected scope')
        group = {d: session(d) for rank in ranks for d in rank}
        outputs = dict(zip(ARMS, ranks))
        for name, lists, penalty in [('rrf_direct', ranks[:2], 0), ('rrf_all', ranks, 0), ('rrf_diverse', ranks, .1)]:
            outputs[name] = [h['id'] for h in fuse(lists, 20, groups=group, diversity=penalty)]
        outputs['candidate_union_upper_bound'] = list(set().union(*map(set, ranks)))
        for name, ranking in outputs.items():
            retrieved = set(ranking)
            recall = len(gold & retrieved) / len(gold)
            allhit = int(gold <= retrieved)
            for bucket in ('overall', rows[0]['category']):
                metrics[(name, bucket)].append((recall, allhit, len(set(map(session, ranking)))))
    return {name: {bucket: {'n': len(values), 'recall': mean(x[0] for x in values), 'all_hit': mean(x[1] for x in values), 'mean_sessions': mean(x[2] for x in values)} for (method, bucket), values in metrics.items() if method == name} for name, _ in metrics}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('quality', type=Path)
    args = p.parse_args()
    print(json.dumps(replay(args.quality), indent=2))
