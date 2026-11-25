"""Scope-local inverted indexes, optional dense retrieval, bounded evidence selection."""
from collections import Counter, defaultdict
from math import log
from heapq import nsmallest
import re

STOP = frozenset('a an the of to in on at for is are was were be been being do does did has have had and or but what which when where how who whom why it its they them their he she his her i me my you your we our this that these those with from as by about can could would should'.split())
CJK = re.compile(r'[\u3400-\u9fff\uf900-\ufaff]')


def terms(text):
    words = []
    for word in re.findall(r'[^\W_]+', text.casefold()):
        if CJK.search(word):
            for part in re.findall(r'[\u3400-\u9fff\uf900-\ufaff]+|[^\u3400-\u9fff\uf900-\ufaff]+', word):
                if CJK.search(part):
                    words.extend(part)
                    words.extend(part[i:i+2] for i in range(len(part)-1))
                elif part not in STOP:
                    words.append(part)
        elif word not in STOP:
            words.append(word)
    return words


class BM25:
    def __init__(self, texts):
        counts = {key: Counter(terms(text)) for key, text in texts.items()}
        lengths = {key: sum(c.values()) for key, c in counts.items()}
        avg = sum(lengths.values()) / max(1, len(counts)) or 1
        postings = defaultdict(dict)
        for key, counter in counts.items():
            norm = 1.5 * (.25 + .75 * lengths[key] / avg)
            for term, tf in counter.items():
                postings[term][key] = 2.5 * tf / (tf + norm)
        n = len(counts)
        self.postings = {}
        for t,p in postings.items():
            idf = log(1 + (n-len(p)+.5)/(len(p)+.5))
            self.postings[t] = {d:w*idf for d,w in p.items()}

    def search(self, query, limit):
        scores = defaultdict(float)
        for term in dict.fromkeys(terms(query)):
            for key, weight in self.postings.get(term, {}).items():
                scores[key] += weight
        return nsmallest(limit,scores,key=lambda d:(-scores[d],d))


def rrf(rankings):
    scores = defaultdict(float)
    for ranking in rankings:
        for i, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1/(60+i)
    return dict(scores)


class Index:
    def __init__(self, rows, groups, links, embed=None):
        self.rows, self.groups, self.links = rows, groups, links
        self.lexical = BM25({k:r['text'] for k,r in rows.items()})
        self.enriched = BM25({k:r['key'] for k,r in rows.items()})
        context = {}
        self.members = defaultdict(list)
        for k,r in rows.items():
            self.members[groups.get(k,k)].append(k)
        for members in self.members.values():
            for i,k in enumerate(members):
                context[k] = ' '.join(rows[j]['key'] for j in members[max(0,i-1):i+2])
        self.context = BM25(context)
        self.vectors, self.embed = {}, embed
        if embed and rows:
            import numpy as np
            self.vector_ids = list(rows)
            v = np.asarray(embed([rows[k]['key'] for k in self.vector_ids]), dtype='float32')
            if v.ndim != 2 or len(v) != len(rows) or not np.isfinite(v).all():
                raise ValueError('invalid document embeddings')
            self.vectors = v / np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)

    def search(self, query, k, budget, allowed, mode='hybrid', cost=None):
        if not allowed:
            return []
        n = min(len(self.rows), max(80,k*5))
        def eligible(ids):
            return [i for i in ids if i in allowed][:n]
        raw = eligible(self.lexical.search(query,n))
        enriched = lambda: eligible(self.enriched.search(query,n))
        def dense():
            if not self.embed:
                raise ValueError('dense mode requires an explicit embedding encoder')
            if not len(self.vectors):
                return []
            import numpy as np
            q = np.asarray(self.embed([query]),dtype='float32')
            if q.shape != (1,self.vectors.shape[1]) or not np.isfinite(q).all():
                raise ValueError('invalid query embedding')
            sim = self.vectors @ (q[0]/max(float(np.linalg.norm(q)),1e-12))
            return eligible([self.vector_ids[i] for i in np.argsort(-sim,kind='stable') if sim[i]>0])
        if mode in ('bm25','enriched','dense'):
            rank = raw if mode=='bm25' else enriched() if mode=='enriched' else dense()
            scores = rrf([rank])
        else:
            rankings = [raw,enriched(),eligible(self.context.search(query,n))]
            if self.embed:
                rankings.append(dense())
            scores = rrf(rankings)
            # One bounded hop through source-grounded entities/events/claim relations.
            base = sorted(scores,key=lambda d:(-scores[d],d))[:min(12,k)]
            for seed in base:
                for j in self.links.get(seed, ())[:8]:
                    if j in allowed and j != seed:
                        scores[j] = max(scores.get(j,0), scores[seed]*.25)
        # Keep high-ranking evidence; modest coverage penalty only for hybrid mode.
        chosen, counts = [], Counter()
        used = 0
        while scores and len(chosen)<k:
            d = min(scores,key=lambda d:(-scores[d]/(1+(.08 if mode=='hybrid' else 0)*counts[self.groups.get(d,d)]),d))
            score = scores.pop(d)
            row = self.rows[d]
            size = cost(row['text']) if cost else len(row['text'].encode('utf-8'))
            if used+size>budget:
                continue
            used += size
            counts[self.groups.get(d,d)] += 1
            chosen.append(dict(row['result'],score=score))
        return chosen
