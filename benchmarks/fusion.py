"""Small, deterministic ranking operators. Never accept evaluator labels."""
from collections import Counter, defaultdict


def fuse(rankings, limit=20, *, groups=None, diversity=0.0):
    """RRF with optional diminishing return for repeatedly selected groups.

    Each list contributes once per ID. Missing group membership is unique.
    The returned score is RRF (not a probability); selection uses a penalty.
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(dict.fromkeys(ranking), 1):
            scores[doc] += 1 / (60 + rank)
    if not diversity:
        return [{"id": d, "score": scores[d]} for d in sorted(scores, key=lambda d: (-scores[d], d))[:limit]]
    groups = groups or {}
    chosen, counts = [], Counter()
    for _ in range(min(limit, len(scores))):
        doc = min(scores, key=lambda d: (-scores[d] / (1 + diversity * counts[groups.get(d, d)]), d))
        chosen.append({'id': doc, 'score': scores.pop(doc)})
        counts[groups.get(doc, doc)] += 1
    return chosen
