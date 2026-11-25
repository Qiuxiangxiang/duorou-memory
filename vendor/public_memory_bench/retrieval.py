"""Independent lexical retrieval baselines with a separate index per scope.

Only document ``id``, ``scope`` and ``text`` are accepted.  Questions, answers,
gold evidence and answer flags are not part of this interface.  Each search
computes its result afresh; there is no query-result cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


TOKEN_PATTERN = r"(?u)\b\w\w+\b"
BM25_K1 = 1.5
BM25_B = 0.75


def method_spec(method: str) -> dict[str, Any]:
    """Return the fixed, predeclared method specification used by this run."""
    if method not in {"bm25", "tfidf"}:
        raise ValueError("method must be 'bm25' or 'tfidf'")
    shared: dict[str, Any] = {
        "method": method,
        "scope_isolation": "independent vocabulary and document statistics per scope",
        "tokenizer": "sklearn CountVectorizer word analyzer",
        "token_pattern": TOKEN_PATTERN,
        "lowercase": True,
        "stop_words": "sklearn built-in english",
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0,
        "max_features": None,
        "strip_accents": None,
        "numeric_dtype": "float64",
        "query_result_cache": False,
        "return_zero_scores": False,
        "tie_break": "document id ascending",
        "hyperparameter_tuning": False,
    }
    if method == "bm25":
        shared.update({
            "k1": BM25_K1,
            "b": BM25_B,
            "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
            "document_term_weight": "idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))",
            "query_term_weight": "binary; repeated query terms count once",
            "document_length": "count of tokens retained by the shared analyzer",
        })
    else:
        shared.update({
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": False,
            "idf": "log((1 + N) / (1 + df)) + 1",
            "similarity": "cosine of l2-normalized tf-idf vectors",
        })
    return shared


@dataclass(frozen=True)
class _ScopeIndex:
    ids: tuple[str, ...]
    vectorizer: CountVectorizer | None
    transformer: TfidfTransformer | None
    weights: sparse.csr_matrix


def _build_scope(documents: list[tuple[str, str]], method: str) -> _ScopeIndex:
    # Sorted rows make stable row order identical to the declared ID tie-break.
    documents.sort(key=lambda item: item[0])
    ids = tuple(item[0] for item in documents)
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        max_features=None,
        strip_accents=None,
        dtype=np.float64,
    )
    try:
        counts = vectorizer.fit_transform(text for _, text in documents).tocsr()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        return _ScopeIndex(ids, None, None,
                           sparse.csr_matrix((len(ids), 0), dtype=np.float64))

    if method == "tfidf":
        transformer = TfidfTransformer(
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
        )
        weights = transformer.fit_transform(counts).tocsr()
        return _ScopeIndex(ids, vectorizer, transformer, weights)

    document_lengths = np.asarray(counts.sum(axis=1)).ravel()
    average_length = float(document_lengths.mean())
    # CountVectorizer produces one nonzero entry per retained term per row.
    document_frequencies = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log1p((counts.shape[0] - document_frequencies + 0.5) /
                  (document_frequencies + 0.5))
    normalizers = BM25_K1 * (1.0 - BM25_B +
                            BM25_B * document_lengths / average_length)
    weights = counts.copy()
    per_entry_normalizers = np.repeat(normalizers, np.diff(weights.indptr))
    weights.data = (idf[weights.indices] * weights.data * (BM25_K1 + 1.0) /
                    (weights.data + per_entry_normalizers))
    return _ScopeIndex(ids, vectorizer, None, weights)


class ScopedRetriever:
    """Load a JSONL corpus and provide read-only, scoped lexical retrieval.

    A document ID must be globally unique.  Input records must contain exactly
    ``id``, ``scope`` and ``text``.  Empty text is accepted and remains a document
    for corpus statistics, but can never produce a positive lexical score.

    BM25 and TF-IDF use the same unigram analyzer and scope boundaries.  Index
    construction is separate from query timing.  ``build_stats`` reports sparse
    array bytes, not total process RSS or a complete Python-object memory size.
    """

    def __init__(self, documents_path: str | Path, method: str = "bm25") -> None:
        begin = time.perf_counter()
        self.method = method
        self.spec = method_spec(method)
        source = Path(documents_path)
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        seen_ids: set[str] = set()
        digest = hashlib.sha256()
        input_bytes = 0
        text_bytes = 0
        with source.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                input_bytes += len(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"invalid JSON on corpus line {line_number}") from exc
                if not isinstance(row, dict) or set(row) != {"id", "scope", "text"}:
                    raise ValueError(f"corpus line {line_number} must contain only id, scope, text")
                if not all(isinstance(row[key], str) for key in ("id", "scope", "text")):
                    raise ValueError(f"corpus line {line_number} fields must be strings")
                if not row["id"] or not row["scope"]:
                    raise ValueError(f"corpus line {line_number} id and scope must be nonempty")
                if row["id"] in seen_ids:
                    raise ValueError(f"duplicate document id on corpus line {line_number}")
                seen_ids.add(row["id"])
                grouped[row["scope"]].append((row["id"], row["text"]))
                text_bytes += len(row["text"].encode("utf-8"))
        if not grouped:
            raise ValueError("corpus contains no documents")

        self._scopes: dict[str, _ScopeIndex] = {}
        document_counts: list[int] = []
        vocabulary_counts: list[int] = []
        nnz = 0
        sparse_bytes = 0
        # Release each scope's raw text as soon as its index is constructed.
        for scope in sorted(grouped):
            documents = grouped.pop(scope)
            index = _build_scope(documents, method)
            self._scopes[scope] = index
            document_counts.append(len(index.ids))
            vocabulary_counts.append(index.weights.shape[1])
            nnz += int(index.weights.nnz)
            sparse_bytes += int(index.weights.data.nbytes + index.weights.indices.nbytes +
                                index.weights.indptr.nbytes)
            del documents
        self.build_stats: dict[str, Any] = {
            "method": method,
            "build_seconds": time.perf_counter() - begin,
            "corpus_sha256": digest.hexdigest(),
            "corpus_bytes": input_bytes,
            "input_text_utf8_bytes": text_bytes,
            "document_count": sum(document_counts),
            "scope_count": len(self._scopes),
            "documents_per_scope": {
                "minimum": min(document_counts),
                "maximum": max(document_counts),
                "mean": sum(document_counts) / len(document_counts),
            },
            "vocabulary_terms_sum": sum(vocabulary_counts),
            "vocabulary_terms_max_per_scope": max(vocabulary_counts),
            "sparse_nonzero_entries": nnz,
            "sparse_array_bytes": sparse_bytes,
            "size_note": "Sparse matrix arrays only; excludes vocabularies, Python objects, query workspace and runtime libraries.",
            "empty_vocabulary_scopes": sum(count == 0 for count in vocabulary_counts),
            "method_spec": self.spec,
        }

    def search(self, query: str, top_k: int, scope: str) -> list[dict[str, Any]]:
        """Return positive-scoring IDs; no result is retrieved from another scope."""
        if not isinstance(query, str) or not isinstance(scope, str):
            raise TypeError("query and scope must be strings")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        index = self._scopes.get(scope)
        if index is None or index.vectorizer is None:
            return []
        query_counts = index.vectorizer.transform([query]).tocsr()
        if not query_counts.nnz:
            return []
        if self.method == "bm25":
            query_counts.data.fill(1.0)
            query_vector = query_counts
        else:
            assert index.transformer is not None
            query_vector = index.transformer.transform(query_counts)
        product = (index.weights @ query_vector.T).tocoo()
        mask = np.isfinite(product.data) & (product.data > 0.0)
        rows = product.row[mask]
        scores = product.data[mask]
        # Rows are ID-sorted. NumPy sorts positive candidates, not Python texts.
        order = np.lexsort((rows, -scores))[:top_k]
        return [{"id": index.ids[int(rows[i])], "score": float(scores[i])} for i in order]
