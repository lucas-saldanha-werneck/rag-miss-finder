"""Classify each labeled query: hit, ranking miss, or vocabulary gap."""
from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25, tokenize


@dataclass
class QueryResult:
    qid: str
    query: str
    relevant: list[str]
    verdict: str  # "hit" | "ranking-miss" | "vocabulary-gap"
    best_rank: int | None  # 1-based rank of best relevant doc; None if it has no score at all
    shared_terms: list[str] = field(default_factory=list)


def analyze(docs: dict[str, str], queries: list[dict], k: int = 3) -> list[QueryResult]:
    bm = BM25(docs)
    out = []
    for q in queries:
        rel = list(q["relevant"])
        unknown = [r for r in rel if r not in docs]
        if unknown:
            raise ValueError(f"query {q['id']}: unknown doc ids {unknown}")
        ranked = bm.rank(q["query"])
        pos = {d: i + 1 for i, (d, _) in enumerate(ranked)}
        score = dict(ranked)
        best = min(pos[r] for r in rel)
        qterms = set(tokenize(q["query"]))
        shared = sorted(qterms & set().union(*(bm.tf[r].keys() for r in rel)))
        if best <= k and any(score[r] > 0 for r in rel if pos[r] <= k):
            verdict = "hit"
        elif not shared or all(score[r] == 0 for r in rel):
            verdict = "vocabulary-gap"
        else:
            verdict = "ranking-miss"
        scored_pos = [pos[r] for r in rel if score[r] > 0]
        out.append(QueryResult(q["id"], q["query"], rel, verdict,
                               min(scored_pos) if scored_pos else None, shared))
    return out


def summarize(results: list[QueryResult]) -> dict:
    n = len(results)
    counts = {"hit": 0, "ranking-miss": 0, "vocabulary-gap": 0}
    for r in results:
        counts[r.verdict] += 1
    rr = [1 / r.best_rank for r in results if r.best_rank]
    return {
        "queries": n,
        **counts,
        "recall_at_k": counts["hit"] / n if n else 0.0,
        "mrr": sum(rr) / n if n else 0.0,
    }
