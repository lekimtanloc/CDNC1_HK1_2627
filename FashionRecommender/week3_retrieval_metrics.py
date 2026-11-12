"""Ranking metrics for the frozen same-item, different-view retrieval protocol."""

from __future__ import annotations

import math
from typing import Iterable


def score_query(ranked_gallery: Iterable[str], relevant_gallery: set[str], k: int) -> dict[str, float]:
    """Score one query. Ranked IDs must be unique gallery images, not item IDs."""
    if k < 1 or not relevant_gallery:
        raise ValueError("k must be positive and every query needs a relevant gallery image")
    ranked = list(ranked_gallery)
    if len(ranked) != len(set(ranked)):
        raise ValueError("Duplicate gallery image in a ranking")
    hits = [int(path in relevant_gallery) for path in ranked[:k]]
    recall = sum(hits) / len(relevant_gallery)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant_gallery))))
    first = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    return {"recall": recall, "ndcg": dcg / ideal, "reciprocal_rank": 1 / first if first else 0.0}


def evaluate(
    rankings: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int,
) -> dict[str, float]:
    """Macro-average over every ground-truth query; missing rankings score zero."""
    if not ground_truth:
        raise ValueError("Ground truth cannot be empty")
    if extra := set(rankings) - set(ground_truth):
        raise ValueError(f"Rankings contain unknown queries: {sorted(extra)[:3]}")
    scores = []
    for query, relevant in ground_truth.items():
        ranked = rankings.get(query, [])
        if query in ranked:
            raise ValueError(f"Query image returned as its own result: {query}")
        scores.append(score_query(ranked, relevant, k))
    return {name: sum(row[name] for row in scores) / len(scores) for name in scores[0]}
