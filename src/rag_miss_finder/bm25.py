"""Small BM25 retriever, standard library only."""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been it its this that these those "
    "with as at by from how do does did i we you what which when why can should".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords. Keeps digits and inner hyphens split."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("corpus is empty")
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf = {i: Counter(tokenize(t)) for i, t in docs.items()}
        self.len = {i: sum(c.values()) for i, c in self.tf.items()}
        self.avg = sum(self.len.values()) / len(self.len) or 1.0
        df: Counter[str] = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, doc_id: str) -> float:
        tf, dl = self.tf[doc_id], self.len[doc_id]
        s = 0.0
        for t in set(tokenize(query)):
            f = tf.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def rank(self, query: str) -> list[tuple[str, float]]:
        """All docs, best first. Ties keep corpus order, so results are deterministic."""
        scored = [(i, self.score(query, i)) for i in self.ids]
        return sorted(scored, key=lambda x: -x[1])
