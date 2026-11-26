from __future__ import annotations

import argparse
import json
import sys

from .analyze import analyze, summarize


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag-miss-finder", description=__doc__)
    p.add_argument("corpus", help="JSONL: {id, text}")
    p.add_argument("queries", help="JSONL: {id, query, relevant: [doc ids]}")
    p.add_argument("-k", type=int, default=3, help="a hit means a relevant doc is in the top k (default 3)")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    a = p.parse_args(argv)
    try:
        docs = {d["id"]: d["text"] for d in load_jsonl(a.corpus)}
        res = analyze(docs, load_jsonl(a.queries), a.k)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    s = summarize(res)
    if a.json:
        print(json.dumps({"summary": s, "results": [r.__dict__ for r in res]}, indent=2))
        return 0
    print(f"queries={s['queries']}  recall@{a.k}={s['recall_at_k']:.2f}  MRR={s['mrr']:.2f}")
    print(f"hit={s['hit']}  ranking-miss={s['ranking-miss']}  vocabulary-gap={s['vocabulary-gap']}\n")
    for r in res:
        if r.verdict != "hit":
            extra = f" shared terms: {', '.join(r.shared_terms) or 'none'}"
            print(f"[{r.verdict}] {r.qid}: {r.query!r} -> wanted {r.relevant}{extra}")
    if s["vocabulary-gap"]:
        print("\nvocabulary-gap = the question and the right document share no scored words."
              " Tuning BM25 will not fix these; query rewriting or embeddings might.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
