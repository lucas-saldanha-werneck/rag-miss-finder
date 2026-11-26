# rag-miss-finder

Independent demo, not affiliated with CI&T. It was written as a small side project after I applied for an AI engineer role, to show how I look at retrieval quality.

When a RAG system gives a bad answer, the first question is where it failed. This tool answers one part of that for a keyword retriever (BM25). You give it a document set and a list of questions, each labeled with the document that should answer it. For every question that misses, it says which of two things happened:

- **ranking-miss**: the question and the right document share scored words, but other documents ranked higher. Tuning or reranking can help.
- **vocabulary-gap**: they share no scored words at all. No BM25 setting fixes this. Query rewriting or embeddings are the next thing to try.

That split tells you whether it is worth trying vector search before you build it.

## Run it

Needs Python 3.10+ and [uv](https://docs.astral.sh/uv/). There are no runtime dependencies.

```
uv sync
uv run rag-miss-finder data/example_corpus.jsonl data/example_queries.jsonl
uv run rag-miss-finder data/example_corpus.jsonl data/example_queries.jsonl -k 1 --json
uv run pytest -q
```

Input files are JSONL. Corpus lines look like `{"id": "d01", "text": "..."}`. Query lines look like `{"id": "q01", "query": "...", "relevant": ["d01"]}`.

The files in `data/` are invented example data (15 short made-up internal-wiki notes, 14 questions). They are not from any real company.

## Example output

```
queries=14  recall@3=0.64  MRR=0.64
hit=9  ranking-miss=0  vocabulary-gap=5

[vocabulary-gap] q09: 'who gets woken up at night when the site is failing' -> wanted ['d04'] shared terms: none
[vocabulary-gap] q10: 'can I download my information' -> wanted ['d11'] shared terms: none
...
```

In this example set, all 5 misses are vocabulary gaps and none are ranking misses. The 8 questions that reuse the documents' own words all hit, plus one paraphrase that still shared a word ("tokens"/"secrets"). I wrote the 5 paraphrased questions to test this, so the result shows the tool works. It says nothing about real-world rates.

## What was tested

- 11 automated tests, all offline (`uv run pytest -q`): tokenizing, BM25 ranking, deterministic tie order, empty corpus, hit / ranking-miss / vocabulary-gap classification, unknown document ids, summary numbers, and the command line on the example data and on a missing file.
- One bug turned up while writing the tests: a question with no matching words was counted as a hit, because ties kept corpus order. A test now covers it.
- The same tests run in GitHub Actions on every push.

## What was not tested

- Real data. Only the invented example set was used.
- Large corpora. BM25 here scores every document for every query, so it is slow past a few thousand documents.
- Languages other than English. The tokenizer and stopword list are English only. No stemming, so "rotated" and "rotate" count as different words.
- Whether embeddings actually fix the vocabulary gaps. The tool points at that question and does not answer it.
- Documents with more than one relevant answer were only lightly tested.
