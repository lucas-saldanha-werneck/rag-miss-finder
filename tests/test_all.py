import json
import pytest

from rag_miss_finder.analyze import analyze, summarize
from rag_miss_finder.bm25 import BM25, tokenize
from rag_miss_finder.cli import main

DOCS = {
    "a": "rollback the deployment to the previous release",
    "b": "rotate api keys every ninety days",
    "c": "nightly backup of the database",
}


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("How do I ROLL back the Release?") == ["roll", "back", "release"]


def test_bm25_ranks_matching_doc_first():
    assert BM25(DOCS).rank("rotate keys")[0][0] == "b"


def test_bm25_zero_score_without_shared_terms():
    assert BM25(DOCS).score("banana smoothie", "a") == 0.0


def test_bm25_ties_are_deterministic():
    r1 = BM25(DOCS).rank("banana")
    r2 = BM25(DOCS).rank("banana")
    assert r1 == r2 and [d for d, _ in r1] == ["a", "b", "c"]


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25({})


def test_hit_and_vocabulary_gap():
    qs = [
        {"id": "1", "query": "rollback release", "relevant": ["a"]},
        {"id": "2", "query": "banana smoothie", "relevant": ["a"]},
    ]
    res = analyze(DOCS, qs, k=1)
    assert [r.verdict for r in res] == ["hit", "vocabulary-gap"]
    assert res[1].best_rank is None


def test_ranking_miss_when_terms_shared_but_outranked():
    docs = {"x": "database database database backup", "y": "restore the database from a snapshot"}
    res = analyze(docs, [{"id": "1", "query": "database backup database", "relevant": ["y"]}], k=1)
    assert res[0].verdict == "ranking-miss"
    assert res[0].shared_terms == ["database"]


def test_unknown_doc_id_raises():
    with pytest.raises(ValueError):
        analyze(DOCS, [{"id": "1", "query": "x", "relevant": ["zzz"]}])


def test_summarize_numbers():
    qs = [
        {"id": "1", "query": "rollback release", "relevant": ["a"]},
        {"id": "2", "query": "banana", "relevant": ["a"]},
    ]
    s = summarize(analyze(DOCS, qs, k=1))
    assert s["queries"] == 2 and s["hit"] == 1 and s["vocabulary-gap"] == 1
    assert s["recall_at_k"] == 0.5 and s["mrr"] == 0.5


def test_cli_on_example_data_json(capsys):
    assert main(["data/example_corpus.jsonl", "data/example_queries.jsonl", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["queries"] == 14
    assert out["summary"]["hit"] + out["summary"]["ranking-miss"] + out["summary"]["vocabulary-gap"] == 14


def test_cli_missing_file_returns_2(capsys):
    assert main(["nope.jsonl", "nope.jsonl"]) == 2
    assert "error" in capsys.readouterr().err
