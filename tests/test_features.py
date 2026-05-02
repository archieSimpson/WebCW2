"""Tests for the BM25 ranker, the avgdl cache, and snippet generation."""

import os
import tempfile

from src.indexer import Indexer
from src.search import SearchEngine


def make_engine(*pages):
    indexer = Indexer()
    for url, html in pages:
        indexer.index_page(url, html)
    return SearchEngine(indexer), indexer


# ---------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------


class TestBM25:

    def setup_method(self):
        self.indexer = Indexer()

    def test_bm25_zero_for_missing_word(self):
        self.indexer.index_page("http://a.com", "<p>hello</p>")
        assert self.indexer.get_bm25("missing", "http://a.com") == 0.0

    def test_bm25_zero_for_missing_url(self):
        self.indexer.index_page("http://a.com", "<p>hello</p>")
        assert self.indexer.get_bm25("hello", "http://b.com") == 0.0

    def test_bm25_zero_when_no_documents(self):
        # avgdl == 0 path
        assert self.indexer.get_bm25("anything", "http://nope.com") == 0.0

    def test_bm25_is_positive_for_present_term(self):
        self.indexer.index_page("http://a.com", "<p>hello world</p>")
        assert self.indexer.get_bm25("hello", "http://a.com") > 0

    def test_bm25_higher_for_higher_frequency(self):
        self.indexer.index_page(
            "http://a.com",
            "<p>fish fish fish fish water sea coral reef</p>",
        )
        self.indexer.index_page(
            "http://b.com",
            "<p>fish water sea coral reef shell rock</p>",
        )
        assert (
            self.indexer.get_bm25("fish", "http://a.com")
            > self.indexer.get_bm25("fish", "http://b.com")
        )

    def test_bm25_rare_word_outscores_common(self):
        for i in range(10):
            self.indexer.index_page(f"http://d{i}.com", "<p>the cat</p>")
        self.indexer.index_page(
            "http://r.com", "<p>the serendipity</p>"
        )
        common = self.indexer.get_bm25("the", "http://r.com")
        rare = self.indexer.get_bm25("serendipity", "http://r.com")
        assert rare > common

    def test_bm25_saturates_with_repeated_term(self):
        # BM25 saturates: doubling the count must NOT double the score.
        self.indexer.index_page(
            "http://a.com", "<p>" + "fish " * 2 + "water</p>"
        )
        self.indexer.index_page(
            "http://b.com", "<p>" + "fish " * 200 + "water</p>"
        )
        s_small = self.indexer.get_bm25("fish", "http://a.com")
        s_big = self.indexer.get_bm25("fish", "http://b.com")
        # 100x more occurrences should not yield 100x more score.
        assert s_big < 100 * s_small

    def test_bm25_length_normalisation(self):
        # Shorter doc with the same TF should outscore a longer one.
        short = "<p>fish water</p>"
        long_ = "<p>fish " + "filler " * 50 + "water</p>"
        self.indexer.index_page("http://short.com", short)
        self.indexer.index_page("http://long.com", long_)
        assert (
            self.indexer.get_bm25("fish", "http://short.com")
            > self.indexer.get_bm25("fish", "http://long.com")
        )

    def test_avgdl_cache_invalidated_on_index(self):
        self.indexer.index_page("http://a.com", "<p>one two three</p>")
        first = self.indexer.avgdl
        self.indexer.index_page(
            "http://b.com", "<p>one two three four five six</p>"
        )
        second = self.indexer.avgdl
        assert first != second

    def test_avgdl_zero_for_empty_index(self):
        assert self.indexer.avgdl == 0.0


class TestBM25Ranking:
    """Sanity-check that BM25 produces sensible end-to-end rankings."""

    def test_doc_with_higher_term_density_ranks_higher(self):
        engine, _ = make_engine(
            ("http://dense.com",
             "<p>fish fish fish water</p>"),
            ("http://sparse.com",
             "<p>fish water sea coral reef shell rock cave</p>"),
        )
        results = engine.find("fish")
        assert results[0][0] == "http://dense.com"

    def test_phrase_match_outranks_separated_terms(self):
        engine, _ = make_engine(
            ("http://phrase.com",
             "<p>good friends together always</p>"),
            ("http://separate.com",
             "<p>good things and reliable friends everywhere</p>"),
        )
        results = engine.find("good friends")
        assert results[0][0] == "http://phrase.com"
