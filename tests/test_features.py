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


# ---------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------


class TestSnippets:

    def test_snippet_contains_match_and_context(self):
        idx = Indexer()
        idx.index_page(
            "http://a.com",
            "<p>the quick brown fox jumps over the lazy dog "
            "while the cat watches from afar</p>",
        )
        snippet = idx.get_snippet("http://a.com", ["fox"])
        assert "[FOX]" in snippet

    def test_snippet_highlights_terms_uppercase(self):
        idx = Indexer()
        idx.index_page("http://a.com", "<p>good friends together</p>")
        snippet = idx.get_snippet(
            "http://a.com", ["good", "friends"]
        )
        assert "[GOOD]" in snippet and "[FRIENDS]" in snippet

    def test_snippet_empty_for_unknown_url(self):
        idx = Indexer()
        idx.index_page("http://a.com", "<p>hello</p>")
        assert idx.get_snippet("http://other.com", ["hello"]) == ""

    def test_snippet_empty_for_no_matching_term(self):
        idx = Indexer()
        idx.index_page("http://a.com", "<p>hello world</p>")
        assert idx.get_snippet("http://a.com", ["banana"]) == ""

    def test_snippet_empty_when_doc_tokens_missing(self):
        idx = Indexer()
        idx.index_page("http://a.com", "<p>hello world</p>")
        # Simulate a v1 file load.
        idx.doc_tokens = {}
        assert idx.get_snippet("http://a.com", ["hello"]) == ""

    def test_snippet_window_truncates_with_ellipsis(self):
        idx = Indexer()
        idx.index_page(
            "http://a.com",
            "<p>" + " ".join(["word"] * 30) + " target "
            + " ".join(["word"] * 30) + "</p>",
        )
        snippet = idx.get_snippet(
            "http://a.com", ["target"], window=3
        )
        assert "..." in snippet
        assert "[TARGET]" in snippet

    def test_snippet_no_leading_ellipsis_when_match_at_start(self):
        idx = Indexer()
        idx.index_page("http://a.com", "<p>target one two three</p>")
        snippet = idx.get_snippet(
            "http://a.com", ["target"], window=10
        )
        assert not snippet.startswith("...")

    def test_snippet_round_trips_through_save_load(self):
        idx = Indexer()
        idx.index_page("http://a.com", "<p>one two target four</p>")
        with tempfile.NamedTemporaryFile(
                suffix='.json', delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            new = Indexer()
            new.load(path)
        finally:
            os.unlink(path)
        assert "[TARGET]" in new.get_snippet("http://a.com", ["target"])


# ---------------------------------------------------------------------
# Boolean query parser + end-to-end search
# ---------------------------------------------------------------------


class TestBooleanParser:

    def setup_method(self):
        self.engine, _ = make_engine(
            ("http://1.com",
             "<p>good life lived well</p>"),
            ("http://2.com",
             "<p>good friends together always</p>"),
            ("http://3.com",
             "<p>life is the enemy of routine</p>"),
            ("http://4.com",
             "<p>good enemy lurks within</p>"),
        )

    def test_implicit_and_unchanged(self):
        # Default conjunctive behaviour preserved.
        urls = {u for u, _ in self.engine.find("good life")}
        assert urls == {"http://1.com"}

    def test_or_unions_postings(self):
        urls = {u for u, _ in self.engine.find("life OR friends")}
        # life: 1, 3 ; friends: 2 → union {1, 2, 3}
        assert urls == {"http://1.com", "http://2.com", "http://3.com"}

    def test_not_excludes_postings(self):
        urls = {u for u, _ in self.engine.find("good NOT enemy")}
        # good: {1, 2, 4}; minus enemy: {4} → {1, 2}
        assert urls == {"http://1.com", "http://2.com"}

    def test_explicit_and_keyword_no_op(self):
        # Explicit AND should behave identically to implicit.
        a = {u for u, _ in self.engine.find("good life")}
        b = {u for u, _ in self.engine.find("good AND life")}
        assert a == b

    def test_or_and_not_combined(self):
        # (good NOT enemy) OR (life)
        urls = {u for u, _ in
                self.engine.find("good NOT enemy OR life")}
        # left clause: {1, 2}; right clause: {1, 3} → {1, 2, 3}
        assert urls == {"http://1.com", "http://2.com", "http://3.com"}

    def test_not_only_clause_ignored(self):
        # 'NOT enemy' alone has no positive evidence — must not return
        # the entire corpus.
        results = self.engine.find("NOT enemy")
        assert results == []

    def test_or_with_unknown_term_still_returns_other_branch(self):
        urls = {u for u, _ in
                self.engine.find("xyznotaword OR friends")}
        assert urls == {"http://2.com"}

    def test_lowercase_or_treated_as_term(self):
        # Operators are uppercase only — lowercase 'or' is a regular
        # term, so this becomes an implicit AND of three missing terms.
        results = self.engine.find("good or life")
        # 'or' isn't in the index, so the AND fails → empty.
        assert results == []

    def test_parse_query_single_term(self):
        clauses = self.engine._parse_query("hello")
        assert clauses == [(["hello"], [])]

    def test_parse_query_or_only(self):
        clauses = self.engine._parse_query("a OR b")
        assert clauses == [(["a"], []), (["b"], [])]

    def test_parse_query_not_attaches_to_next_term(self):
        clauses = self.engine._parse_query("a NOT b c")
        assert clauses == [(["a", "c"], ["b"])]

    def test_parse_query_empty(self):
        assert self.engine._parse_query("") == []


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
