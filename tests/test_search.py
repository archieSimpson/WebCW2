import pytest
from src.indexer import Indexer
from src.search import SearchEngine


def make_engine(*pages):
    indexer = Indexer()
    for url, html in pages:
        indexer.index_page(url, html)
    return SearchEngine(indexer), indexer


class TestSearchFind:

    def setup_method(self):
        self.engine, _ = make_engine(
            ("http://a.com",
             "<p>the good life is a life well lived</p>"),
            ("http://b.com",
             "<p>indifference is the enemy of good</p>"),
            ("http://c.com",
             "<p>good friends make life good</p>"),
        )

    def test_find_single_word_returns_match(self):
        results = self.engine.find("indifference")
        assert "http://b.com" in [r[0] for r in results]

    def test_find_excludes_non_matching_docs(self):
        results = self.engine.find("indifference")
        assert "http://a.com" not in [r[0] for r in results]

    def test_find_multi_word_conjunctive(self):
        results = self.engine.find("good life")
        urls = [r[0] for r in results]
        assert "http://b.com" not in urls

    def test_find_multi_word_all_present(self):
        results = self.engine.find("good life")
        urls = [r[0] for r in results]
        assert "http://a.com" in urls
        assert "http://c.com" in urls

    def test_find_nonexistent_word_returns_empty(self):
        assert self.engine.find("nonexistentword") == []

    def test_find_empty_string_returns_empty(self):
        assert self.engine.find("") == []

    def test_find_whitespace_only_returns_empty(self):
        assert self.engine.find("   ") == []

    def test_find_none_returns_empty(self):
        assert self.engine.find(None) == []

    def test_find_results_sorted_by_score_descending(self):
        results = self.engine.find("good")
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_case_insensitive(self):
        lower = {r[0] for r in self.engine.find("good")}
        upper = {r[0] for r in self.engine.find("GOOD")}
        assert lower == upper

    def test_find_mixed_case_query(self):
        results = self.engine.find("GoOd LiFe")
        assert len(results) > 0

    def test_find_returns_list_of_tuples(self):
        results = self.engine.find("good")
        assert all(isinstance(r, tuple) and len(r) == 2
                   for r in results)

    def test_find_scores_are_floats(self):
        results = self.engine.find("good")
        assert all(isinstance(r[1], float) for r in results)

    def test_find_one_term_missing_returns_empty(self):
        assert self.engine.find("good xyznotaword") == []

    def test_find_single_result(self):
        results = self.engine.find("indifference")
        assert len(results) == 1

    def test_find_multiple_results(self):
        results = self.engine.find("good")
        assert len(results) == 3

    def test_find_scores_are_positive(self):
        results = self.engine.find("good")
        assert all(r[1] > 0 for r in results)
