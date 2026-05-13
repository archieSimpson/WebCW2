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


class TestSearchPhraseBonus:

    def test_adjacent_words_rank_higher(self):
        engine, _ = make_engine(
            ("http://phrase.com",
             "<p>good friends together always</p>"),
            ("http://separate.com",
             "<p>good things and friends everywhere</p>"),
        )
        results = engine.find("good friends")
        assert results[0][0] == "http://phrase.com"

    def test_phrase_bonus_not_applied_to_single_term(self):
        engine, _ = make_engine(
            ("http://a.com", "<p>good good good</p>"),
        )
        results = engine.find("good")
        assert len(results) == 1

    def test_phrase_bonus_zero_when_not_adjacent(self):
        engine, _ = make_engine(
            ("http://a.com", "<p>good day bad friends</p>"),
        )
        bonus = engine._phrase_bonus(['good', 'friends'], 'http://a.com')
        assert bonus == 0.0

    def test_phrase_bonus_positive_when_adjacent(self):
        engine, _ = make_engine(
            ("http://a.com", "<p>good friends</p>"),
        )
        bonus = engine._phrase_bonus(['good', 'friends'], 'http://a.com')
        assert bonus > 0.0

    def test_phrase_bonus_returns_zero_for_missing_url(self):
        engine, _ = make_engine(
            ("http://a.com", "<p>good friends</p>"),
        )
        bonus = engine._phrase_bonus(
            ['good', 'friends'], 'http://notindexed.com')
        assert bonus == 0.0

    def test_phrase_bonus_scales_with_occurrences(self):
        engine, _ = make_engine(
            ("http://a.com",
             "<p>good friends good friends good friends</p>"),
        )
        bonus = engine._phrase_bonus(['good', 'friends'], 'http://a.com')
        assert bonus >= 6.0

    def test_three_word_phrase_detected(self):
        engine, _ = make_engine(
            ("http://a.com", "<p>the good life</p>"),
            ("http://b.com", "<p>the life good</p>"),
        )
        results = engine.find("the good life")
        assert results[0][0] == "http://a.com"


class TestSearchPrintIndex:

    def setup_method(self):
        self.engine, _ = make_engine(
            ("http://a.com", "<p>hello world hello</p>"),
        )

    def test_word_not_found_message(self, capsys):
        self.engine.print_index("notaword")
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_empty_word_message(self, capsys):
        self.engine.print_index("")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_valid_word_prints_url(self, capsys):
        self.engine.print_index("hello")
        captured = capsys.readouterr()
        assert "http://a.com" in captured.out

    def test_valid_word_prints_count(self, capsys):
        self.engine.print_index("hello")
        captured = capsys.readouterr()
        assert "2" in captured.out

    def test_valid_word_prints_tfidf_header(self, capsys):
        self.engine.print_index("hello")
        captured = capsys.readouterr()
        assert "TF-IDF" in captured.out

    def test_valid_word_prints_positions(self, capsys):
        self.engine.print_index("hello")
        captured = capsys.readouterr()
        assert "[" in captured.out

    def test_word_case_insensitive(self, capsys):
        self.engine.print_index("HELLO")
        captured = capsys.readouterr()
        assert "http://a.com" in captured.out


class TestSearchSuggest:

    def setup_method(self):
        self.engine, _ = make_engine(
            ("http://a.com",
             "<p>good goodbye goodness great green</p>"),
        )

    def test_suggest_returns_completions(self):
        suggestions = self.engine.suggest("go")
        assert "good" in suggestions

    def test_suggest_no_match_returns_empty(self):
        assert self.engine.suggest("zzz") == []

    def test_suggest_empty_string_returns_empty(self):
        assert self.engine.suggest("") == []

    def test_suggest_max_five_results(self):
        suggestions = self.engine.suggest("go")
        assert len(suggestions) <= 5

    def test_suggest_case_insensitive(self):
        suggestions = self.engine.suggest("GO")
        assert "good" in suggestions

    def test_suggest_exact_word_included(self):
        suggestions = self.engine.suggest("good")
        assert "good" in suggestions

    def test_suggest_returns_list(self):
        suggestions = self.engine.suggest("go")
        assert isinstance(suggestions, list)

    def test_suggest_none_input_returns_empty(self):
        assert self.engine.suggest(None) == []