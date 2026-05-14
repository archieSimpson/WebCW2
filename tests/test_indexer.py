import pytest
import os
import tempfile
from src.indexer import Indexer


class TestIndexerTokenise:

    def setup_method(self):
        self.indexer = Indexer()

    def test_lowercase(self):
        tokens = self.indexer._tokenise("Hello WORLD")
        assert 'hello' in tokens
        assert 'world' in tokens

    def test_strips_punctuation(self):
        tokens = self.indexer._tokenise("great!")
        assert 'great' in tokens
        assert '!' not in tokens

    def test_strips_numbers(self):
        tokens = self.indexer._tokenise("page123")
        assert '123' not in tokens

    def test_empty_string(self):
        assert self.indexer._tokenise("") == []

    def test_multiple_spaces(self):
        tokens = self.indexer._tokenise("hello   world")
        assert 'hello' in tokens
        assert 'world' in tokens

    def test_hyphenated_word_split(self):
        tokens = self.indexer._tokenise("well-known")
        assert 'well' in tokens
        assert 'known' in tokens


class TestIndexerExtractText:

    def setup_method(self):
        self.indexer = Indexer()

    def test_removes_script(self):
        html = "<script>var x = 1;</script><p>hello</p>"
        text = self.indexer._extract_text(html)
        assert 'var' not in text
        assert 'hello' in text

    def test_removes_style(self):
        html = "<style>.cls { color: red; }</style><p>visible</p>"
        text = self.indexer._extract_text(html)
        assert 'color' not in text
        assert 'visible' in text

    def test_extracts_paragraph_text(self):
        html = "<p>tropical fish</p>"
        text = self.indexer._extract_text(html)
        assert 'tropical' in text
        assert 'fish' in text

    def test_empty_html_returns_string(self):
        text = self.indexer._extract_text("")
        assert isinstance(text, str)

    def test_nested_tags_extracted(self):
        html = "<div><p><span>deep text</span></p></div>"
        text = self.indexer._extract_text(html)
        assert 'deep' in text
        assert 'text' in text


class TestIndexerIndexPage:

    def setup_method(self):
        self.indexer = Indexer()

    def test_word_count_stored(self):
        self.indexer.index_page("http://a.com", "<p>fish fish tropical</p>")
        assert self.indexer.index['fish']['http://a.com']['count'] == 2

    def test_positions_stored(self):
        self.indexer.index_page("http://a.com", "<p>one two one</p>")
        positions = self.indexer.index['one']['http://a.com']['positions']
        assert len(positions) == 2

    def test_case_insensitive(self):
        self.indexer.index_page("http://a.com", "<p>Good good GOOD</p>")
        assert self.indexer.index['good']['http://a.com']['count'] == 3

    def test_missing_word_not_in_index(self):
        self.indexer.index_page("http://a.com", "<p>hello world</p>")
        assert 'xyz' not in self.indexer.index

    def test_doc_count_increments(self):
        self.indexer.index_page("http://a.com", "<p>hello</p>")
        self.indexer.index_page("http://b.com", "<p>world</p>")
        assert self.indexer.doc_count == 2

    def test_doc_length_stored(self):
        self.indexer.index_page("http://a.com", "<p>one two three</p>")
        assert self.indexer.doc_lengths['http://a.com'] > 0

    def test_multiple_pages_independent(self):
        self.indexer.index_page("http://a.com", "<p>fish</p>")
        self.indexer.index_page("http://b.com", "<p>tropical</p>")
        assert 'http://b.com' not in self.indexer.index.get('fish', {})

    def test_position_order_correct(self):
        self.indexer.index_page("http://a.com", "<p>alpha beta alpha</p>")
        positions = self.indexer.index['alpha']['http://a.com']['positions']
        assert positions[0] < positions[1]

    def test_scripts_not_indexed(self):
        self.indexer.index_page(
            "http://a.com", "<script>javascript</script><p>hello</p>")
        assert 'javascript' not in self.indexer.index

    def test_empty_page_does_not_crash(self):
        self.indexer.index_page("http://a.com", "")
        assert self.indexer.doc_count == 1

    def test_single_word_page(self):
        self.indexer.index_page("http://a.com", "<p>serendipity</p>")
        assert 'serendipity' in self.indexer.index

    def test_positions_are_integers(self):
        self.indexer.index_page("http://a.com", "<p>hello world</p>")
        positions = self.indexer.index['hello']['http://a.com']['positions']
        assert all(isinstance(p, int) for p in positions)


class TestIndexerSaveLoad:

    def setup_method(self):
        self.indexer = Indexer()

    def _save_and_reload(self):
        with tempfile.NamedTemporaryFile(
                suffix='.json', delete=False) as f:
            fname = f.name
        self.indexer.save(fname)
        new_indexer = Indexer()
        new_indexer.load(fname)
        os.unlink(fname)
        return new_indexer

    def test_counts_preserved(self):
        self.indexer.index_page("http://a.com", "<p>fish fish tropical</p>")
        new = self._save_and_reload()
        assert new.index['fish']['http://a.com']['count'] == 2

    def test_doc_count_preserved(self):
        self.indexer.index_page("http://a.com", "<p>hello</p>")
        new = self._save_and_reload()
        assert new.doc_count == 1

    def test_positions_preserved(self):
        self.indexer.index_page("http://a.com", "<p>one two one</p>")
        new = self._save_and_reload()
        assert len(new.index['one']['http://a.com']['positions']) == 2

    def test_doc_lengths_preserved(self):
        self.indexer.index_page("http://a.com", "<p>one two three</p>")
        new = self._save_and_reload()
        assert 'http://a.com' in new.doc_lengths

    def test_multiple_pages_preserved(self):
        self.indexer.index_page("http://a.com", "<p>fish</p>")
        self.indexer.index_page("http://b.com", "<p>tropical fish</p>")
        new = self._save_and_reload()
        assert new.doc_count == 2

    def test_save_creates_file(self):
        self.indexer.index_page("http://a.com", "<p>hello</p>")
        with tempfile.NamedTemporaryFile(
                suffix='.json', delete=False) as f:
            fname = f.name
        try:
            self.indexer.save(fname)
            assert os.path.exists(fname)
        finally:
            os.unlink(fname)

    def test_loaded_index_is_searchable(self):
        self.indexer.index_page("http://a.com", "<p>hello world</p>")
        new = self._save_and_reload()
        assert 'hello' in new.index
        assert 'world' in new.index


class TestIndexerTFIDF:

    def setup_method(self):
        self.indexer = Indexer()

    def test_higher_frequency_scores_higher(self):
        self.indexer.index_page("http://a.com", "<p>fish fish fish tropical water sea</p>")
        self.indexer.index_page("http://b.com", "<p>fish tropical water sea coral reef</p>")
        assert (self.indexer.get_tfidf('fish', 'http://a.com') >
                self.indexer.get_tfidf('fish', 'http://b.com'))

    def test_missing_word_returns_zero(self):
        self.indexer.index_page("http://a.com", "<p>hello</p>")
        assert self.indexer.get_tfidf('notaword', 'http://a.com') == 0.0

    def test_missing_url_returns_zero(self):
        self.indexer.index_page("http://a.com", "<p>hello</p>")
        assert self.indexer.get_tfidf('hello', 'http://other.com') == 0.0

    def test_returns_float(self):
        self.indexer.index_page("http://a.com", "<p>hello</p>")
        assert isinstance(
            self.indexer.get_tfidf('hello', 'http://a.com'), float)

    def test_rare_word_scores_higher_than_common(self):
        for i in range(10):
            self.indexer.index_page(f"http://doc{i}.com", "<p>the</p>")
        self.indexer.index_page(
            "http://rare.com", "<p>the serendipity</p>")
        score_common = self.indexer.get_tfidf('the', 'http://rare.com')
        score_rare = self.indexer.get_tfidf(
            'serendipity', 'http://rare.com')
        assert score_rare > score_common

    def test_score_is_positive_for_present_word(self):
        self.indexer.index_page("http://a.com", "<p>hello world</p>")
        assert self.indexer.get_tfidf('hello', 'http://a.com') > 0

    def test_both_missing_returns_zero(self):
        assert self.indexer.get_tfidf('nothing', 'http://nowhere.com') == 0.0