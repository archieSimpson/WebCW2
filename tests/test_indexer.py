import pytest
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
