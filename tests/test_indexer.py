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
