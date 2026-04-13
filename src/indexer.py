"""Inverted index for the COMP3011 search engine."""

import re

from bs4 import BeautifulSoup


class Indexer:
    """Builds a word -> {url -> count} inverted index."""

    _TOKEN_RE = re.compile(r'[a-z]+')

    def __init__(self):
        self.index = {}
        self.doc_count = 0

    def _tokenise(self, text):
        """Lowercase text and return all ASCII-letter runs.

        Drops numbers and punctuation; splits hyphenated words.
        """
        return self._TOKEN_RE.findall(text.lower())

    def _extract_text(self, html):
        """Return visible text from html, stripped of script/style."""
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ')

    def index_page(self, url, html):
        """Index a single page — updates word counts per url."""
        text = self._extract_text(html)
        words = self._tokenise(text)
        self.doc_count += 1
        for word in words:
            if word not in self.index:
                self.index[word] = {}
            if url not in self.index[word]:
                self.index[word][url] = {'count': 0}
            self.index[word][url]['count'] += 1
