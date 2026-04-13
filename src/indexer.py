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
        """Index a single page — counts plus positions per word per url."""
        text = self._extract_text(html)
        words = self._tokenise(text)
        self.doc_count += 1
        for position, word in enumerate(words):
            if word not in self.index:
                self.index[word] = {}
            if url not in self.index[word]:
                self.index[word][url] = {'count': 0, 'positions': []}
            self.index[word][url]['count'] += 1
            self.index[word][url]['positions'].append(position)
