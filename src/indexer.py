"""Inverted index for the COMP3011 search engine."""

import math
import re

from bs4 import BeautifulSoup


class Indexer:
    """Builds a positional inverted index from {url: html} pages."""

    _TOKEN_RE = re.compile(r'[a-z]+')

    def __init__(self):
        self.index = {}
        self.doc_count = 0
        self.doc_lengths = {}

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
        """Index a single page — counts, positions, and length."""
        text = self._extract_text(html)
        words = self._tokenise(text)
        self.doc_count += 1
        self.doc_lengths[url] = len(words)
        for position, word in enumerate(words):
            if word not in self.index:
                self.index[word] = {}
            if url not in self.index[word]:
                self.index[word][url] = {'count': 0, 'positions': []}
            self.index[word][url]['count'] += 1
            self.index[word][url]['positions'].append(position)

    def build_from_pages(self, pages):
        """Index every (url, html) pair in pages."""
        for url, html in pages.items():
            print(f"Indexing: {url}")
            self.index_page(url, html)

    def get_tfidf(self, word, url):
        """Return TF-IDF for word in document url.

        Uses smoothed IDF (log((N+1)/(df+1)) + 1) so terms appearing in
        every document don't blow up on log(0).
        """
        if word not in self.index or url not in self.index[word]:
            return 0.0
        tf = self.index[word][url]['count'] / max(
            self.doc_lengths.get(url, 1), 1)
        df = len(self.index[word])
        idf = math.log((self.doc_count + 1) / (df + 1)) + 1
        return tf * idf
