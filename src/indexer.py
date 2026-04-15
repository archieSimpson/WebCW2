"""Inverted index for the COMP3011 search engine."""

import json
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
        """Return TF-IDF for word in document url."""
        if word not in self.index or url not in self.index[word]:
            return 0.0
        tf = self.index[word][url]['count'] / max(
            self.doc_lengths.get(url, 1), 1)
        df = len(self.index[word])
        idf = math.log((self.doc_count + 1) / (df + 1)) + 1
        return tf * idf

    def save(self, filepath):
        """Serialise the index to filepath as JSON."""
        data = {
            'index': self.index,
            'doc_count': self.doc_count,
            'doc_lengths': self.doc_lengths,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Index saved to {filepath}")

    def load(self, filepath):
        """Load a previously saved index from filepath."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.index = data['index']
        self.doc_count = data['doc_count']
        self.doc_lengths = data['doc_lengths']
        print(f"Index loaded from {filepath}")
