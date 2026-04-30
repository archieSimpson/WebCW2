"""Inverted index for the COMP3011 search engine.

The indexer extracts visible text from HTML, tokenises it into
lowercase ASCII words, and records every occurrence's position. The
resulting structure is::

    index[word][url] = {"count": int, "positions": [int, ...]}

Storing positions (rather than only counts) is what will let the
:class:`SearchEngine` give an exact-phrase boost without needing to
re-fetch the source documents.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup


# JSON schema version. Bump when the on-disk format changes so that
# load() can reject incompatible files instead of silently mis-parsing.
INDEX_SCHEMA_VERSION = 1


class Indexer:
    """Builds a positional inverted index from {url: html} pages.

    Attributes:
        index: ``word -> {url -> {"count": int, "positions": [int]}}``
        doc_count: Number of indexed documents (used in IDF).
        doc_lengths: ``url -> token count`` (used in TF normalisation).
    """

    _TOKEN_RE = re.compile(r'[a-z]+')

    def __init__(self) -> None:
        self.index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.doc_count: int = 0
        self.doc_lengths: Dict[str, int] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        # Cached average document length used by BM25; invalidated
        # whenever a new document is indexed or the index is reloaded.
        self._avgdl_cache: Optional[float] = None

    def _tokenise(self, text: str) -> List[str]:
        """Lowercase text and return all ASCII-letter runs."""
        return self._TOKEN_RE.findall(text.lower())

    def _extract_text(self, html: str) -> str:
        """Return visible text from html, stripped of script/style."""
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ')

    def index_page(self, url: str, html: str) -> None:
        """Index a single page — counts, positions, length, tokens."""
        text = self._extract_text(html)
        words = self._tokenise(text)
        self.doc_count += 1
        self.doc_lengths[url] = len(words)
        self.doc_tokens[url] = words
        self._avgdl_cache = None
        for position, word in enumerate(words):
            if word not in self.index:
                self.index[word] = {}
            if url not in self.index[word]:
                self.index[word][url] = {'count': 0, 'positions': []}
            self.index[word][url]['count'] += 1
            self.index[word][url]['positions'].append(position)

    def build_from_pages(self, pages: Dict[str, str]) -> None:
        """Index every (url, html) pair in pages."""
        for url, html in pages.items():
            print(f"Indexing: {url}")
            self.index_page(url, html)

    @property
    def avgdl(self) -> float:
        """Average document length across the collection."""
        if self._avgdl_cache is None:
            if not self.doc_lengths:
                self._avgdl_cache = 0.0
            else:
                self._avgdl_cache = (
                    sum(self.doc_lengths.values())
                    / len(self.doc_lengths)
                )
        return self._avgdl_cache

    def get_tfidf(self, word: str, url: str) -> float:
        """Return TF-IDF for ``word`` in document ``url``.

        Uses smoothed IDF (log((N+1)/(df+1)) + 1) to avoid the
        log(N/df) zero-division for terms appearing everywhere.
        """
        if word not in self.index or url not in self.index[word]:
            return 0.0
        tf = self.index[word][url]['count'] / max(
            self.doc_lengths.get(url, 1), 1)
        df = len(self.index[word])
        idf = math.log((self.doc_count + 1) / (df + 1)) + 1
        return tf * idf

    def get_bm25(
        self,
        word: str,
        url: str,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> float:
        """Return Okapi BM25 score for ``word`` in document ``url``.

        BM25 saturates term frequency (so a 10× repeated word doesn't
        score 10× higher) and length-normalises against avgdl.

        IDF uses the Lucene-style smoothing
        ``log(1 + (N - df + 0.5) / (df + 0.5))`` which is always
        non-negative — even for terms that appear in every document.
        """
        if word not in self.index or url not in self.index[word]:
            return 0.0
        f = self.index[word][url]['count']
        dl = self.doc_lengths.get(url, 0)
        avgdl = self.avgdl
        if avgdl == 0:
            return 0.0
        df = len(self.index[word])
        idf = math.log(
            1 + (self.doc_count - df + 0.5) / (df + 0.5)
        )
        norm = (f * (k1 + 1)) / (
            f + k1 * (1 - b + b * dl / avgdl)
        )
        return idf * norm

    def save(self, filepath: str) -> None:
        """Serialise the index to ``filepath`` as JSON."""
        data = {
            'schema_version': INDEX_SCHEMA_VERSION,
            'index': self.index,
            'doc_count': self.doc_count,
            'doc_lengths': self.doc_lengths,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Index saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load a previously saved index from ``filepath``.

        Files written without ``schema_version`` are treated as v1 for
        backwards compatibility; files with a higher version raise so
        a stale binary doesn't mis-parse a newer format.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        version = data.get('schema_version', 1)
        if version > INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"Index file schema v{version} is newer than the "
                f"supported v{INDEX_SCHEMA_VERSION}."
            )
        self.index = data['index']
        self.doc_count = data['doc_count']
        self.doc_lengths = data['doc_lengths']
        self._avgdl_cache = None
        print(f"Index loaded from {filepath}")
