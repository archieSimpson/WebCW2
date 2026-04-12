"""Inverted index for the COMP3011 search engine."""

import re


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
