"""Inverted index for the COMP3011 search engine."""


class Indexer:
    """Builds a word -> {url -> count} inverted index."""

    def __init__(self):
        self.index = {}
        self.doc_count = 0
