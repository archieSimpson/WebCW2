"""Query-time search over an Indexer."""


class SearchEngine:
    """Wraps an Indexer and answers user queries."""

    def __init__(self, indexer):
        self.indexer = indexer
