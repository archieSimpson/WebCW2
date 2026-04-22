"""Query-time search over an Indexer."""


class SearchEngine:
    """Wraps an Indexer and answers user queries."""

    def __init__(self, indexer):
        self.indexer = indexer

    def find(self, query):
        """Return [(url, score), ...] for documents matching query.

        Multi-word queries are implicitly AND — all terms must appear.
        Score is the sum of TF-IDF over the query terms per document.
        """
        if not query or not query.strip():
            return []

        terms = [t.lower() for t in query.split()]

        # Intersect postings to get docs that contain every term.
        match = None
        for term in terms:
            if term not in self.indexer.index:
                return []
            postings = set(self.indexer.index[term].keys())
            match = postings if match is None else match & postings

        if not match:
            return []

        results = []
        for url in match:
            score = sum(
                self.indexer.get_tfidf(term, url) for term in terms
            )
            results.append((url, round(score, 4)))
        return sorted(results, key=lambda x: x[1], reverse=True)
