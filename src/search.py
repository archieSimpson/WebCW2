"""Query-time search over an Indexer."""


class SearchEngine:
    """Wraps an Indexer and answers user queries."""

    # Multiplicative weight applied to each contiguous-phrase match.
    PHRASE_WEIGHT = 2.0

    def __init__(self, indexer):
        self.indexer = indexer

    def find(self, query):
        """Return [(url, score), ...] for documents matching query.

        Multi-word queries are implicitly AND — all terms must appear.
        Adjacent occurrences get a phrase bonus on top of the TF-IDF
        sum, so an exact phrase outranks "two terms anywhere".
        """
        if not query or not query.strip():
            return []

        terms = [t.lower() for t in query.split()]

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
            if len(terms) > 1:
                score += self._phrase_bonus(terms, url)
            results.append((url, round(score, 4)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _phrase_bonus(self, terms, url):
        """Return PHRASE_WEIGHT * (number of exact-phrase matches).

        Position-set arithmetic: the phrase ``terms`` occurs starting
        at position p iff p ∈ positions(terms[0]) and p+i ∈ positions(
        terms[i]) for every later term. Implemented as repeated
        intersection of position sets, each shifted left by i.
        """
        try:
            base_positions = set(
                self.indexer.index[terms[0]][url]['positions'])
            for i, term in enumerate(terms[1:], 1):
                shifted = set(
                    p - i for p in
                    self.indexer.index[term][url]['positions'])
                base_positions = base_positions.intersection(shifted)
            return self.PHRASE_WEIGHT * len(base_positions)
        except (KeyError, TypeError):
            return 0.0
