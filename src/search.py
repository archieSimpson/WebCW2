"""Query-time search over an Indexer."""


class SearchEngine:
    """Wraps an Indexer and answers user queries."""

    # Multiplicative weight applied to each contiguous-phrase match.
    PHRASE_WEIGHT = 2.0

    # Maximum number of suggestions returned by ``suggest``.
    MAX_SUGGESTIONS = 5

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

    def suggest(self, partial_word):
        """Return up to MAX_SUGGESTIONS prefix-match candidates.

        Useful when the user typed a partial word — ``go`` → ``good``.
        """
        if not partial_word:
            return []
        partial = partial_word.lower()
        matches = [
            w for w in self.indexer.index if w.startswith(partial)
        ]
        return matches[:self.MAX_SUGGESTIONS]

    def print_index(self, word):
        """Pretty-print the inverted-index entry for ``word``."""
        word = word.lower().strip()
        if not word:
            print("Please provide a word.")
            return
        if word not in self.indexer.index:
            print(f"'{word}' not found in index.")
            return

        entries = self.indexer.index[word]
        print(f"\nInverted index for '{word}' "
              f"({len(entries)} document(s)):")
        print(
            f"\n  {'URL':<60} {'Count':>6}  {'TF-IDF':>8}  "
            f"Positions (first 10)"
        )
        print(f"  {'-' * 100}")
        for url, data in sorted(
                entries.items(),
                key=lambda x: x[1]['count'],
                reverse=True):
            tfidf = round(self.indexer.get_tfidf(word, url), 4)
            positions_preview = data['positions'][:10]
            print(
                f"  {url:<60} {data['count']:>6}  "
                f"{tfidf:>8}  {positions_preview}"
            )
        print()

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
