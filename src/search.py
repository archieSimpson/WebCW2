"""Query-time search over an :class:`Indexer`.

Implements:

* ``find(query)`` — multi-word implicit-AND search ranked by TF-IDF,
  with an exact-phrase bonus when the terms appear in sequence.
* ``print_index(word)`` — pretty-prints the inverted-index entry for
  a single word (count, TF-IDF, positions).
* ``suggest(word)`` — prefix-based "did you mean" candidates (fuzzy
  matching arrives in a later commit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from indexer import Indexer


class SearchEngine:
    """Wraps an :class:`Indexer` and answers user queries."""

    # Multiplicative weight applied to each contiguous-phrase match.
    PHRASE_WEIGHT: float = 2.0

    # Maximum number of suggestions returned by ``suggest``.
    MAX_SUGGESTIONS: int = 5

    def __init__(self, indexer: "Indexer") -> None:
        self.indexer: "Indexer" = indexer

    def find(self, query: str | None) -> List[Tuple[str, float]]:
        """Return ``[(url, score), ...]`` ranked best-first.

        Multi-word queries are implicitly AND; adjacent occurrences
        get a phrase bonus on top of the TF-IDF sum.
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

        results: List[Tuple[str, float]] = []
        for url in match:
            score = sum(
                self.indexer.get_tfidf(term, url) for term in terms
            )
            if len(terms) > 1:
                score += self._phrase_bonus(terms, url)
            results.append((url, round(score, 4)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def suggest(self, partial_word: str | None) -> List[str]:
        """Return up to ``MAX_SUGGESTIONS`` prefix-match candidates."""
        if not partial_word:
            return []
        partial = partial_word.lower()
        matches = [
            w for w in self.indexer.index if w.startswith(partial)
        ]
        return matches[:self.MAX_SUGGESTIONS]

    def print_index(self, word: str) -> None:
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

    def _phrase_bonus(self, terms: List[str], url: str) -> float:
        """Return PHRASE_WEIGHT * (number of exact-phrase matches)."""
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
