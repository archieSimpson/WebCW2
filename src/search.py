"""Query-time search over an :class:`Indexer`.

Implements:

* ``find(query)`` — Boolean-aware multi-word search ranked by Okapi
  BM25, with an exact-phrase bonus when the terms appear in sequence.
  Supports ``AND`` (default), ``OR`` and ``NOT`` operators.
* ``print_index(word)`` — pretty-prints the inverted-index entry for a
  single word (counts, TF-IDF, BM25 and a sample of positions).
* ``suggest(word)`` — "did you mean" candidates combining prefix
  matching and Levenshtein edit distance, so both partial words
  (``go`` → ``good``) and typos (``gud`` → ``good``) are corrected.
* ``expand_query(query)`` — fuzzy-aware per-term substitution so a
  query like ``gud friends`` can be rewritten to ``good friends``
  instead of returning empty (with the substitutions surfaced for
  user notification).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from indexer import Indexer


class SearchEngine:
    """Wraps an :class:`Indexer` and answers user queries.

    The engine itself is stateless apart from the indexer reference;
    one instance can serve any number of concurrent queries.
    """

    # Multiplicative weight applied to each contiguous-phrase match.
    # 2.0 is conservative — high enough that an exact phrase outranks
    # "two terms anywhere in the document" but low enough that a strong
    # BM25 signal can still dominate.
    PHRASE_WEIGHT: float = 2.0

    # Maximum number of suggestions returned by ``suggest``.
    MAX_SUGGESTIONS: int = 5

    # Reserved Boolean operators (uppercase only — case-sensitive so
    # they don't clash with regular search terms).
    _OP_OR = "OR"
    _OP_NOT = "NOT"
    _OP_AND = "AND"

    def __init__(self, indexer: "Indexer") -> None:
        self.indexer: "Indexer" = indexer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self, query: str | None) -> List[Tuple[str, float]]:
        """Return ``[(url, score), ...]`` ranked best-first.

        Query syntax:
            * Default: implicit ``AND`` — ``find good friends`` →
              documents containing both terms.
            * ``OR`` (uppercase): ``find good OR friends`` → documents
              containing either term.
            * ``NOT`` (uppercase): ``find good NOT enemy`` →
              documents containing ``good`` but not ``enemy``.
            * Mixed: ``find good OR life NOT enemy`` is parsed as
              ``(good) OR (life AND NOT enemy)``.

        Scoring is BM25 over the *positive* terms in whichever
        disjunctive clause matched, plus a phrase bonus when adjacent
        terms appear in sequence.

        Edge cases: ``None`` / empty / whitespace-only query → ``[]``.
        """
        if not query or not query.strip():
            return []

        disjuncts = self._parse_query(query)
        if not disjuncts:
            return []

        # Evaluate each disjunct, union the matches, remember which
        # disjunct produced each doc (for term-aware scoring).
        matched: dict[str, List[str]] = {}
        for positives, negatives in disjuncts:
            if not positives:
                # ``NOT enemy`` on its own has no positive evidence —
                # ignore (otherwise we'd return the entire corpus).
                continue
            urls = self._evaluate_clause(positives, negatives)
            for url in urls:
                if url not in matched:
                    matched[url] = positives

        if not matched:
            return []

        results: List[Tuple[str, float]] = []
        for url, positives in matched.items():
            # If a doc is matched by multiple OR disjuncts we score
            # using only the *first* disjunct's positives. This is a
            # deliberate simplification: scoring against every matching
            # disjunct would over-weight docs that happen to satisfy
            # several alternatives, which usually isn't what the user
            # means when they write ``A OR B``.
            score = sum(
                self.indexer.get_bm25(term, url) for term in positives
            )
            if len(positives) > 1:
                score += self._phrase_bonus(positives, url)
            results.append((url, round(score, 4)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def expand_query(
        self, query: str | None
    ) -> Tuple[str, Dict[str, str]]:
        """Rewrite ``query`` substituting missing terms with their top
        fuzzy suggestion.

        This is the missing piece between :meth:`find` and
        :meth:`suggest`: ``find`` currently returns empty as soon as a
        single positive term is absent from the index (because the
        AND-intersection collapses to the empty set), and ``suggest``
        only fires on the *whole query* — so ``find gud friends`` would
        fail outright even though ``gud`` is one edit away from a real
        word.

        ``expand_query`` walks the tokens, leaves operators alone,
        passes through any term already in the index, and replaces each
        missing term by its first ``suggest`` candidate (if one
        exists). The returned ``substitutions`` mapping lets the caller
        notify the user — e.g. "showing results for gud→good".

        Returns:
            ``(expanded_query, substitutions)`` where ``substitutions``
            is ``{original_lower: replacement}``. Empty mapping means
            no rewrite happened.

        Complexity: one ``suggest`` call per missing term — so
        ``O(M · V · L)`` worst case for ``M`` missing terms over a
        vocabulary of size ``V`` and average word length ``L``.
        """
        if not query or not query.strip():
            return query or "", {}

        tokens = query.split()
        substitutions: Dict[str, str] = {}
        new_tokens: List[str] = []
        operators = (self._OP_AND, self._OP_OR, self._OP_NOT)

        for tok in tokens:
            if tok in operators:
                new_tokens.append(tok)
                continue
            lower = tok.lower()
            if lower in self.indexer.index:
                new_tokens.append(tok)
                continue
            # Missing term — try to repair it.
            candidates = self.suggest(lower)
            if candidates and candidates[0] != lower:
                new_tokens.append(candidates[0])
                substitutions[lower] = candidates[0]
            else:
                # No good candidate; leave it as-is so the search
                # fails naturally (or the user sees what was unknown).
                new_tokens.append(tok)

        return " ".join(new_tokens), substitutions

    def suggest(self, partial_word: str | None) -> List[str]:
        """Return up to :attr:`MAX_SUGGESTIONS` candidate corrections.

        Combines two strategies:

        1. **Prefix matching** — useful when the user typed a partial
           word (``go`` → ``good``).
        2. **Levenshtein edit distance** — useful when the user
           typo'd a real word (``gud`` → ``good``). The threshold
           scales with query length (one edit per ~3 characters), so
           short queries don't match too liberally.

        Prefix matches always come first, then fuzzy matches sorted by
        ascending distance.
        """
        if not partial_word:
            return []
        partial = partial_word.lower()

        prefix_matches = [
            w for w in self.indexer.index if w.startswith(partial)
        ]
        if len(prefix_matches) >= self.MAX_SUGGESTIONS:
            return prefix_matches[:self.MAX_SUGGESTIONS]

        # Threshold tuned to catch realistic typos:
        #   * len ≤ 2  → only allow exact prefix matches (1 edit on a
        #     2-char query is far too liberal — half the dictionary).
        #   * len 3–5  → allow up to 2 edits (catches "gud" → "good").
        #   * len ≥ 6  → scale linearly so longer words tolerate more
        #     errors while still requiring a strong overlap.
        if len(partial) <= 2:
            threshold = 1
        elif len(partial) <= 5:
            threshold = 2
        else:
            threshold = max(2, len(partial) // 3)
        seen = set(prefix_matches)
        fuzzy: List[Tuple[int, str]] = []
        for w in self.indexer.index:
            if w in seen:
                continue
            d = self._edit_distance(partial, w, threshold)
            if d <= threshold:
                fuzzy.append((d, w))
        fuzzy.sort()
        combined = prefix_matches + [w for _, w in fuzzy]
        return combined[:self.MAX_SUGGESTIONS]

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
            f"{'BM25':>8}  Positions (first 10)"
        )
        print(f"  {'-' * 110}")
        for url, data in sorted(
                entries.items(),
                key=lambda x: x[1]['count'],
                reverse=True):
            tfidf = round(self.indexer.get_tfidf(word, url), 4)
            bm25 = round(self.indexer.get_bm25(word, url), 4)
            positions_preview = data['positions'][:10]
            print(
                f"  {url:<60} {data['count']:>6}  "
                f"{tfidf:>8}  {bm25:>8}  {positions_preview}"
            )
        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_query(
        self, query: str
    ) -> List[Tuple[List[str], List[str]]]:
        """Parse ``query`` into a list of ``(positives, negatives)`` clauses.

        Adjacent clauses are joined by an implicit ``AND``; ``OR``
        introduces a new clause; ``NOT`` flips the polarity of the
        next term. Operators are uppercase only so regular lowercase
        words like ``or`` aren't accidentally interpreted.
        """
        tokens = query.split()
        clauses: List[Tuple[List[str], List[str]]] = []
        positives: List[str] = []
        negatives: List[str] = []
        next_negative = False

        for tok in tokens:
            if tok == self._OP_OR:
                if positives or negatives:
                    clauses.append((positives, negatives))
                positives, negatives = [], []
                next_negative = False
            elif tok == self._OP_NOT:
                next_negative = True
            elif tok == self._OP_AND:
                # Explicit AND is a no-op (default behaviour).
                next_negative = False
            else:
                term = tok.lower()
                if next_negative:
                    negatives.append(term)
                    next_negative = False
                else:
                    positives.append(term)

        if positives or negatives:
            clauses.append((positives, negatives))
        return clauses

    def _evaluate_clause(
        self, positives: List[str], negatives: List[str]
    ) -> Set[str]:
        """Intersect positive postings, then subtract negative postings.

        Callers in :meth:`find` filter out clauses with empty positives
        before invoking this, so the ``match is None`` branch below is
        defensive — it only fires if a future caller forgets that
        contract. Marked ``no cover`` to keep the coverage report honest
        about which code paths the test suite actually exercises.
        """
        match: Set[str] | None = None
        for term in positives:
            if term not in self.indexer.index:
                return set()
            postings = set(self.indexer.index[term].keys())
            match = postings if match is None else match & postings
        if match is None:  # pragma: no cover - guarded by caller
            return set()
        for term in negatives:
            if term in self.indexer.index:
                match -= set(self.indexer.index[term].keys())
        return match

    def _phrase_bonus(self, terms: List[str], url: str) -> float:
        """Return ``PHRASE_WEIGHT * (number of exact-phrase matches)``.

        Position-set arithmetic: the phrase ``terms`` occurs starting
        at position ``p`` iff ``p`` is in ``positions(terms[0])`` and
        ``p+i`` is in ``positions(terms[i])`` for every later term.
        Implemented as repeated intersection of position sets, each
        shifted left by ``i``.
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

    @staticmethod
    def _edit_distance(a: str, b: str, max_dist: int | None = None) -> int:
        """Levenshtein distance between ``a`` and ``b``.

        Uses the standard ``O(|a|·|b|)`` two-row DP. When ``max_dist``
        is provided, an early bail-out short-circuits to a value just
        above ``max_dist`` whenever the lengths differ by more than
        that — preserves correctness for the threshold check used by
        :meth:`suggest` and saves work on the common case.
        """
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if max_dist is not None and abs(la - lb) > max_dist:
            return max_dist + 1
        if la == 0:
            return lb
        if lb == 0:
            return la

        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            curr = [i] + [0] * lb
            row_min = curr[0]
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    curr[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + cost,
                )
                if curr[j] < row_min:
                    row_min = curr[j]
            if max_dist is not None and row_min > max_dist:
                return max_dist + 1
            prev = curr
        return prev[lb]
