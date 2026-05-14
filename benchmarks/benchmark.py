"""Microbenchmarks for the search engine.

Reports three things, all over a synthetic corpus so the numbers are
reproducible without hitting the network:

1. **Ranker comparison** — median latency per query for
   :meth:`Indexer.get_tfidf` vs :meth:`Indexer.get_bm25` summed over
   the query terms (the same loop ``SearchEngine.find`` uses).
2. **Suggest latency vs vocabulary size** — how
   :meth:`SearchEngine.suggest` scales as the indexed vocabulary
   grows (the inner Levenshtein loop is ``O(V · p · max_dist)`` so
   we expect roughly linear behaviour in ``V``).
3. **Indexing throughput** — pages indexed per second for a fixed
   document length, useful for sizing future crawls.

Output is plain text plus a markdown table that can be pasted into
the README. The harness uses only ``time.perf_counter`` and ``statistics``
from the standard library — no third-party dependency.

Run from the repo root::

    python benchmarks/benchmark.py

Optional flags::

    --quick       smaller corpus, fewer iterations (used in CI smoke)
    --no-table    suppress the markdown table at the end

Determinism: a fixed ``random.seed`` makes successive runs comparable.
Wall-clock numbers will of course vary between machines.
"""

from __future__ import annotations

import argparse
import random
import statistics
import string
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

from indexer import Indexer  # noqa: E402
from search import SearchEngine  # noqa: E402


# ---------------------------------------------------------------------
# Synthetic corpus helpers
# ---------------------------------------------------------------------

# A fixed vocabulary so query terms are guaranteed to be in the index.
_COMMON_WORDS = [
    "good", "life", "friends", "indifference", "enemy", "love",
    "world", "people", "think", "know", "things", "time", "live",
    "true", "great", "beautiful", "happy", "free", "mind", "soul",
]


def _random_word(rng: random.Random, length: int = 6) -> str:
    """Return a random lowercase ASCII word of ``length`` letters."""
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _build_synthetic_index(
    n_docs: int,
    doc_length: int,
    vocab_size: int,
    seed: int = 42,
) -> Indexer:
    """Build a fresh ``Indexer`` populated with synthetic documents.

    Every document is the same length so BM25's length normalisation
    isn't doing exotic work here — we just want the loop cost.
    """
    rng = random.Random(seed)
    vocab = list(_COMMON_WORDS)
    while len(vocab) < vocab_size:
        vocab.append(_random_word(rng))
    vocab = vocab[:vocab_size]

    indexer = Indexer()
    for i in range(n_docs):
        words = [rng.choice(vocab) for _ in range(doc_length)]
        html = "<p>" + " ".join(words) + "</p>"
        indexer.index_page(f"http://doc{i}.com", html)
    return indexer


# ---------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------


def _time_ms(fn: Callable[[], None], iterations: int) -> Tuple[float, float]:
    """Return ``(median_ms, stdev_ms)`` over ``iterations`` calls."""
    samples: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return median, stdev


# ---------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------


def bench_rankers(
    n_docs: int, doc_length: int, iterations: int
) -> List[Tuple[str, float, float]]:
    """Compare TF-IDF vs BM25 scoring cost over a fixed query set.

    The query set mirrors realistic find() usage: single-term,
    two-term and three-term queries. We sum per-term scores the same
    way SearchEngine.find does, so the number is directly comparable.
    """
    indexer = _build_synthetic_index(n_docs, doc_length, vocab_size=200)
    urls = list(indexer.doc_lengths.keys())
    queries = [
        ["good"],
        ["good", "life"],
        ["good", "life", "friends"],
    ]

    results: List[Tuple[str, float, float]] = []
    for terms in queries:
        label = " ".join(terms)

        def tfidf_run(terms=terms, urls=urls):
            for url in urls:
                sum(indexer.get_tfidf(t, url) for t in terms)

        def bm25_run(terms=terms, urls=urls):
            for url in urls:
                sum(indexer.get_bm25(t, url) for t in terms)

        med_tf, sd_tf = _time_ms(tfidf_run, iterations)
        med_bm, sd_bm = _time_ms(bm25_run, iterations)
        results.append((f"tfidf  '{label}'", med_tf, sd_tf))
        results.append((f"bm25   '{label}'", med_bm, sd_bm))
    return results


def bench_suggest(
    vocab_sizes: List[int], iterations: int
) -> List[Tuple[int, float, float]]:
    """Time ``suggest`` as the vocabulary grows.

    Uses ``gud`` — a 3-character typo of ``good`` — so the fuzzy
    branch fires (the prefix list is empty). This exercises the worst
    case: every vocabulary word gets a (truncated) edit-distance
    computation.
    """
    rows: List[Tuple[int, float, float]] = []
    for v in vocab_sizes:
        # Single doc that uses the whole vocabulary, so the index has
        # exactly ``v`` distinct tokens.
        rng = random.Random(42)
        vocab = list(_COMMON_WORDS)
        while len(vocab) < v:
            vocab.append(_random_word(rng))
        vocab = vocab[:v]
        indexer = Indexer()
        indexer.index_page(
            "http://corpus.com", "<p>" + " ".join(vocab) + "</p>"
        )
        engine = SearchEngine(indexer)
        median, stdev = _time_ms(lambda: engine.suggest("gud"), iterations)
        rows.append((v, median, stdev))
    return rows


def bench_indexing(
    n_docs: int, doc_length: int, iterations: int = 3
) -> Tuple[float, float, float]:
    """Return ``(median_seconds_total, median_pages_per_sec, stdev_pps)``."""
    samples_secs: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _build_synthetic_index(n_docs, doc_length, vocab_size=200)
        samples_secs.append(time.perf_counter() - t0)
    med = statistics.median(samples_secs)
    pps = [n_docs / s for s in samples_secs]
    return med, statistics.median(pps), (
        statistics.stdev(pps) if len(pps) > 1 else 0.0
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def _print_block(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _ranker_table(rows: List[Tuple[str, float, float]]) -> str:
    lines = [
        "| Ranker / query             | Median (ms) | Stdev (ms) |",
        "|----------------------------|-------------|------------|",
    ]
    for label, median, stdev in rows:
        lines.append(
            f"| `{label:<24}` | {median:>11.3f} | {stdev:>10.3f} |"
        )
    return "\n".join(lines)


def _suggest_table(rows: List[Tuple[int, float, float]]) -> str:
    lines = [
        "| Vocabulary size | Median (ms) | Stdev (ms) |",
        "|-----------------|-------------|------------|",
    ]
    for vocab, median, stdev in rows:
        lines.append(
            f"| {vocab:>15} | {median:>11.3f} | {stdev:>10.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true",
        help="Smaller corpus, fewer iterations (CI smoke).",
    )
    parser.add_argument(
        "--no-table", action="store_true",
        help="Suppress markdown tables at the end.",
    )
    args = parser.parse_args(argv)

    if args.quick:
        n_docs, doc_length, iters = 20, 100, 10
        vocab_sizes = [50, 200, 800]
    else:
        n_docs, doc_length, iters = 50, 400, 30
        vocab_sizes = [50, 200, 800, 3200]

    print("=" * 60)
    print(f"COMP3011 Search Engine — Microbenchmarks")
    print(f"corpus: {n_docs} docs × {doc_length} tokens, "
          f"{iters} iterations per measurement")
    print("=" * 60)

    _print_block("1. TF-IDF vs BM25 scoring latency")
    ranker_rows = bench_rankers(n_docs, doc_length, iters)
    for label, median, stdev in ranker_rows:
        print(f"  {label:<26}  median={median:7.3f} ms  "
              f"stdev={stdev:6.3f} ms")

    _print_block("2. suggest() latency vs vocabulary size")
    suggest_rows = bench_suggest(vocab_sizes, iters)
    for vocab, median, stdev in suggest_rows:
        print(f"  vocab={vocab:>5}  median={median:7.3f} ms  "
              f"stdev={stdev:6.3f} ms")

    _print_block("3. Indexing throughput")
    total, pps, pps_sd = bench_indexing(n_docs, doc_length)
    print(f"  built {n_docs} docs of {doc_length} tokens in "
          f"{total*1000:.1f} ms")
    print(f"  ≈ {pps:.1f} pages/sec  (stdev {pps_sd:.1f})")

    if not args.no_table:
        print()
        print("---")
        print("## Ranker comparison\n")
        print(_ranker_table(ranker_rows))
        print()
        print("## Suggest latency vs vocabulary\n")
        print(_suggest_table(suggest_rows))


if __name__ == "__main__":
    main()
