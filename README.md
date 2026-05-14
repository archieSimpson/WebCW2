# Quotes Search Engine

A polite web crawler, inverted indexer, and search tool for
[`https://quotes.toscrape.com/`](https://quotes.toscrape.com/), built for
**COMP3011 — Web Services and Web Data, Coursework 2**.

The tool ships as a small command-line shell with four commands —
`build`, `load`, `print`, `find` — and provides:

- positional inverted indexing
- **Okapi BM25** ranking (with TF-IDF kept alongside for comparison)
- exact-phrase boosting via stored positions
- **Boolean query operators** — `AND` (default), `OR`, `NOT`
- **Fuzzy "did you mean"** combining prefix matching and **Levenshtein
  edit distance** so both partial words and typos get corrected
- **Per-term query expansion** — `find gud friends` is silently rewritten
  to `find good friends` with the substitution surfaced to the user,
  instead of failing the AND-intersection on the missing token
- **Result snippets** with query-term highlighting
- **Benchmark harness** validating the complexity analysis on a
  synthetic corpus (see [Benchmarks](#benchmarks))

---

## Project overview

The project implements the classic three-stage pipeline of a search
engine:

1. **Crawler** ([src/crawler.py](src/crawler.py)) — performs a
   breadth-first traversal of the target site, observing a 6-second
   politeness window between requests and respecting `robots.txt`. URLs
   are normalised (fragment/query stripped) and same-domain only.
2. **Indexer** ([src/indexer.py](src/indexer.py)) — extracts visible
   text with BeautifulSoup, tokenises it (lowercase, ASCII-letters
   only), and builds a positional inverted index of the form
   `word → {url → {count, positions}}`. Document lengths and the
   collection size are tracked so TF-IDF can be computed on demand.
3. **Search engine** ([src/search.py](src/search.py)) — supports
   single-word lookup, multi-word queries with **Boolean operators**
   (`AND` / `OR` / `NOT`), exact-phrase bonuses (using stored
   positions), **fuzzy "did you mean"** suggestions (prefix matching
   plus Levenshtein edit distance), and **highlighted snippets**.
   Results are ranked by **BM25** plus a phrase bonus.

The shell ([src/main.py](src/main.py)) wires these together and persists
the index to `data/index.json`.

---

## Architecture

```
                    ┌──────────────┐
   quotes.toscrape  │   Crawler    │   pages: {url -> html}
   ──────────────►  │ (BFS, polite)│ ───────────────┐
                    └──────────────┘                ▼
                                           ┌──────────────┐
                                           │   Indexer    │
                                           │  (tokenise,  │
                                           │   inverted   │
                                           │   index)     │
                                           └──────┬───────┘
                                                  │ index.json
                                                  ▼
   user query ──►  ┌──────────────┐        ┌──────────────┐
   "good friends"  │ SearchEngine │◄───────│   Storage    │
                   │ (TF-IDF +    │        │ (load/save)  │
                   │  phrase)     │        └──────────────┘
                   └──────┬───────┘
                          ▼
                     ranked URLs
```

### Inverted index data structure

```python
index = {
    "good": {
        "https://quotes.toscrape.com/": {
            "count": 4,
            "positions": [12, 47, 88, 102],
        },
        "https://quotes.toscrape.com/page/2/": {
            "count": 2,
            "positions": [33, 91],
        },
    },
    ...
}

# alongside the inverted index, the schema-v2 file also stores:
doc_count    = 50
doc_lengths  = {"https://quotes.toscrape.com/": 437, ...}
doc_tokens   = {"https://quotes.toscrape.com/": ["the", "good", ...], ...}
```

Storing **positions** (not just counts) is what enables exact-phrase
ranking without re-fetching documents: for a query `t1 t2 ... tk`, the
phrase occurs at position `p` in document `d` iff `p ∈ positions(t1, d)`
and `p+i ∈ positions(t_{i+1}, d)` for all `i`. This is computed by
intersecting position sets after shifting — `O(min |posList|)` per
candidate document.

Storing **`doc_tokens`** (the original token list per document) lets
the engine produce highlighted snippets at query time without
re-fetching the page. Total overhead for the quotes.toscrape.com
corpus is around 25 KB — negligible next to the inverted index.

### Why JSON?

The brief requires "save the entire index to a single file". JSON is
chosen for its **inspectability** (the file is human-readable, useful
for debugging and demonstration) and **portability** (no `pickle`
trust-boundary issues, and the file is reproducible across Python
versions). For larger collections a binary format such as `pickle` or a
dedicated key-value store would be preferable; for the
quotes.toscrape.com corpus (~50 pages, ~3 MB index) JSON is comfortably
fast enough.

---

## Installation

Requires Python 3.10+.

```bash
git clone <repository-url>
cd WebCW2

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies (all pinned in [requirements.txt](requirements.txt)):

| Package | Purpose |
|---|---|
| `requests` | HTTP client with session/keep-alive |
| `beautifulsoup4` | HTML parsing |
| `pytest` + `pytest-cov` | Test runner and coverage reporting |

---

## Usage

Launch the interactive shell:

```bash
python src/main.py
```

You will see:

```
============================================================
  COMP3011 Search Engine
  Commands: build | load | print <word> | find <query>
  Type 'quit' to exit
============================================================

>
```

### `build` — crawl and index

```
> build
```

Crawls every same-domain page reachable from
`https://quotes.toscrape.com/`, observing the 6-second politeness window.
Builds the inverted index and writes it to `data/index.json`. Allow
roughly 5–6 minutes for a full crawl.

### `load` — load a previously built index

```
> load
```

Reads `data/index.json` and rebuilds the in-memory index. Fails
gracefully if no index file exists.

### `print` — show the inverted index for a word

```
> print nonsense
```

```
Inverted index for 'nonsense' (2 document(s)):

  URL                                                                Count    TF-IDF  Positions (first 10)
  ----------------------------------------------------------------------------------------------------
  https://quotes.toscrape.com/tag/inspirational/page/1/                  1    0.0421  [142]
  https://quotes.toscrape.com/page/4/                                    1    0.0387  [233]
```

### `find` — search the index

Single-word query:

```
> find indifference
```

Multi-word conjunctive query — all terms must appear; adjacent
occurrences are boosted by the phrase bonus:

```
> find good friends
```

Output (with highlighted snippet):

```
Found 3 page(s) for 'good friends':
     Score  URL
  ----------------------------------------------------------------------
    4.1283  https://quotes.toscrape.com/tag/friends/
            ... that is what makes [GOOD] [FRIENDS] true ...
    2.0814  https://quotes.toscrape.com/
            ... need [GOOD] [FRIENDS] in life ...
    1.7251  https://quotes.toscrape.com/page/2/
            ... the [GOOD] of his [FRIENDS] always ...
```

#### Boolean operators

`AND` is the default. `OR` and `NOT` are uppercase keywords (so plain
words like *or* or *not* aren't accidentally interpreted):

| Query | Meaning |
|---|---|
| `find good friends` | docs containing **both** `good` and `friends` |
| `find good AND friends` | identical (explicit AND) |
| `find good OR friends` | docs containing **either** term |
| `find good NOT enemy` | docs containing `good` but **not** `enemy` |
| `find good OR life NOT enemy` | parsed as `(good) OR (life AND NOT enemy)` |

#### Fuzzy "did you mean"

When `find` returns no results, the engine offers up to five
candidates from the index. Both partial words and typos are caught:

```
> find indifrence
No pages found for 'indifrence'.
Did you mean: indifference?

> find go
No pages found for 'go'.
Did you mean: good, goodbye, goodness?
```

### Edge cases

| Input | Behaviour |
|---|---|
| `find` (no argument) | prints usage |
| `find xyz` (non-existent) | prints "No pages found", offers prefix suggestions |
| `find ""` / `find    ` | prints empty-result message |
| `print` (no argument) | prints usage |
| `print xyz` (not in index) | prints "not found" |
| `load` before `build` | prints "No index found. Run 'build' first." |
| `find` on empty index | prints "Index is empty. Run 'build' or 'load' first." |
| `GOOD` / `Good` / `good` | treated identically (case-insensitive) |

---

## Testing

The test suite uses **pytest** and covers crawler, indexer, search
engine, the shell command handlers, and integration paths. The crawler
is tested with `unittest.mock` so no network requests are made during
testing.

Run all tests:

```bash
pytest
```

With coverage report:

```bash
pytest --cov=src --cov-report=term-missing
```

Run a single test module:

```bash
pytest tests/test_indexer.py -v
```

### Testing strategy

- **Unit tests** — each public method of `Crawler`, `Indexer`, and
  `SearchEngine` has dedicated tests covering happy paths, boundary
  conditions, and failure modes.
- **Mock-based crawler tests** — network I/O is mocked so the suite is
  fast (<1s) and deterministic.
- **Save/load round-trip tests** — the index is persisted and reloaded
  to assert serialisation correctness for counts, positions, and
  document lengths.
- **TF-IDF semantic tests** — verify that rare terms outscore common
  terms and that frequency monotonicity holds.
- **Integration tests** — exercise the full crawler→indexer→search
  pipeline end-to-end against fabricated HTML.
- **Shell tests** — the `run_build` / `run_load` / `run_print` /
  `run_find` entry points are tested directly with captured stdout.

---

## Project structure

```
WebCW2/
├── src/
│   ├── crawler.py        # BFS, politeness, robots.txt
│   ├── indexer.py        # tokenise, inverted index, TF-IDF
│   ├── search.py         # find / print / phrase bonus / suggest
│   └── main.py           # interactive shell
├── tests/
│   ├── test_crawler.py
│   ├── test_indexer.py
│   ├── test_search.py
│   ├── test_main.py
│   └── test_integration.py
├── data/
│   └── index.json        # produced by `build`
├── conftest.py           # pytest path fixture
├── pytest.ini            # pytest config
├── requirements.txt
└── README.md
```

---

## Design decisions and trade-offs

| Decision | Rationale |
|---|---|
| BFS over DFS | BFS reaches "important" pages (close to root) first, which is preferable when the crawl could be interrupted. |
| Politeness via `time.sleep` | Simple and easy to reason about; aligns with the brief's 6-second requirement. A token-bucket or async scheduler would be needed for a larger crawl. |
| `requests.Session` | Reuses TCP connection — avoids re-handshake overhead for the ~50 pages crawled. |
| Lowercase ASCII tokens (`[a-z]+`) | Matches the brief's case-insensitive requirement and avoids noise from numbers and punctuation. A more sophisticated tokeniser (Unicode, stemming) would be needed for production but is out of scope. |
| Positional index | Enables exact-phrase ranking without re-fetching pages. ~10× larger than a count-only index but still trivially small for this corpus. |
| **BM25** as primary ranker | TF-IDF was the 1970s baseline; **BM25** (Robertson, 1994) is the modern de-facto standard. It saturates term frequency (a 100× repeated word doesn't score 100× higher) and length-normalises against the average document length, both of which TF-IDF gets wrong. TF-IDF is still computed and shown by `print` for comparison. |
| **Lucene-style smoothed IDF** for BM25 | `log(1 + (N − df + 0.5)/(df + 0.5))` is always non-negative — even for terms that appear in every document — so the score stays well-behaved. |
| Boolean operators uppercase-only | Keeps the lowercase words `or` / `and` / `not` available as ordinary search terms. Documented and visible in the shell's `help` text. |
| Hybrid prefix + edit-distance suggest | Pure prefix misses typos (`gud` → `good`); pure edit distance misses partials (`go` → `good`). Combining the two covers both. Threshold is tuned (1 edit for ≤ 2-char queries, 2 for 3-5 chars, then ~⅓ length) to balance recall against false positives. |
| **`suggest` tiebreak by document frequency** (Norvig 2007) | At the same edit distance, multiple candidates are usually plausible (`gud` is 2 edits from both `good` and `and`). Pure alphabetical order picks `and` — confidently wrong. Sorting by `(distance, -df, word)` prefers the more common word, matching Peter Norvig's spelling-corrector heuristic (more frequent ⇒ more likely intended). |
| **Per-term fuzzy rewrite** (`expand_query`) | `find` returns empty as soon as one AND-term is missing, and `suggest` only fires on the *whole* query — so `find gud friends` would fail outright. `expand_query` walks the tokens, substitutes any missing term with its top fuzzy candidate, and surfaces the substitutions to the user (*"showing results for `good friends` — corrected: `gud→good`"*). Operators are preserved; unresolvable terms are left untouched so the search fails legibly rather than silently. |
| Snippets stored as token lists | Generating snippets requires the original document text. Storing the tokens (~25 KB total for this corpus) is tiny next to the inverted index and avoids any need to re-crawl when serving snippets. |
| Default conjunctive multi-word semantics | The brief shows `find good friends` returning pages containing both terms — "AND" is the obvious reading. Boolean operators are layered on top without changing the default. |

---

## Complexity (back-of-envelope)

| Operation | Time | Space |
|---|---|---|
| Crawl (N pages) | `O(N · politeness)` wall-clock; CPU-bound work is `O(N · |page|)` | `O(total HTML)` |
| Index a page of length L | `O(L)` | `O(L)` |
| `find` (k terms, BM25 + phrase) | `O(k · |postings| + |intersection| · k + Σ |positions|)` | `O(|intersection|)` |
| `print` a word | `O(|postings|)` | `O(|postings|)` |
| `suggest` (V vocabulary, partial of length p) | `O(V · p²)` worst case; truncated by the early-exit Levenshtein bound to `O(V · p · max_dist)` | `O(p)` |
| Save / load (JSON) | `O(|index|)` | `O(|index|)` |

---

## Benchmarks

The complexity table above is the theoretical story; the harness in
[benchmarks/benchmark.py](benchmarks/benchmark.py) validates it
empirically against a synthetic corpus so the numbers are reproducible
without hitting the network.

```bash
python benchmarks/benchmark.py            # full run, ~2 s
python benchmarks/benchmark.py --quick    # smoke run, ~0.5 s
```

The full run reports three things, all on a fixed corpus of **50
documents × 400 tokens** with **30 iterations** per measurement
(Python 3.14, Apple Silicon laptop — your numbers will vary):

### 1. TF-IDF vs BM25 scoring latency

Per-query, summed over the documents in the corpus the same way
`SearchEngine.find` does:

| Ranker / query             | Median (ms) | Stdev (ms) |
|----------------------------|-------------|------------|
| `tfidf  'good'`            |       0.020 |      0.014 |
| `bm25   'good'`            |       0.025 |      0.005 |
| `tfidf  'good life'`       |       0.032 |      0.006 |
| `bm25   'good life'`       |       0.043 |      0.000 |
| `tfidf  'good life friends'`|      0.045 |      0.004 |
| `bm25   'good life friends'`|      0.062 |      0.001 |

**Reading the table:** BM25 is ~25–40 % slower than TF-IDF per term,
which lines up with the formula being slightly more expensive (an
extra `f * (k1 + 1) / (f + k1 · (1 − b + b · dl/avgdl))` saturating
ratio per term). At sub-100 µs per query over a 50-doc corpus the
difference is irrelevant in absolute terms; the relative cost is
worth knowing if the corpus grew by 100×. The trade-off chosen is
**ranking quality over a vanishingly small amount of extra CPU** —
see the BM25 saturation point in the rationale table above.

### 2. `suggest()` latency vs vocabulary size

Using `gud` as the partial — a typo of `good` with no prefix overlap,
so the fuzzy branch is forced to consider every word in the index:

| Vocabulary size | Median (ms) | Stdev (ms) |
|-----------------|-------------|------------|
|              50 |       0.035 |      0.002 |
|             200 |       0.052 |      0.001 |
|             800 |       0.118 |      0.012 |
|            3200 |       0.393 |      0.022 |

**Reading the table:** ~4× the vocabulary gives ~3× the time — slightly
sub-linear because the `max_dist` early-exit short-circuits more
distant words before they finish the full DP. This matches the
`O(V · p · max_dist)` bound from the complexity table. At 3200 words
we're at ~400 µs, well below human-perceptible latency for an
interactive shell.

### 3. Indexing throughput

Built **50 docs × 400 tokens in ≈16 ms ≈ 3 100 pages/sec** on the
same machine. The quotes.toscrape.com corpus is ~50 pages, so the
indexing stage takes a small fraction of a second once the HTML is
in memory — the wall-clock crawl is dominated by the politeness
window (6 s × N pages), exactly as expected.

---

## References

- Python `requests` documentation — <https://docs.python-requests.org/>
- BeautifulSoup 4 documentation — <https://www.crummy.com/software/BeautifulSoup/bs4/doc/>
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*,
  CUP 2008 — Chapters 1–6 (inverted indexing, TF-IDF, phrase queries),
  Chapter 11 (BM25 / probabilistic IR), Chapter 3 (edit distance and
  spell correction).
- Robertson, S. & Zaragoza, H. (2009), *The Probabilistic Relevance
  Framework: BM25 and Beyond*, Foundations and Trends in Information
  Retrieval — primary reference for the BM25 formulation used.
- Levenshtein, V. I. (1966), *Binary codes capable of correcting
  deletions, insertions, and reversals* — origin of the edit-distance
  algorithm used by `suggest`.
- Norvig, P. (2007), *How to Write a Spelling Corrector* —
  <https://norvig.com/spell-correct.html>. Source of the "rank by
  corpus frequency among candidates at equal edit distance" heuristic
  used to break ties in `suggest`.

---

## GenAI declaration

See the accompanying video demonstration for the declaration of GenAI
tool usage and critical evaluation, as required by the assessment
brief.
