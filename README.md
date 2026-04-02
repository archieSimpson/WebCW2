# Quotes Search Engine

A small web crawler, indexer, and search tool for
[`https://quotes.toscrape.com/`](https://quotes.toscrape.com/), built
for **COMP3011 — Web Services and Web Data, Coursework 2**.

The plan is the classic three-stage pipeline:

1. **Crawler** — polite BFS over the target site
2. **Indexer** — extract text, build an inverted index
3. **Search** — single + multi-word lookup, ranked

More to come as it lands.
