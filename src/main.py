"""Interactive command-line shell for the COMP3011 search engine.

Provides four commands as required by the assessment brief:

* ``build`` — crawl the seed site, build an inverted index, persist it.
* ``load``  — read a previously persisted index from disk.
* ``print <word>`` — show the inverted-index entry for a single word.
* ``find <query>`` — Boolean-aware search ranked by BM25 with an
  exact-phrase bonus. Supports ``AND`` (default), ``OR`` and ``NOT``.

Run with ``python src/main.py`` (or ``python -m src.main``).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from crawler import Crawler  # noqa: E402
from indexer import Indexer  # noqa: E402
from search import SearchEngine  # noqa: E402

BASE_URL: str = "https://quotes.toscrape.com/"
INDEX_FILE: str = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'index.json')


def run_build(indexer: Indexer) -> SearchEngine:
    """Crawl the target site, populate ``indexer``, and persist to disk.

    Returns a :class:`SearchEngine` bound to the freshly built index so
    that subsequent ``find``/``print`` commands have something to query.
    """
    print(f"Crawling {BASE_URL} ...")
    crawler = Crawler(BASE_URL, politeness_window=6)
    pages = crawler.crawl()
    print(f"\nCrawled {len(pages)} pages. Building index...")
    indexer.build_from_pages(pages)
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    indexer.save(INDEX_FILE)
    engine = SearchEngine(indexer)
    print(f"Done. Indexed {len(pages)} pages, "
          f"{len(indexer.index)} unique words.")
    return engine


def run_load(indexer: Indexer) -> Optional[SearchEngine]:
    """Restore the index from :data:`INDEX_FILE`.

    Returns ``None`` (and prints a hint) if no index file exists yet,
    so the caller can decide whether to keep the stale engine.
    """
    if not os.path.exists(INDEX_FILE):
        print("No index found. Run 'build' first.")
        return None
    indexer.load(INDEX_FILE)
    engine = SearchEngine(indexer)
    print(f"Index loaded: {indexer.doc_count} documents, "
          f"{len(indexer.index)} unique words.")
    return engine


def run_print(
    engine: SearchEngine, indexer: Indexer, argument: str
) -> None:
    """Handle the ``print <word>`` shell command."""
    if not argument:
        print("Usage: print <word>")
        return
    if not indexer.index:
        print("Index is empty. Run 'build' or 'load' first.")
        return
    engine.print_index(argument)


def _query_terms(query: str) -> list[str]:
    """Return the non-operator terms used for snippet highlighting."""
    return [
        tok.lower() for tok in query.split()
        if tok not in ("AND", "OR", "NOT")
    ]


def run_find(
    engine: SearchEngine, indexer: Indexer, argument: str
) -> None:
    """Handle the ``find <query>`` shell command."""
    if not argument:
        print("Usage: find <word> [word2 ...]   "
              "(AND/OR/NOT supported, uppercase)")
        return
    if not indexer.index:
        print("Index is empty. Run 'build' or 'load' first.")
        return
    results = engine.find(argument)
    if not results:
        print(f"No pages found for '{argument}'.")
        terms = _query_terms(argument)
        if terms:
            suggestions = engine.suggest(terms[0])
            if suggestions:
                print(f"Did you mean: {', '.join(suggestions)}?")
        return

    terms = _query_terms(argument)
    print(f"\nFound {len(results)} page(s) for '{argument}':")
    print(f"  {'Score':>8}  URL")
    print(f"  {'-' * 70}")
    for url, score in results:
        print(f"  {score:>8.4f}  {url}")
        snippet = indexer.get_snippet(url, terms)
        if snippet:
            print(f"            {snippet}")
    print()


def main() -> None:
    """Run the read-eval-print loop until the user exits."""
    indexer = Indexer()
    engine = SearchEngine(indexer)

    print("=" * 60)
    print("  COMP3011 Search Engine")
    print("  Commands: build | load | print <word> | find <query>")
    print("  Query operators: AND (default), OR, NOT (uppercase)")
    print("  Type 'help' for more, 'quit' to exit")
    print("=" * 60)
    print()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""

        if command in ("quit", "exit"):
            print("Goodbye.")
            break
        elif command == "build":
            engine = run_build(indexer)
        elif command == "load":
            result = run_load(indexer)
            if result:
                engine = result
        elif command == "print":
            run_print(engine, indexer, argument)
        elif command == "find":
            run_find(engine, indexer, argument)
        elif command in ("help", "?"):
            print(
                "Commands:\n"
                "  build           Crawl the site and build the index.\n"
                "  load            Load a previously saved index.\n"
                "  print <word>    Show the inverted index for a word.\n"
                "  find <query>    Search the index. Supports:\n"
                "                    'find good friends'   (implicit AND)\n"
                "                    'find good OR life'   (union)\n"
                "                    'find good NOT enemy' (exclude)\n"
                "                  Operators are uppercase only.\n"
                "  quit            Exit the shell."
            )
        else:
            print(f"Unknown command: '{command}'.")
            print("Try: build, load, print <word>, find <query>")


if __name__ == "__main__":
    main()
