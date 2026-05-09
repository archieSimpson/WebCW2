"""Interactive command-line shell for the COMP3011 search engine."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from crawler import Crawler  # noqa: E402
from indexer import Indexer  # noqa: E402
from search import SearchEngine  # noqa: E402

BASE_URL = "https://quotes.toscrape.com/"
INDEX_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'index.json')


def run_build(indexer):
    """Crawl the target site, populate indexer, and persist to disk."""
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
