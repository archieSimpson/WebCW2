"""Integration tests for the full crawler -> indexer -> search pipeline.

These tests fabricate a small "site" of HTML pages, drive the
:class:`Crawler` against a mocked ``requests.Session.get``, build a
real :class:`Indexer`, persist it to a temporary file, reload it, and
then run searches end-to-end. Network I/O is mocked; everything else
exercises real code.
"""

import os
import tempfile
from unittest.mock import Mock, patch

from src.crawler import Crawler
from src.indexer import Indexer
from src.search import SearchEngine


def _html(body: str) -> str:
    """Wrap ``body`` in a minimal HTML document."""
    return f"<html><body>{body}</body></html>"


def _mock_response(body: str) -> Mock:
    """Return a ``requests.Response``-shaped mock for ``body``."""
    r = Mock()
    r.text = body
    r.headers = {'Content-Type': 'text/html; charset=utf-8'}
    r.raise_for_status = Mock()
    return r


class TestCrawlToSearchPipeline:
    """End-to-end tests with a mocked requests session."""

    def setup_method(self):
        self.pages = {
            "https://quotes.toscrape.com/": _html(
                '<p>good life lived well</p>'
                '<a href="https://quotes.toscrape.com/page/2/">next</a>'
            ),
            "https://quotes.toscrape.com/page/2/": _html(
                '<p>good friends together always</p>'
                '<a href="https://quotes.toscrape.com/author/x/">x</a>'
            ),
            "https://quotes.toscrape.com/author/x/": _html(
                '<p>indifference is unique</p>'
            ),
        }

    def _fake_get(self, url, timeout=None):
        return _mock_response(self.pages[url])

    @patch('src.crawler.time.sleep')
    @patch('src.crawler.urllib.robotparser.RobotFileParser')
    def test_full_pipeline_finds_word(self, _rp, _sleep):
        crawler = Crawler(
            "https://quotes.toscrape.com/", politeness_window=0)
        crawler.session.get = Mock(side_effect=self._fake_get)
        crawled = crawler.crawl()

        indexer = Indexer()
        indexer.build_from_pages(crawled)
        engine = SearchEngine(indexer)

        results = engine.find("indifference")
        assert "https://quotes.toscrape.com/author/x/" in [
            url for url, _ in results
        ]

    @patch('src.crawler.time.sleep')
    @patch('src.crawler.urllib.robotparser.RobotFileParser')
    def test_full_pipeline_phrase_query(self, _rp, _sleep):
        crawler = Crawler(
            "https://quotes.toscrape.com/", politeness_window=0)
        crawler.session.get = Mock(side_effect=self._fake_get)

        indexer = Indexer()
        indexer.build_from_pages(crawler.crawl())
        engine = SearchEngine(indexer)

        results = engine.find("good friends")
        urls = [url for url, _ in results]
        assert "https://quotes.toscrape.com/page/2/" in urls

    @patch('src.crawler.time.sleep')
    @patch('src.crawler.urllib.robotparser.RobotFileParser')
    def test_pipeline_persists_and_reloads(self, _rp, _sleep):
        crawler = Crawler(
            "https://quotes.toscrape.com/", politeness_window=0)
        crawler.session.get = Mock(side_effect=self._fake_get)

        original = Indexer()
        original.build_from_pages(crawler.crawl())

        with tempfile.NamedTemporaryFile(
                suffix='.json', delete=False) as f:
            path = f.name
        try:
            original.save(path)
            reloaded = Indexer()
            reloaded.load(path)
        finally:
            os.unlink(path)

        engine = SearchEngine(reloaded)
        results_before = SearchEngine(original).find("good friends")
        results_after = engine.find("good friends")
        assert {u for u, _ in results_before} == {
            u for u, _ in results_after}


class TestPipelineRobustness:

    @patch('src.crawler.time.sleep')
    @patch('src.crawler.urllib.robotparser.RobotFileParser')
    def test_crawl_skips_external_links(self, _rp, _sleep):
        body = _html(
            '<a href="https://external.com/">leave</a>'
            '<p>only this</p>'
        )
        crawler = Crawler(
            "https://quotes.toscrape.com/", politeness_window=0)
        crawler.session.get = Mock(return_value=_mock_response(body))
        crawled = crawler.crawl()
        assert all('quotes.toscrape.com' in u for u in crawled)

    @patch('src.crawler.time.sleep')
    @patch('src.crawler.urllib.robotparser.RobotFileParser')
    def test_crawl_normalises_url_with_query(self, _rp, _sleep):
        body = _html(
            '<a href="/?ref=home">a</a>'
            '<p>hello</p>'
        )
        crawler = Crawler(
            "https://quotes.toscrape.com/", politeness_window=0)
        crawler.session.get = Mock(return_value=_mock_response(body))
        crawler.crawl()
        # The seed URL is added to ``visited`` already; verify the
        # normalisation doesn't enqueue ``/?ref=home`` as a duplicate.
        assert all('?' not in u for u in crawler.visited)

    @patch('src.crawler.time.sleep')
    @patch('src.crawler.urllib.robotparser.RobotFileParser')
    def test_max_pages_cap_respected(self, _rp, _sleep):
        body = _html(
            '<a href="/page/2/">2</a>'
            '<a href="/page/3/">3</a>'
            '<a href="/page/4/">4</a>'
            '<p>x</p>'
        )
        crawler = Crawler(
            "https://quotes.toscrape.com/",
            politeness_window=0,
            max_pages=2,
        )
        crawler.session.get = Mock(return_value=_mock_response(body))
        crawled = crawler.crawl()
        assert len(crawled) <= 2

    def test_index_round_trip_preserves_tfidf_ordering(self):
        original = Indexer()
        original.index_page(
            "http://a.com", "<p>fish fish fish water sea coral</p>")
        original.index_page(
            "http://b.com", "<p>fish water sea coral reef shell</p>")

        with tempfile.NamedTemporaryFile(
                suffix='.json', delete=False) as f:
            path = f.name
        try:
            original.save(path)
            reloaded = Indexer()
            reloaded.load(path)
        finally:
            os.unlink(path)

        engine_before = SearchEngine(original)
        engine_after = SearchEngine(reloaded)
        assert engine_before.find("fish") == engine_after.find("fish")


class TestCrawlerRobotsBranches:
    """Cover the robots.txt-disallow branch of the crawl loop."""

    @patch('src.crawler.time.sleep')
    @patch('src.crawler.urllib.robotparser.RobotFileParser')
    def test_disallowed_url_is_skipped(self, _rp, _sleep, capsys):
        crawler = Crawler(
            "https://quotes.toscrape.com/", politeness_window=0)
        crawler.rp.can_fetch = Mock(return_value=False)
        crawler.session.get = Mock()
        crawler.crawl()
        # The seed was disallowed, so no GET should fire.
        crawler.session.get.assert_not_called()
        captured = capsys.readouterr()
        assert "robots.txt" in captured.out

    @patch('src.crawler.urllib.robotparser.RobotFileParser')
    def test_robots_txt_load_failure_is_non_fatal(self, MockRP):
        # Simulate a robots.txt fetch failure: read() raises.
        instance = Mock()
        instance.read = Mock(side_effect=Exception("network down"))
        MockRP.return_value = instance
        # Constructor must not raise.
        crawler = Crawler("https://quotes.toscrape.com/")
        assert crawler.base_url == "https://quotes.toscrape.com/"


class TestIndexFileSchema:

    def test_load_rejects_future_schema_version(self):
        import json
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'schema_version': 999,
                'index': {},
                'doc_count': 0,
                'doc_lengths': {},
            }, f)
            path = f.name
        try:
            indexer = Indexer()
            try:
                indexer.load(path)
                assert False, "expected ValueError"
            except ValueError:
                pass
        finally:
            os.unlink(path)

    def test_load_accepts_legacy_schema_without_version(self):
        import json
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'index': {'hello': {'http://a.com': {
                    'count': 1, 'positions': [0]}}},
                'doc_count': 1,
                'doc_lengths': {'http://a.com': 1},
            }, f)
            path = f.name
        try:
            indexer = Indexer()
            indexer.load(path)
            assert indexer.doc_count == 1
            assert 'hello' in indexer.index
        finally:
            os.unlink(path)
