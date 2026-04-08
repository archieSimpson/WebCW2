import pytest
from unittest.mock import patch
from src.crawler import Crawler


class TestCrawlerInit:

    def setup_method(self):
        with patch('src.crawler.urllib.robotparser.RobotFileParser'):
            self.crawler = Crawler("https://quotes.toscrape.com/")

    def test_base_url_stored(self):
        assert self.crawler.base_url == "https://quotes.toscrape.com/"

    def test_default_politeness_window(self):
        with patch('src.crawler.urllib.robotparser.RobotFileParser'):
            c = Crawler("https://quotes.toscrape.com/")
        assert c.politeness_window == 6

    def test_custom_politeness_window(self):
        with patch('src.crawler.urllib.robotparser.RobotFileParser'):
            c = Crawler("https://quotes.toscrape.com/", politeness_window=10)
        assert c.politeness_window == 10

    def test_pages_starts_empty(self):
        assert self.crawler.pages == {}

    def test_visited_starts_empty_before_crawl(self):
        assert isinstance(self.crawler.visited, set)

    def test_session_has_user_agent(self):
        assert 'User-Agent' in self.crawler.session.headers


class TestCrawlerDomainCheck:

    def setup_method(self):
        with patch('src.crawler.urllib.robotparser.RobotFileParser'):
            self.crawler = Crawler("https://quotes.toscrape.com/")

    def test_same_domain_subpath(self):
        assert self.crawler._is_same_domain(
            "https://quotes.toscrape.com/page/2/") is True

    def test_external_domain_false(self):
        assert self.crawler._is_same_domain(
            "https://external.com/") is False

    def test_google_false(self):
        assert self.crawler._is_same_domain(
            "https://google.com") is False

    def test_same_domain_deep_path(self):
        assert self.crawler._is_same_domain(
            "https://quotes.toscrape.com/author/Albert-Einstein/") is True

    def test_completely_different_domain(self):
        assert self.crawler._is_same_domain(
            "https://wikipedia.org/") is False
