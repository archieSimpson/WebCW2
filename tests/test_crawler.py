import pytest
from unittest.mock import patch, MagicMock, Mock
import requests
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


class TestCrawlerNormalise:

    def setup_method(self):
        with patch('src.crawler.urllib.robotparser.RobotFileParser'):
            self.crawler = Crawler("https://quotes.toscrape.com/")

    def test_strips_fragment(self):
        result = self.crawler._normalise_url(
            "https://quotes.toscrape.com/page/1/#section")
        assert '#' not in result

    def test_strips_query_string(self):
        result = self.crawler._normalise_url(
            "https://quotes.toscrape.com/page/1/?ref=home")
        assert '?' not in result

    def test_clean_url_unchanged(self):
        url = "https://quotes.toscrape.com/page/1/"
        assert self.crawler._normalise_url(url) == url

    def test_strips_both_fragment_and_query(self):
        result = self.crawler._normalise_url(
            "https://quotes.toscrape.com/page/1/?ref=home#section")
        assert '#' not in result
        assert '?' not in result


class TestCrawlerRobots:

    def setup_method(self):
        with patch('src.crawler.urllib.robotparser.RobotFileParser'):
            self.crawler = Crawler("https://quotes.toscrape.com/")

    def test_allowed_returns_true_when_can_fetch_true(self):
        self.crawler.rp.can_fetch = Mock(return_value=True)
        assert self.crawler._is_allowed(
            "https://quotes.toscrape.com/any") is True

    def test_allowed_returns_false_when_can_fetch_false(self):
        self.crawler.rp.can_fetch = Mock(return_value=False)
        assert self.crawler._is_allowed(
            "https://quotes.toscrape.com/any") is False

    def test_allowed_exception_returns_true(self):
        self.crawler.rp.can_fetch = Mock(side_effect=Exception("error"))
        assert self.crawler._is_allowed(
            "https://quotes.toscrape.com/any") is True


class TestCrawlerCrawl:

    def setup_method(self):
        with patch('src.crawler.urllib.robotparser.RobotFileParser'):
            self.crawler = Crawler("https://quotes.toscrape.com/",
                                   politeness_window=0)

    @patch('src.crawler.time.sleep')
    def test_crawl_returns_dict(self, mock_sleep):
        mock_response = Mock()
        mock_response.text = '<html><body><p>hello</p></body></html>'
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status = Mock()
        self.crawler.session.get = Mock(return_value=mock_response)
        result = self.crawler.crawl()
        assert isinstance(result, dict)

    @patch('src.crawler.time.sleep')
    def test_crawl_stores_page_content(self, mock_sleep):
        mock_response = Mock()
        mock_response.text = '<html><body><p>hello</p></body></html>'
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status = Mock()
        self.crawler.session.get = Mock(return_value=mock_response)
        result = self.crawler.crawl()
        assert "https://quotes.toscrape.com/" in result

    @patch('src.crawler.time.sleep')
    def test_crawl_skips_non_html(self, mock_sleep):
        mock_response = Mock()
        mock_response.text = 'raw content'
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.raise_for_status = Mock()
        self.crawler.session.get = Mock(return_value=mock_response)
        result = self.crawler.crawl()
        assert len(result) == 0

    @patch('src.crawler.time.sleep')
    def test_crawl_handles_request_exception(self, mock_sleep):
        self.crawler.session.get = Mock(
            side_effect=requests.RequestException("timeout"))
        result = self.crawler.crawl()
        assert result == {}

    @patch('src.crawler.time.sleep')
    def test_crawl_does_not_revisit_pages(self, mock_sleep):
        mock_response = Mock()
        mock_response.text = (
            '<html><body>'
            '<a href="https://quotes.toscrape.com/page/2/">next</a>'
            '</body></html>'
        )
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status = Mock()
        self.crawler.visited.add("https://quotes.toscrape.com/page/2/")
        self.crawler.session.get = Mock(return_value=mock_response)
        self.crawler.crawl()
        assert self.crawler.session.get.call_count == 1

    @patch('src.crawler.time.sleep')
    def test_crawl_respects_politeness_window(self, mock_sleep):
        mock_response = Mock()
        mock_response.text = '<html><body></body></html>'
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status = Mock()
        with patch('src.crawler.urllib.robotparser.RobotFileParser'):
            c = Crawler("https://quotes.toscrape.com/", politeness_window=6)
        c.session.get = Mock(return_value=mock_response)
        c.crawl()
        mock_sleep.assert_called_with(6)