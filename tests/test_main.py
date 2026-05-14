import pytest
from unittest.mock import patch, MagicMock
from src.indexer import Indexer
from src.search import SearchEngine
import src.main as main_module


class TestRunBuild:

    @patch('src.main.Crawler')
    def test_build_calls_crawl(self, MockCrawler):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {}
        MockCrawler.return_value = mock_crawler
        indexer = Indexer()
        with patch('src.main.os.makedirs'):
            with patch.object(indexer, 'save'):
                main_module.run_build(indexer)
        mock_crawler.crawl.assert_called_once()

    @patch('src.main.Crawler')
    def test_build_returns_search_engine(self, MockCrawler):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {}
        MockCrawler.return_value = mock_crawler
        indexer = Indexer()
        with patch('src.main.os.makedirs'):
            with patch.object(indexer, 'save'):
                result = main_module.run_build(indexer)
        assert type(result).__name__ == 'SearchEngine'

    @patch('src.main.Crawler')
    def test_build_indexes_crawled_pages(self, MockCrawler):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "http://a.com": "<p>hello world</p>"
        }
        MockCrawler.return_value = mock_crawler
        indexer = Indexer()
        with patch('src.main.os.makedirs'):
            with patch.object(indexer, 'save'):
                main_module.run_build(indexer)
        assert indexer.doc_count == 1


class TestRunLoad:

    def test_load_returns_none_when_no_file(self):
        indexer = Indexer()
        with patch('src.main.os.path.exists', return_value=False):
            result = main_module.run_load(indexer)
        assert result is None

    def test_load_returns_engine_when_file_exists(self):
        indexer = Indexer()
        with patch('src.main.os.path.exists', return_value=True):
            with patch.object(indexer, 'load'):
                result = main_module.run_load(indexer)
        assert type(result).__name__ == 'SearchEngine'

    def test_load_prints_message_when_no_file(self, capsys):
        indexer = Indexer()
        with patch('src.main.os.path.exists', return_value=False):
            main_module.run_load(indexer)
        captured = capsys.readouterr()
        assert "build" in captured.out.lower()

    def test_load_calls_indexer_load_when_file_exists(self):
        indexer = Indexer()
        with patch('src.main.os.path.exists', return_value=True):
            with patch.object(indexer, 'load') as mock_load:
                main_module.run_load(indexer)
        mock_load.assert_called_once()


class TestRunPrint:

    def setup_method(self):
        indexer = Indexer()
        indexer.index_page("http://a.com", "<p>hello world</p>")
        self.engine = SearchEngine(indexer)
        self.indexer = indexer

    def test_no_argument_shows_usage(self, capsys):
        main_module.run_print(self.engine, self.indexer, "")
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_empty_index_shows_message(self, capsys):
        empty_indexer = Indexer()
        engine = SearchEngine(empty_indexer)
        main_module.run_print(engine, empty_indexer, "hello")
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_valid_word_shows_url(self, capsys):
        main_module.run_print(self.engine, self.indexer, "hello")
        captured = capsys.readouterr()
        assert "http://a.com" in captured.out

    def test_missing_word_shows_not_found(self, capsys):
        main_module.run_print(self.engine, self.indexer, "xyznotaword")
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()


class TestRunFind:

    def setup_method(self):
        indexer = Indexer()
        indexer.index_page(
            "http://a.com", "<p>good life lived well</p>")
        indexer.index_page(
            "http://b.com", "<p>good friends forever</p>")
        self.engine = SearchEngine(indexer)
        self.indexer = indexer

    def test_no_argument_shows_usage(self, capsys):
        main_module.run_find(self.engine, self.indexer, "")
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_empty_index_shows_message(self, capsys):
        empty_indexer = Indexer()
        engine = SearchEngine(empty_indexer)
        main_module.run_find(engine, empty_indexer, "good")
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_valid_query_shows_results(self, capsys):
        main_module.run_find(self.engine, self.indexer, "good")
        captured = capsys.readouterr()
        assert "http" in captured.out

    def test_no_results_shows_message(self, capsys):
        main_module.run_find(self.engine, self.indexer, "xyznotaword")
        captured = capsys.readouterr()
        assert "No pages found" in captured.out

    def test_no_results_shows_suggestions_or_expansion(self, capsys):
        # 'goo' has no direct hits. The CLI should either auto-expand
        # it (goo→good) and show results, fall back to a 'Did you mean'
        # hint, or show 'No pages found'. All three are acceptable; what
        # we never want is a silent empty response.
        main_module.run_find(self.engine, self.indexer, "goo")
        captured = capsys.readouterr()
        assert (
            "Did you mean" in captured.out
            or "No pages found" in captured.out
            or "corrected" in captured.out
        )

    def test_multi_word_query_shows_results(self, capsys):
        main_module.run_find(self.engine, self.indexer, "good life")
        captured = capsys.readouterr()
        assert "http://a.com" in captured.out

    def test_score_displayed_in_output(self, capsys):
        main_module.run_find(self.engine, self.indexer, "good")
        captured = capsys.readouterr()
        assert "Score" in captured.out

    def test_case_insensitive_find(self, capsys):
        main_module.run_find(self.engine, self.indexer, "GOOD")
        captured = capsys.readouterr()
        assert "http" in captured.out