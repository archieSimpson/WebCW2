"""Tests for the interactive shell loop in :func:`src.main.main`.

These exercise the read-eval-print loop directly using monkeypatched
``input`` so coverage of dispatch logic, the help command, and unknown
commands is captured.
"""

from unittest.mock import MagicMock, patch

import src.main as main_module


def _drive_main(inputs):
    """Run ``main()`` with a fixed sequence of ``input`` returns."""
    iterator = iter(inputs)

    def fake_input(prompt=""):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError()

    with patch('builtins.input', side_effect=fake_input):
        main_module.main()


class TestMainLoop:

    def test_main_quit_exits_cleanly(self, capsys):
        _drive_main(["quit"])
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_main_exit_alias(self, capsys):
        _drive_main(["exit"])
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_main_eof_exits_cleanly(self, capsys):
        _drive_main([])
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_main_unknown_command_message(self, capsys):
        _drive_main(["banana", "quit"])
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_main_help_command(self, capsys):
        _drive_main(["help", "quit"])
        captured = capsys.readouterr()
        assert "build" in captured.out.lower()
        assert "find" in captured.out.lower()

    def test_main_question_alias_help(self, capsys):
        _drive_main(["?", "quit"])
        captured = capsys.readouterr()
        assert "build" in captured.out.lower()

    def test_main_blank_line_skipped(self, capsys):
        _drive_main(["", "   ", "quit"])
        captured = capsys.readouterr()
        # No "Unknown command" message should appear for blank lines.
        assert "Unknown command" not in captured.out

    def test_main_print_before_build(self, capsys):
        _drive_main(["print hello", "quit"])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_main_find_before_build(self, capsys):
        _drive_main(["find hello", "quit"])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_main_load_no_file(self, capsys):
        with patch('src.main.os.path.exists', return_value=False):
            _drive_main(["load", "quit"])
        captured = capsys.readouterr()
        assert "No index found" in captured.out

    @patch('src.main.Crawler')
    def test_main_build_then_find(self, MockCrawler, capsys):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "http://a.com": "<p>hello world</p>",
        }
        MockCrawler.return_value = mock_crawler
        with patch('src.main.os.makedirs'):
            with patch('src.main.Indexer.save'):
                _drive_main(["build", "find hello", "quit"])
        captured = capsys.readouterr()
        assert "http://a.com" in captured.out

    @patch('src.main.Crawler')
    def test_main_build_then_print(self, MockCrawler, capsys):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "http://a.com": "<p>hello world</p>",
        }
        MockCrawler.return_value = mock_crawler
        with patch('src.main.os.makedirs'):
            with patch('src.main.Indexer.save'):
                _drive_main(["build", "print hello", "quit"])
        captured = capsys.readouterr()
        assert "http://a.com" in captured.out

    @patch('src.main.Crawler')
    def test_main_load_after_build_persists(self, MockCrawler, capsys):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "http://a.com": "<p>hello world</p>",
        }
        MockCrawler.return_value = mock_crawler
        with patch('src.main.os.makedirs'):
            with patch('src.main.Indexer.save'):
                with patch('src.main.os.path.exists', return_value=True):
                    with patch('src.main.Indexer.load'):
                        _drive_main(["build", "load", "quit"])
        captured = capsys.readouterr()
        assert "Index loaded" in captured.out


class TestRunFindEdgeCases:

    @patch('src.main.Crawler')
    def test_find_with_no_match_offers_suggestions_or_expansion(
            self, MockCrawler, capsys):
        # 'goo' is a partial that auto-expands to 'good' via the new
        # per-term rewrite. We allow either path (expanded results,
        # 'Did you mean' hint, or 'No pages found') — what matters is
        # that the user never gets a silent empty response.
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "http://a.com": "<p>good goodbye</p>",
        }
        MockCrawler.return_value = mock_crawler
        with patch('src.main.os.makedirs'):
            with patch('src.main.Indexer.save'):
                _drive_main(["build", "find goo", "quit"])
        captured = capsys.readouterr()
        assert ("Did you mean" in captured.out
                or "No pages found" in captured.out
                or "corrected" in captured.out)

    @patch('src.main.Crawler')
    def test_find_auto_expands_mistyped_term(
            self, MockCrawler, capsys):
        # 'gud friends' would normally return empty because 'gud' isn't
        # indexed. run_find should rewrite via expand_query and surface
        # the substitution to the user.
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "http://a.com": "<p>good friends together always</p>",
        }
        MockCrawler.return_value = mock_crawler
        with patch('src.main.os.makedirs'):
            with patch('src.main.Indexer.save'):
                _drive_main(["build", "find gud friends", "quit"])
        captured = capsys.readouterr()
        assert "gud" in captured.out  # original surfaced in notice
        assert "good" in captured.out  # replacement surfaced
        assert "http://a.com" in captured.out  # actually returned the hit
