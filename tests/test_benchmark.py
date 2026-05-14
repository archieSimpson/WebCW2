"""Smoke tests for the benchmark harness.

We don't assert specific timings (CI machines vary wildly); we just
verify the harness runs end-to-end on a tiny corpus and emits the
expected sections. Catches structural regressions like a renamed
function or a broken import without making the suite flaky.
"""

import sys
from pathlib import Path

# Add benchmarks/ to sys.path for the test (the package isn't on the
# normal import path because it isn't shipped as a library).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import benchmark as bench


class TestBenchmarkHarness:

    def test_synthetic_index_has_requested_doc_count(self):
        idx = bench._build_synthetic_index(
            n_docs=5, doc_length=20, vocab_size=30
        )
        assert idx.doc_count == 5
        assert len(idx.doc_lengths) == 5

    def test_synthetic_index_doc_length_matches(self):
        idx = bench._build_synthetic_index(
            n_docs=3, doc_length=40, vocab_size=20
        )
        for length in idx.doc_lengths.values():
            assert length == 40

    def test_bench_rankers_returns_six_rows(self):
        # 3 query lengths × 2 rankers = 6 rows.
        rows = bench.bench_rankers(n_docs=5, doc_length=30, iterations=3)
        assert len(rows) == 6
        assert all(med >= 0.0 for _, med, _ in rows)

    def test_bench_rankers_emits_both_rankers(self):
        rows = bench.bench_rankers(n_docs=5, doc_length=30, iterations=3)
        labels = [label for label, _, _ in rows]
        assert any("tfidf" in lbl for lbl in labels)
        assert any("bm25" in lbl for lbl in labels)

    def test_bench_suggest_returns_one_row_per_vocab(self):
        rows = bench.bench_suggest([10, 50], iterations=3)
        assert len(rows) == 2
        assert rows[0][0] == 10 and rows[1][0] == 50
        assert all(med > 0.0 for _, med, _ in rows)

    def test_bench_indexing_reports_positive_throughput(self):
        total, pps, _ = bench.bench_indexing(
            n_docs=5, doc_length=20, iterations=2
        )
        assert total > 0
        assert pps > 0

    def test_main_runs_in_quick_mode(self, capsys):
        # End-to-end smoke through the argparse entry point.
        bench.main(["--quick", "--no-table"])
        captured = capsys.readouterr()
        assert "TF-IDF vs BM25" in captured.out
        assert "suggest()" in captured.out
        assert "Indexing throughput" in captured.out

    def test_main_emits_markdown_table_without_flag(self, capsys):
        bench.main(["--quick"])
        captured = capsys.readouterr()
        assert "## Ranker comparison" in captured.out
        assert "## Suggest latency vs vocabulary" in captured.out
