"""Unit tests for the account-size-floor ledger arithmetic (research tier)."""

import json

import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts.account_size_floor import (  # noqa: E402
    achieved_positions_by_refresh,
    refresh_concordance,
    summarize,
    summarize_run,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def replay_dir(tmp_path):
    """A two-refresh synthetic replay run.

    Refresh 1 targets a 0.5/-0.5/0.02 book at $10k; the 0.02 leg ($200) is
    below one $400 share, so the venue never sees it (the dust-censoring
    class under measurement). Refresh 2 exits AAA. A stray DDD fill exercises
    the unpriced-residual reporting path.
    """
    run = tmp_path / "run"
    run.mkdir()
    _write_jsonl(
        run / "targets.jsonl",
        [
            {
                "refresh_bar": "2026-01-02",
                "decision_bar": "2026-01-02",
                "equity": 10_000.0,
                "targets": {"AAA": 0.5, "BBB": -0.5, "CCC": 0.02},
                "reference_prices": {"AAA": 100.0, "BBB": 50.0, "CCC": 400.0},
            },
            {
                "refresh_bar": "2026-02-02",
                "decision_bar": "2026-02-02",
                "equity": 10_000.0,
                "targets": {"BBB": -0.5},
                "reference_prices": {"BBB": 50.0},
            },
        ],
    )
    _write_jsonl(
        run / "fills.jsonl",
        [
            {"client_order_id": "a", "symbol": "AAA", "qty": 50.0, "fill_price": 100.0,
             "reference_price": 100.0, "decision_bar": "2026-01-02", "filled_bar": "2026-01-03"},
            {"client_order_id": "b", "symbol": "BBB", "qty": -100.0, "fill_price": 50.0,
             "reference_price": 50.0, "decision_bar": "2026-01-02", "filled_bar": "2026-01-03"},
            {"client_order_id": "d", "symbol": "DDD", "qty": 1.0, "fill_price": 10.0,
             "reference_price": 10.0, "decision_bar": "2026-01-02", "filled_bar": "2026-01-03"},
            {"client_order_id": "c", "symbol": "AAA", "qty": -50.0, "fill_price": 100.0,
             "reference_price": 100.0, "decision_bar": "2026-02-02", "filled_bar": "2026-02-03"},
        ],
    )
    _write_jsonl(
        run / "equity.jsonl",
        [
            {"decision_bar": "2026-01-02", "equity": 10_000.0, "cash": 10_000.0},
            {"decision_bar": "2026-02-02", "equity": 11_000.0, "cash": 500.0},
        ],
    )
    from prism.live.state import LoopState, StateStore

    StateStore(run / "state.json").save(
        LoopState(positions={"BBB": -100.0, "DDD": 1.0}, cash=500.0)
    )
    return run


def test_achieved_positions_accumulate_per_refresh(replay_dir):
    from prism.live import read_fills_ledger, read_targets_ledger

    books = achieved_positions_by_refresh(
        read_targets_ledger(replay_dir / "targets.jsonl"),
        read_fills_ledger(replay_dir / "fills.jsonl"),
    )
    assert books[0] == {"AAA": 50.0, "BBB": -100.0, "DDD": 1.0}
    assert books[1] == {"BBB": -100.0, "DDD": 1.0}  # AAA exited to exactly zero


def test_refresh_concordance_measures_censoring(replay_dir):
    from prism.live import read_fills_ledger, read_targets_ledger

    rows = read_targets_ledger(replay_dir / "targets.jsonl")
    books = achieved_positions_by_refresh(rows, read_fills_ledger(replay_dir / "fills.jsonl"))
    first = refresh_concordance(rows[0], books[0])
    # AAA and BBB replicate exactly; the censored CCC leg is the whole gap.
    assert first["active_share"] == pytest.approx(0.5 * 0.02)
    assert first["censored"] == ["CCC"] and first["censored_names"] == 1
    assert first["gross_held"] == pytest.approx(1.0)
    assert first["gross_target"] == pytest.approx(1.02)
    assert first["unpriced_names"] == ["DDD"]  # reported, never silently dropped
    second = refresh_concordance(rows[1], books[1])
    assert second["active_share"] == pytest.approx(0.0)
    assert second["censored_names"] == 0


def test_summarize_run_terminal_stats(replay_dir):
    summary = summarize_run(replay_dir)
    assert summary["sessions"] == 2
    assert summary["total_return"] == pytest.approx(0.10)
    assert summary["final_names"] == 2
    assert summary["active_share_mean"] == pytest.approx(0.5 * 0.02 / 2)
    assert summary["censored_names_mean"] == pytest.approx(0.5)


def test_summarize_gap_vs_largest_cash_baseline(replay_dir):
    result = summarize({10_000: replay_dir, 1_000_000: replay_dir})
    assert result["baseline_cash"] == 1_000_000
    assert result["runs"]["10000"]["return_gap_vs_baseline"] == pytest.approx(0.0)
    assert set(result["runs"]) == {"10000", "1000000"}
