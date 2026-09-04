"""Guards that must be non-bypassable on the counted path (trend_v1 driver) and the
value-aware registered-cell check of the adjudicator. Pure-function tests; nothing is run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.scripts import trend_adjudicate, trend_wfo


def _ledger_with_rows(path: Path, n: int) -> Path:
    path.write_text("".join(json.dumps({"oos_periodic_sharpe": 0.01 * i, "output_dir": f"d{i}"}) + "\n" for i in range(n)))
    return path


def test_budget_guard_is_exactly_six_on_the_counted_path(tmp_path: Path) -> None:
    ledger = _ledger_with_rows(tmp_path / "ledger.jsonl", 6)
    with pytest.raises(SystemExit, match="exactly 6"):
        trend_wfo._guard_budget(ledger, 7, counted=True)
    with pytest.raises(SystemExit, match="exactly 6"):
        trend_wfo._guard_budget(ledger, 100, counted=True)
    with pytest.raises(SystemExit, match="exhausted"):
        trend_wfo._guard_budget(ledger, 6, counted=True)
    trend_wfo._guard_budget(_ledger_with_rows(tmp_path / "empty.jsonl", 0), 6, counted=True)
    trend_wfo._guard_budget(_ledger_with_rows(tmp_path / "five.jsonl", 5), 6, counted=True)


@pytest.mark.parametrize(
    "argv,label",
    [
        ([], "T0"),
        (["--lookback", "126"], "T1"),
        (["--skip", "0"], "T2"),
        (["--decision_every", "63"], "T3"),
        (["--sizing", "equal_notional"], "T4"),
        (["--allow_post_ratification", "--end_date", "2027-07-20"], "T5"),
    ],
)
def test_registered_cells_are_accepted_on_the_counted_path(argv: list[str], label: str) -> None:
    assert trend_wfo._guard_registered_cell(trend_wfo.parse_args(argv), counted=True) == label


@pytest.mark.parametrize(
    "argv",
    [
        ["--lookback", "200"],
        ["--skip", "5"],
        ["--decision_every", "42"],
        ["--lookback", "126", "--skip", "0"],
        ["--allow_post_ratification", "--end_date", "2027-07-20", "--lookback", "126"],
    ],
)
def test_unregistered_cells_are_refused_on_the_counted_path(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="not a registered cell"):
        trend_wfo._guard_registered_cell(trend_wfo.parse_args(argv), counted=True)


def test_distributions_dir_is_provenance_not_a_knob() -> None:
    # a different data location changes the hash but not cell membership (per-file sha256s carry provenance)
    args = trend_wfo.parse_args(["--distributions_dir", "/tmp/not/the/pinned/dir"])
    assert trend_wfo._guard_registered_cell(args, counted=True) == "T0"
    assert trend_wfo._config_hash(trend_wfo._config_payload(args)) != trend_wfo._config_hash(
        trend_wfo._config_payload(trend_wfo.parse_args([]))
    )


def test_unregistered_cells_pass_uncounted_with_a_label() -> None:
    assert trend_wfo._guard_registered_cell(trend_wfo.parse_args(["--lookback", "200"]), counted=False) == "uncounted:unregistered"
    assert trend_wfo._guard_registered_cell(trend_wfo.parse_args([]), counted=False) == "T0"


def test_distributions_dir_hash_is_spelling_invariant() -> None:
    a = trend_wfo._config_hash(trend_wfo._config_payload(trend_wfo.parse_args([])))
    b = trend_wfo._config_hash(trend_wfo._config_payload(trend_wfo.parse_args(["--distributions_dir", "./data/distributions"])))
    c = trend_wfo._config_hash(
        trend_wfo._config_payload(trend_wfo.parse_args(["--distributions_dir", str(Path("data/distributions").resolve())]))
    )
    assert a == b == c


def test_uncounted_rehearsals_never_read_the_pinned_cache(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="pinned cache"):
        trend_wfo._guard_uncounted_data(trend_wfo.parse_args(["--data_dir", "data"]), counted=False)
    with pytest.raises(SystemExit, match="pinned cache"):
        trend_wfo._guard_uncounted_data(trend_wfo.parse_args(["--data_dir", str(Path("data").resolve() / "sub")]), counted=False)
    trend_wfo._guard_uncounted_data(trend_wfo.parse_args(["--data_dir", str(tmp_path)]), counted=False)
    trend_wfo._guard_uncounted_data(trend_wfo.parse_args(["--data_dir", "data"]), counted=True)


def _write_cfg(root: Path, name: str, argv: list[str]) -> Path:
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(trend_wfo._config_payload(trend_wfo.parse_args(argv)), default=str))
    return d


def test_adjudicator_single_knob_check_is_value_aware(tmp_path: Path) -> None:
    t0 = _write_cfg(tmp_path, "t0", [])
    good = [
        _write_cfg(tmp_path, "t1", ["--lookback", "126"]),
        _write_cfg(tmp_path, "t2", ["--skip", "0"]),
        _write_cfg(tmp_path, "t3", ["--decision_every", "63"]),
        _write_cfg(tmp_path, "t4", ["--sizing", "equal_notional"]),
    ]
    assert trend_adjudicate._single_knob_problems(t0, good) == []
    bad = [
        _write_cfg(tmp_path, "b1", ["--lookback", "200"]),
        _write_cfg(tmp_path, "b2", ["--skip", "5"]),
        _write_cfg(tmp_path, "b3", ["--decision_every", "42"]),
        _write_cfg(tmp_path, "b4", ["--sizing", "equal_notional"]),
    ]
    problems = trend_adjudicate._single_knob_problems(t0, bad)
    assert len(problems) == 3 and all("registered value" in x for x in problems)
    wrong_t0 = _write_cfg(tmp_path, "t0bad", ["--lookback", "126"])
    assert any("pinned" in x for x in trend_adjudicate._single_knob_problems(wrong_t0, good))


def test_fragility_read_requires_the_ledger(tmp_path: Path) -> None:
    assert trend_adjudicate._ledger_order_problems(None, tmp_path, []) != []
