"""The per-session bar-panel log (``prism.live.panel_log``).

What these guard: a recorded panel comes back cell for cell after any number of
deltas, including a revised historical bar, a bar that left the trailing window
and a symbol that joined or left; a digest that does not match is refused rather
than returned; recording the same decision twice is a no-op for the same content
and an error for different content; and the base/delta cadence keeps the chain
short. Sizes are asserted loosely (a delta is a small fraction of a base) so the
test does not pin a compressor's output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.live import panel_log
from prism.live.panel_log import (
    PanelDigestMismatch,
    PanelLogConflict,
    PanelLogError,
    canonicalize,
    latest_record,
    read_index,
    reconstruct_panel,
    record_panel,
)


def _panel(
    n_bars: int = 30,
    n_symbols: int = 6,
    *,
    seed: int = 3,
    start: str = "2026-01-05",
    tz: str | None = "America/New_York",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_bars, freq="B", tz=tz)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    close = pd.DataFrame(
        np.round(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(n_bars, n_symbols)), axis=0)), 2),
        index=index,
        columns=symbols,
    )
    volume = pd.DataFrame(np.round(rng.lognormal(13.0, 0.5, size=(n_bars, n_symbols))), index=index, columns=symbols)
    return close, volume


def _bars(close: pd.DataFrame) -> list[str]:
    return [str(stamp.date()) for stamp in close.index]


def _assert_same(reconstructed: dict[str, pd.DataFrame], **expected: pd.DataFrame) -> None:
    assert set(reconstructed) == set(expected)
    for field, frame in expected.items():
        pd.testing.assert_frame_equal(reconstructed[field], frame, check_freq=False, check_names=False)


# ---------------------------------------------------------------------------
# round trip and the canonical digest
# ---------------------------------------------------------------------------


def test_round_trip_of_close_and_volume(tmp_path: Path) -> None:
    close, volume = _panel()
    bar = _bars(close)[-1]
    record = record_panel(tmp_path, bar, {"close": close, "volume": volume})
    assert record.kind == "base" and record.reason == "first" and record.base_bar == bar
    assert record.fields == ("close", "volume") and record.tz == "America/New_York"
    assert (record.n_bars, record.n_symbols) == close.shape
    assert (record.first_bar, record.last_bar) == (_bars(close)[0], bar)
    assert (tmp_path / "panels" / record.file).stat().st_size == record.bytes
    _assert_same(reconstruct_panel(tmp_path, bar), close=close, volume=volume)
    assert latest_record(tmp_path) == record


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_round_trip_of_a_close_only_panel_in_other_timezones(tmp_path: Path, tz: str | None) -> None:
    close, _ = _panel(tz=tz)
    bar = _bars(close)[-1]
    record = record_panel(tmp_path, bar, {"close": close})
    assert record.fields == ("close",) and record.tz == tz
    _assert_same(reconstruct_panel(tmp_path, bar), close=close)


def test_digest_is_canonical_over_order_and_sensitive_to_values() -> None:
    close, volume = _panel()
    reference = canonicalize({"close": close, "volume": volume}).sha256()
    shuffled_columns = close.iloc[:, ::-1]
    shuffled_rows = volume.iloc[::-1]
    assert canonicalize({"volume": shuffled_rows, "close": shuffled_columns}).sha256() == reference
    assert canonicalize({"close": close}).sha256() != reference  # a field is part of the content
    bumped = close.copy()
    bumped.iloc[3, 2] += 0.01
    assert canonicalize({"close": bumped, "volume": volume}).sha256() != reference
    # NaN and signed zero normalize: two frames that print the same hash the same.
    a = close.copy()
    b = close.copy()
    a.iloc[0, 0] = np.nan
    b.iloc[0, 0] = np.float64("nan")
    a.iloc[1, 1] = 0.0
    b.iloc[1, 1] = -0.0
    assert canonicalize({"close": a}).sha256() == canonicalize({"close": b}).sha256()


def test_canonical_form_refuses_malformed_panels() -> None:
    close, volume = _panel()
    with pytest.raises(TypeError):
        canonicalize({})
    with pytest.raises(TypeError):
        canonicalize({"close": close, "volume": None})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="empty"):
        canonicalize({"close": close.iloc[0:0]})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        canonicalize({"close": close.reset_index(drop=True)})
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize({"close": pd.concat([close, close.iloc[-1:]])})
    with pytest.raises(ValueError, match="timezones"):
        canonicalize({"close": close, "volume": volume.tz_convert("UTC")})


def test_fields_on_different_grids_are_placed_on_the_union() -> None:
    close, volume = _panel()
    canonical = canonicalize({"close": close.iloc[:, :4], "volume": volume.iloc[2:]})
    assert canonical.symbols == tuple(close.columns)
    assert canonical.n_bars == len(close)
    frames = canonical.to_frames()
    assert frames["close"]["S05"].isna().all()
    assert frames["volume"].iloc[:2].isna().all().all()


# ---------------------------------------------------------------------------
# deltas: appended bars, dropped bars, revisions, symbols joining and leaving
# ---------------------------------------------------------------------------


def test_deltas_reconstruct_after_several_sessions_including_a_revision(tmp_path: Path) -> None:
    close, volume = _panel(n_bars=40)
    window = 30
    sessions: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for step in range(5):
        c = close.iloc[step : step + window].copy()
        v = volume.iloc[step : step + window].copy()
        if step == 2:
            c.iloc[7, 1] = round(c.iloc[7, 1] * 1.5, 2)  # the vendor revised a historical bar
        if step >= 3:
            c = c.drop(columns=["S05"])  # a symbol left the universe
            v = v.drop(columns=["S05"])
        if step == 4:
            c["S99"] = 42.0  # a symbol joined
            v["S99"] = 1.0e6
            c = c[sorted(c.columns)]
            v = v[sorted(v.columns)]
        bar = _bars(c)[-1]
        sessions[bar] = (c, v)
        record_panel(tmp_path, bar, {"close": c, "volume": v})
    records = read_index(tmp_path)
    assert [r.kind for r in records] == ["base", "delta", "delta", "delta", "delta"]
    assert all(r.base_bar == records[0].decision_bar for r in records)
    # The trailing window moved by one bar: one new bar, one gone, for every symbol.
    assert (records[1].n_upserts, records[1].n_deletes) == (6, 6)
    assert records[2].n_upserts == 7  # plus the revised cell
    assert records[3].n_deletes == 5 + window  # one bar for five names, and the whole departed symbol
    assert records[4].n_upserts == 5 + window  # one bar for five names, and the whole new symbol
    for bar, (c, v) in sessions.items():
        _assert_same(reconstruct_panel(tmp_path, bar), close=c, volume=v)


def test_a_one_day_delta_is_a_small_fraction_of_a_base(tmp_path: Path) -> None:
    """At a size where parquet's fixed overhead no longer dominates. (Measured at
    752 x 500: base about 2.7 MB, one-day delta about 8.5 KB.)"""
    close, volume = _panel(n_bars=201, n_symbols=60)
    bars = _bars(close)
    base = record_panel(tmp_path, bars[-2], {"close": close.iloc[:-1], "volume": volume.iloc[:-1]})
    delta = record_panel(tmp_path, bars[-1], {"close": close.iloc[1:], "volume": volume.iloc[1:]})
    assert delta.kind == "delta" and (delta.n_upserts, delta.n_deletes) == (60, 60)
    assert delta.bytes * 10 < base.bytes


def test_a_revised_bar_alone_is_a_tiny_delta(tmp_path: Path) -> None:
    close, volume = _panel()
    bars = _bars(close)
    record_panel(tmp_path, bars[-2], {"close": close.iloc[:-1], "volume": volume.iloc[:-1]})
    revised = close.iloc[:-1].copy()
    revised.iloc[4, 0] += 1.0
    # The next session saw the same window plus one bar, with one historical cell revised.
    full = pd.concat([revised, close.iloc[-1:]])
    record = record_panel(tmp_path, bars[-1], {"close": full, "volume": volume})
    assert record.kind == "delta" and record.n_upserts == close.shape[1] + 1 and record.n_deletes == 0
    _assert_same(reconstruct_panel(tmp_path, bars[-1]), close=full, volume=volume)


# ---------------------------------------------------------------------------
# refusals: digest mismatch, conflicts, order
# ---------------------------------------------------------------------------


def test_tampered_file_is_refused(tmp_path: Path) -> None:
    close, volume = _panel()
    bars = _bars(close)
    record_panel(tmp_path, bars[-2], {"close": close.iloc[:-1], "volume": volume.iloc[:-1]})
    delta = record_panel(tmp_path, bars[-1], {"close": close, "volume": volume})
    path = tmp_path / "panels" / delta.file
    path.write_bytes(path.read_bytes()[:-1] + b"\0")
    with pytest.raises(PanelDigestMismatch, match="hashes to"):
        reconstruct_panel(tmp_path, bars[-1])
    # Recording the next session cannot chain on an unreadable previous panel: it
    # writes a fresh base, loudly, so the decision is still recorded in full.
    nxt = record_panel(tmp_path, "2026-12-31", {"close": close * 1.01, "volume": volume})
    assert nxt.kind == "base" and nxt.reason == "previous_unreadable"


def test_index_digest_that_disagrees_with_the_content_is_refused(tmp_path: Path) -> None:
    close, volume = _panel()
    bar = _bars(close)[-1]
    record = record_panel(tmp_path, bar, {"close": close, "volume": volume})
    index = tmp_path / "panels" / "index.jsonl"
    row = json.loads(index.read_text().splitlines()[0])
    row["sha256"] = "0" * 64
    index.write_text(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(PanelDigestMismatch, match="reconstructs to"):
        reconstruct_panel(tmp_path, bar)
    assert read_index(tmp_path)[0].file == record.file


def test_same_content_is_idempotent_and_different_content_is_refused(tmp_path: Path) -> None:
    close, volume = _panel()
    bar = _bars(close)[-1]
    first = record_panel(tmp_path, bar, {"close": close, "volume": volume})
    again = record_panel(tmp_path, bar, {"volume": volume, "close": close.iloc[::-1]})  # same content, other order
    assert again == first
    assert len(read_index(tmp_path)) == 1
    assert sorted(p.name for p in (tmp_path / "panels").iterdir()) == sorted(["index.jsonl", first.file])
    changed = close.copy()
    changed.iloc[-1, 0] += 0.5
    with pytest.raises(PanelLogConflict, match="already recorded"):
        record_panel(tmp_path, bar, {"close": changed, "volume": volume})
    assert len(read_index(tmp_path)) == 1


def test_out_of_order_bars_and_unknown_bars_are_refused(tmp_path: Path) -> None:
    close, volume = _panel()
    bars = _bars(close)
    record_panel(tmp_path, bars[-1], {"close": close, "volume": volume})
    with pytest.raises(PanelLogError, match="earlier than"):
        record_panel(tmp_path, bars[-2], {"close": close.iloc[:-1], "volume": volume.iloc[:-1]})
    with pytest.raises(PanelLogError, match="no panel recorded"):
        reconstruct_panel(tmp_path, bars[0])
    with pytest.raises(ValueError, match="ISO date"):
        record_panel(tmp_path, "today", {"close": close})
    assert latest_record(tmp_path / "nowhere") is None


# ---------------------------------------------------------------------------
# cadence: when a base is written
# ---------------------------------------------------------------------------


def test_base_rolls_over_on_the_session_cadence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(panel_log, "BASE_EVERY_SESSIONS", 3)
    close, volume = _panel(n_bars=36)
    window = 30
    expected: dict[str, pd.DataFrame] = {}
    for step in range(6):
        c = close.iloc[step : step + window]
        expected[_bars(c)[-1]] = c
        record_panel(tmp_path, _bars(c)[-1], {"close": c, "volume": volume.iloc[step : step + window]})
    records = read_index(tmp_path)
    assert [r.kind for r in records] == ["base", "delta", "delta", "base", "delta", "delta"]
    assert [r.reason for r in records][3] == "cadence"
    assert records[4].base_bar == records[3].decision_bar
    for bar, c in expected.items():
        _assert_same({"close": reconstruct_panel(tmp_path, bar)["close"]}, close=c)


def test_a_delta_exceeding_half_the_panel_becomes_a_base(tmp_path: Path) -> None:
    close, volume = _panel()
    bars = _bars(close)
    record_panel(tmp_path, bars[-2], {"close": close.iloc[:-1], "volume": volume.iloc[:-1]})
    rewritten = close * 1.1  # a wholesale split-basis rewrite touches every cell
    record = record_panel(tmp_path, bars[-1], {"close": rewritten, "volume": volume})
    assert record.kind == "base" and record.reason == "delta_exceeds_half" and record.base_bar == bars[-1]
    _assert_same(reconstruct_panel(tmp_path, bars[-1]), close=rewritten, volume=volume)


def test_changing_the_field_set_forces_a_base(tmp_path: Path) -> None:
    close, volume = _panel()
    bars = _bars(close)
    record_panel(tmp_path, bars[-2], {"close": close.iloc[:-1], "volume": volume.iloc[:-1]})
    record = record_panel(tmp_path, bars[-1], {"close": close})
    assert record.kind == "base" and record.reason == "fields_changed"
    _assert_same(reconstruct_panel(tmp_path, bars[-1]), close=close)
    _assert_same(reconstruct_panel(tmp_path, bars[-2]), close=close.iloc[:-1], volume=volume.iloc[:-1])


def test_missing_file_is_named(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    close, volume = _panel()
    bar = _bars(close)[-1]
    record = record_panel(tmp_path, bar, {"close": close, "volume": volume})
    (tmp_path / "panels" / record.file).unlink()
    with pytest.raises(PanelLogError, match="missing"):
        reconstruct_panel(tmp_path, bar)
    with pytest.raises(PanelLogError, match="missing"):
        record_panel(tmp_path, bar, {"close": close, "volume": volume})  # same content, but nothing on disk
    with caplog.at_level(logging.ERROR, logger="prism.live.panel_log"):
        nxt = record_panel(tmp_path, "2026-12-31", {"close": close, "volume": volume})
    assert nxt.kind == "base" and nxt.reason == "previous_unreadable"
    assert any("cannot reconstruct the previous panel" in r.getMessage() for r in caplog.records)
