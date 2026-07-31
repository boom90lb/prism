"""live/ JSONL ledger mechanics (SPEC §7.7; the I-9 record discipline).

Pins the three properties prism.live.ledgers guarantees: durable appends,
a torn tail that is skipped loudly and healed — never a wedged loop — and
monotone idempotency for the per-session ledgers.
"""

from __future__ import annotations

import json
import logging

from prism.live.ledgers import (
    append_monotone,
    append_rows,
    last_value,
    read_frame,
    read_rows,
)


def test_append_and_read_round_trip(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_rows(path, [{"a": 1}, {"a": 2}])
    append_rows(path, [{"a": 3}])
    assert read_rows(path) == [{"a": 1}, {"a": 2}, {"a": 3}]
    frame = read_frame(path)
    assert list(frame["a"]) == [1, 2, 3]


def test_missing_ledger_reads_empty(tmp_path) -> None:
    path = tmp_path / "absent.jsonl"
    assert read_rows(path) == []
    assert read_frame(path).empty
    assert last_value(path, "a") is None


def test_torn_tail_is_skipped_loudly_never_fatal(tmp_path, caplog) -> None:
    # A crash mid-append leaves a partial final line. The reader must skip it
    # with an ERROR — the pre-2026-07-29 readers raised, converting one crash
    # into a permanently wedged cycle (halt_reason reads the equity ledger
    # before every decision).
    path = tmp_path / "ledger.jsonl"
    append_rows(path, [{"decision_bar": "2026-07-01", "equity": 100.0}])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"decision_bar": "2026-07-0')  # torn: no close, no newline
    with caplog.at_level(logging.ERROR):
        rows = read_rows(path)
    assert rows == [{"decision_bar": "2026-07-01", "equity": 100.0}]
    assert any("unparseable" in record.message for record in caplog.records)


def test_append_after_torn_tail_heals_the_fragment(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_rows(path, [{"a": 1}])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"torn": ')  # no trailing newline
    append_rows(path, [{"a": 2}])
    # The fragment was sealed as its own (skipped) line; the new row is intact.
    assert read_rows(path) == [{"a": 1}, {"a": 2}]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1] == '{"torn": '  # quarantined, not fused with the new row
    assert json.loads(lines[2]) == {"a": 2}


def test_non_object_lines_are_skipped(tmp_path, caplog) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('42\n{"a": 1}\n', encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        assert read_rows(path) == [{"a": 1}]


def test_append_monotone_is_idempotent_and_monotone(tmp_path) -> None:
    path = tmp_path / "equity.jsonl"
    assert append_monotone(path, "decision_bar", {"decision_bar": "2026-05-01", "equity": 1.0})
    # Same bar: skipped (the write-ahead restart), earlier bar: skipped.
    assert not append_monotone(path, "decision_bar", {"decision_bar": "2026-05-01", "equity": 2.0})
    assert not append_monotone(path, "decision_bar", {"decision_bar": "2026-04-30", "equity": 3.0})
    assert append_monotone(path, "decision_bar", {"decision_bar": "2026-05-02", "equity": 4.0})
    assert [row["decision_bar"] for row in read_rows(path)] == ["2026-05-01", "2026-05-02"]


def test_append_monotone_survives_a_torn_tail(tmp_path) -> None:
    # The monotone key is read from the last PARSEABLE row, so a torn tail
    # neither blocks the append nor resets the key.
    path = tmp_path / "equity.jsonl"
    append_monotone(path, "decision_bar", {"decision_bar": "2026-05-01", "equity": 1.0})
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"decision_bar": "2026-05-0')
    assert not append_monotone(path, "decision_bar", {"decision_bar": "2026-05-01", "equity": 9.0})
    assert append_monotone(path, "decision_bar", {"decision_bar": "2026-05-02", "equity": 2.0})
    assert [row["decision_bar"] for row in read_rows(path)] == ["2026-05-01", "2026-05-02"]


def test_last_value_scans_past_rows_without_the_field(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    append_rows(path, [{"decision_bar": "2026-05-01"}, {"note": "no bar here"}])
    assert last_value(path, "decision_bar") == "2026-05-01"
