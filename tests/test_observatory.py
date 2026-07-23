"""Append-only observatory capture store (W5)."""

from __future__ import annotations

import pytest

from prism.io.observatory import append_capture, count_captures, iter_captures


def test_append_and_read_jsonl(tmp_path):
    path = tmp_path / "news.jsonl"
    append_capture(path, {"captured_at": "2026-07-22T12:00:00Z", "lane": "news", "n": 1})
    append_capture(path, {"captured_at": "2026-07-22T13:00:00Z", "lane": "news", "n": 2})
    rows = list(iter_captures(path))
    assert len(rows) == 2
    assert rows[0]["n"] == 1 and rows[1]["n"] == 2
    assert count_captures(path) == 2


def test_append_gzip(tmp_path):
    path = tmp_path / "edgar.jsonl.gz"
    append_capture(path, {"captured_at": "2026-07-22T12:00:00Z", "lane": "edgar", "payload": {"k": "v"}})
    rows = list(iter_captures(path))
    assert rows[0]["lane"] == "edgar"
    assert rows[0]["payload"] == {"k": "v"}


def test_missing_required_field_raises(tmp_path):
    with pytest.raises(ValueError, match="captured_at"):
        append_capture(tmp_path / "x.jsonl", {"lane": "news"})
    with pytest.raises(ValueError, match="lane"):
        append_capture(tmp_path / "x.jsonl", {"captured_at": "t"})


def test_missing_file_iter_is_empty(tmp_path):
    assert list(iter_captures(tmp_path / "nope.jsonl")) == []
    assert count_captures(tmp_path / "nope.jsonl") == 0
