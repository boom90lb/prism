"""Offline tests for Alpaca split history (prism.live.split_history)."""

import pytest

from prism.live import split_history as sh


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {}
        self.text = ""

    def json(self):
        return self._payload


class _Session:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def request(self, method, url, headers=None, timeout=None, params=None):
        self.calls.append(params)
        return _Response(self.pages.pop(0))


def test_fetch_splits_reads_both_kinds_across_pages():
    pages = [
        {"corporate_actions": {"forward_splits": [
            {"symbol": "AAPL", "ex_date": "2020-08-31", "old_rate": 1, "new_rate": 4}]}, "next_page_token": "p2"},
        {"corporate_actions": {"reverse_splits": [
            {"symbol": "GE", "ex_date": "2021-08-02", "old_rate": 8, "new_rate": 1}]}},
    ]
    session = _Session(pages)
    frame = sh.fetch_splits(["AAPL", "GE"], "2020-01-01", "2026-06-30", key_id="k", secret_key="s", session=session)
    assert list(frame["symbol"]) == ["AAPL", "GE"]
    assert frame.set_index("symbol").loc["AAPL", "ratio"] == 4.0
    assert frame.set_index("symbol").loc["GE", "ratio"] == pytest.approx(0.125)
    assert session.calls[0]["types"] == "forward_split,reverse_split"
    assert session.calls[1]["page_token"] == "p2"


def test_fetch_splits_fails_loud_on_shape_drift_and_bad_records():
    with pytest.raises(ValueError):
        sh.fetch_splits(["X"], "2020-01-01", "2020-12-31", key_id="k", secret_key="s",
                        session=_Session([{"unexpected": {}}]))
    bad = {"corporate_actions": {"forward_splits": [{"symbol": "X", "ex_date": "2020-05-01", "old_rate": 0,
                                                     "new_rate": 2}]}}
    with pytest.raises(ValueError):
        sh.fetch_splits(["X"], "2020-01-01", "2020-12-31", key_id="k", secret_key="s", session=_Session([bad]))
    empty = sh.fetch_splits(["X"], "2020-01-01", "2020-12-31", key_id="k", secret_key="s",
                            session=_Session([{"corporate_actions": {}}]))
    assert empty.empty
