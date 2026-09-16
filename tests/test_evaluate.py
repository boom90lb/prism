"""The stage 1 smoke evaluator (engine specification §3.8, stage 1 exit).

The smoke path exists to make the stage's exit demonstrable: a specification
loads, its components score causally, the sub-targets compose into a book, and
the book accounts through the retained reproduction engine
(``backtest_target_weights``). Two properties are worth more than the summary
numbers it prints and are tested as such:

* **Next-open fills.** A target decided at close *t* must not be filled before
  open *t + 1*, in backtest exactly as live (SPEC N3). A test that only checked
  the metrics would pass on a simulator that filled at close *t*.
* **It writes nothing.** No ``ResultRecord``, no trial-log row. The recording
  protocol is stage 2's, and a smoke run that quietly minted an evidence row
  would put an engineering run into the selection set that promotion decisions
  are read from.

The third is the reproduction engine's own caveat, asserted rather than assumed:
weights are held constant between decision bars, which is what the
``historical_constant_weight`` label means.

From stage 3b the smoke path can also run the ``canonical`` engine, and the
fourth property is that the two are told apart: the canonical run needs a
declared cost stack, says ``fixed_share`` in its summary, and does not produce
the constant-weight number on the same decisions.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from prism.cli import main
from prism.evaluate import (
    RECORDING_STATEMENT,
    CanonicalSettings,
    EvaluationError,
    SmokeResult,
    _dividends,
    smoke_evaluate,
)
from prism.execution.cost_model import CostStack
from prism.io.dataset import Dataset
from prism.io.synthetic import EQUITY_INDEX, write_synthetic_dataset
from prism.spec import load_spec
from prism.workspace import Workspace

START = dt.date(2026, 1, 1)
END = dt.date(2027, 12, 31)
VINTAGE = "synthetic_v1"
PACKAGE = "evaluate_strategies"

EXAMPLE_SIGNAL = """
import pandas as pd
from prism.signal.base import Signal


class TrailingReturnRank(Signal):
    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None
    refit_every_bars = 1
    training_window_bars = 0

    def __init__(self, lookback_bars: int = 60) -> None:
        self.lookback_bars = lookback_bars

    @property
    def horizon_bars(self) -> int:
        return 21

    @property
    def required_history(self) -> int:
        return self.lookback_bars + 1

    def fit(self, close, volume=None):
        return self

    def score(self, close, volume=None):
        if len(close) < self.required_history:
            raise ValueError("panel too short")
        return close.iloc[-1] / close.iloc[-1 - self.lookback_bars] - 1.0


class TrainOnlyRank(TrailingReturnRank):
    fitted_state = "train_only"
"""

BOOK_TOML = """
[strategy]
name = "trailing_rank"
universe_index = "synthetic_100"
combination = "position_space"

[[strategy.components]]
id = "trailing"
path = "{package}.my_signal:{cls}"
weight = 1.0
horizon_bars = 21
update_every_bars = 21
valid_for_bars = 26
to_weights = "decile_long_short"

[strategy.components.params]
lookback_bars = 60

[strategy.components.to_weights_params]
decile = 0.20
max_gross = 1.0
max_symbol_abs_weight = 0.35

[strategy.eligibility]
min_price = 5.0
min_median_dollar_volume = 1000000.0
listing_seasoning_bars = 20

[construction]
kind = "heuristic"

[policy]
profile = "profiles/paper.toml"
costs = "costs/retail_v1.toml"
execution = "profiles/execution_synthetic.toml"
monitoring = "policies/monitoring.toml"
recovery = "policies/recovery.toml"
"""

POLICY_FILES = {
    "profiles/paper.toml": 'name = "paper"\n',
    "costs/retail_v1.toml": "commission_bps = 1.0\n",
    "profiles/execution_synthetic.toml": 'venue = "synthetic"\n',
    "policies/monitoring.toml": "review_bars = 21\n",
    "policies/recovery.toml": 'mode = "reduce_only"\n',
}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """A private project whose ``data/`` holds the synthetic fixture."""
    root = tmp_path_factory.mktemp("evaluate") / "project"
    ws = Workspace(root).ensure()
    write_synthetic_dataset(ws.data, seed=11, scenarios="all", n_equities=14, n_etfs=3, start=START, end=END)
    (root / PACKAGE).mkdir()
    (root / PACKAGE / "__init__.py").write_text("", encoding="utf-8")
    (root / PACKAGE / "my_signal.py").write_text(textwrap.dedent(EXAMPLE_SIGNAL), encoding="utf-8")
    for relative, content in POLICY_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "specs").mkdir()
    (root / "specs" / "book.toml").write_text(BOOK_TOML.format(package=PACKAGE, cls="TrailingReturnRank"), "utf-8")
    (root / "specs" / "train_only.toml").write_text(BOOK_TOML.format(package=PACKAGE, cls="TrainOnlyRank"), "utf-8")

    before_path = list(sys.path)
    before_modules = set(sys.modules)
    yield ws
    sys.path[:] = before_path
    for name in set(sys.modules) - before_modules:
        del sys.modules[name]


@pytest.fixture(scope="module")
def result(workspace: Workspace) -> SmokeResult:
    spec = load_spec(workspace.root / "specs" / "book.toml", workspace=workspace)
    return smoke_evaluate(spec, workspace=workspace)


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if getattr(handler, "_te_managed", False):
                root.removeHandler(handler)
                handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)


# ---- the run ---------------------------------------------------------------


def test_the_smoke_run_completes_over_the_fixture(result: SmokeResult) -> None:
    assert result.engine == "historical_constant_weight"
    assert result.vintage == VINTAGE
    assert len(result.decision_dates) > 1
    assert not result.backtest.returns.empty
    assert result.backtest.returns.notna().all()
    # The book actually traded: an all-zero target frame would produce a flat
    # equity curve and a vacuous pass.
    assert float(result.backtest.fill_weights.abs().sum().sum()) > 0.0
    assert float(result.backtest.costs["turnover"].sum()) > 0.0
    for key in ("total_return", "sharpe", "avg_gross", "avg_turnover"):
        assert pd.notna(result.metrics[key])


def test_the_universe_is_every_sid_the_index_held_over_the_span(result: SmokeResult, workspace: Workspace) -> None:
    """A name removed mid-sample still has to be exited, so it needs a column."""
    dataset = Dataset.open(workspace.data)
    at_end = dataset.view("research_vintage", END.isoformat(), vintage=VINTAGE).membership(EQUITY_INDEX)

    assert set(result.universe) >= at_end
    assert len(result.universe) > len(at_end), "the membership scenario adds and removes a name inside the sample"


def test_a_target_decided_at_close_fills_at_the_next_open(result: SmokeResult) -> None:
    targets = result.backtest.target_weights
    fills = result.backtest.fill_weights
    index = list(fills.index)

    decided = [stamp for stamp in targets.index if targets.loc[stamp].notna().any() and stamp in index]
    assert decided, "no decision row reached the accounting engine"

    moved = 0
    for stamp in decided:
        position = index.index(stamp)
        if position + 1 >= len(index):
            continue
        target = targets.loc[stamp].fillna(0.0)
        if float((target - fills.loc[stamp]).abs().sum()) == 0.0:
            continue  # nothing to trade at this decision
        moved += 1
        # Not filled on the decision bar itself...
        assert float((fills.loc[stamp] - fills.iloc[position - 1]).abs().sum()) == 0.0 or position == 0
        # ... and filled in full at the next open.
        pd.testing.assert_series_equal(fills.iloc[position + 1], target, check_names=False)
    assert moved > 0, "every decision was a no-op, so the fill timing was never exercised"


def test_weights_are_held_constant_between_decisions(result: SmokeResult) -> None:
    """The historical_constant_weight caveat, asserted rather than assumed (§3.8)."""
    fills = result.backtest.fill_weights
    decisions = {pd.Timestamp(day).date() for day in result.decision_dates}
    index = list(fills.index)

    for position in range(1, len(index)):
        changed = float((fills.iloc[position] - fills.iloc[position - 1]).abs().sum()) > 0.0
        if changed:
            assert (
                index[position - 1].date() in decisions
            ), f"the book changed at {index[position]} without a decision at {index[position - 1]}"


def test_cash_distributions_reach_the_accounting_engine(workspace: Workspace, result: SmokeResult) -> None:
    """A distribution the fixture writes must arrive as an amount, never a silent zero."""
    dataset = Dataset.open(workspace.data)
    view = dataset.view("research_vintage", END.isoformat(), vintage=VINTAGE)
    opens = view.panel(list(result.universe), END.isoformat(), "open")
    events = view.events(list(result.universe), START.isoformat(), END.isoformat())
    cash = events.loc[events["event_type"] == "cash_distribution"]
    assert not cash.empty, "the fixture writes a cash distribution on an index member"

    frame = _dividends(view, result.universe, opens, START.isoformat(), END.isoformat())

    for row in cash.itertuples():
        amount = float(json.loads(row.payload)["amount_per_share"])
        stamp = pd.Timestamp(str(row.effective_date), tz=opens.index.tz)
        assert amount > 0.0
        assert float(frame.loc[stamp, row.sid]) == amount
    # Everything else is a real zero, not a broadcast of the distribution.
    assert float(frame.to_numpy().sum()) == pytest.approx(
        float(sum(json.loads(row.payload)["amount_per_share"] for row in cash.itertuples()))
    )
    assert "dividend_return" in result.backtest.costs.columns


def test_a_distribution_with_no_amount_refuses_rather_than_reading_as_zero(
    workspace: Workspace, result: SmokeResult
) -> None:
    """The silent zero N7 forbids, asserted on the evaluator's own seam.

    A ``cash_distribution`` row whose payload carries no ``amount_per_share`` is
    a defect in the events table. Reading it as ``0.0`` would drop cash income
    from the ledger with nothing anywhere saying so, which is exactly the
    failure the fail-loud rule exists to prevent.
    """
    dataset = Dataset.open(workspace.data)
    view = dataset.view("research_vintage", END.isoformat(), vintage=VINTAGE)
    opens = view.panel(list(result.universe), END.isoformat(), "open")
    events = view.events(list(result.universe), START.isoformat(), END.isoformat())
    cash = events.loc[events["event_type"] == "cash_distribution"]
    assert not cash.empty

    stripped = events.copy()
    target = cash.index[0]
    stripped.loc[target, "payload"] = json.dumps({"currency": "USD"})

    class _AmountlessView:
        def __init__(self, inner: object, frame: pd.DataFrame) -> None:
            self._inner, self._frame = inner, frame

        def events(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
            return self._frame

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    with pytest.raises(Exception) as excinfo:
        _dividends(_AmountlessView(view, stripped), result.universe, opens, START.isoformat(), END.isoformat())
    assert "amount_per_share" in str(excinfo.value)


def test_an_ex_date_off_the_session_calendar_is_credited_not_dropped(
    workspace: Workspace, result: SmokeResult
) -> None:
    """§3.2's forward roll, asserted through the evaluator rather than only in ``io.events``.

    The evaluator once carried its own placement, which dropped an ex-date that
    was not on the panel's index. An issuer that goes ex on a Saturday would
    have had its cash silently vanish; the rule is that it credits the next
    session.
    """
    dataset = Dataset.open(workspace.data)
    view = dataset.view("research_vintage", END.isoformat(), vintage=VINTAGE)
    opens = view.panel(list(result.universe), END.isoformat(), "open")
    events = view.events(list(result.universe), START.isoformat(), END.isoformat())
    cash = events.loc[events["event_type"] == "cash_distribution"]
    assert not cash.empty

    row = cash.iloc[0]
    naive = opens.index.tz_localize(None)
    start_at = int(naive.searchsorted(pd.Timestamp(str(row["effective_date"]))))
    # The first weekend (or holiday) gap at or after the recorded ex-date: a
    # calendar day that is strictly between two sessions and carries no bar.
    gap = next(
        (i for i in range(start_at, len(naive) - 1) if (naive[i + 1] - naive[i]) > pd.Timedelta(days=1)),
        None,
    )
    assert gap is not None, "the fixture must contain a non-consecutive session pair after the ex-date"
    following = opens.index[gap + 1]
    off_session = (naive[gap] + pd.Timedelta(days=1)).date()
    assert pd.Timestamp(off_session).tz_localize(opens.index.tz) not in opens.index

    moved = events.copy()
    moved.loc[cash.index[0], "effective_date"] = off_session.isoformat()

    class _MovedView:
        def __init__(self, inner: object, frame: pd.DataFrame) -> None:
            self._inner, self._frame = inner, frame

        def events(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
            return self._frame

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    frame = _dividends(_MovedView(view, moved), result.universe, opens, START.isoformat(), END.isoformat())
    amount = float(json.loads(row["payload"])["amount_per_share"])
    assert float(frame.loc[following, row["sid"]]) == pytest.approx(amount), "the off-session ex-date lost its cash"
    assert float(frame.to_numpy().sum()) == pytest.approx(
        float(sum(json.loads(r.payload)["amount_per_share"] for r in cash.itertuples()))
    ), "the roll must move the amount, never duplicate or drop it"


# ---- what it must not write ------------------------------------------------


def test_the_smoke_run_writes_no_record_and_no_trial_row(workspace: Workspace, result: SmokeResult) -> None:
    for directory in (workspace.results, workspace.trials, workspace.runs, workspace.state):
        assert list(directory.iterdir()) == [], f"{directory} is not empty after a smoke run"
    assert not list(workspace.root.rglob("*.jsonl"))
    assert RECORDING_STATEMENT in result.text()
    assert "no research question" in RECORDING_STATEMENT


def test_the_summary_names_the_scenarios_this_stage_does_not_apply(workspace: Workspace, result: SmokeResult) -> None:
    manifest = Dataset.open(workspace.data).manifest
    written_not_applied = manifest.assumptions["scenarios_written_not_applied"]

    assert list(result.unexercised_scenarios) == list(written_not_applied)
    assert "spinoff" in result.unexercised_scenarios
    text = result.text()
    for scenario in result.unexercised_scenarios:
        assert scenario in text
    assert result.unexercised_note in text


def test_the_summary_also_names_the_scenarios_it_selects_but_reads_wrong(
    workspace: Workspace, result: SmokeResult
) -> None:
    """Section 4's "written but not exercised" table is eight rows, not six.

    ``split`` and ``halt`` sit inside the equity index, so the evaluation
    selects and prices those sids while applying neither the share change nor
    the halt. A summary that named only the six out-of-index scenarios would
    understate what this stage leaves untested in exactly the case where a
    reader is most likely to be misled: the numbers include those names.
    """
    manifest = Dataset.open(workspace.data).manifest
    in_universe = manifest.assumptions["scenarios_unapplied_in_universe"]

    assert list(result.in_universe_unapplied_scenarios) == list(in_universe)
    assert set(in_universe) == {"split", "halt"}
    assert set(in_universe).isdisjoint(result.unexercised_scenarios)

    text = result.text()
    for scenario in in_universe:
        assert scenario in text
    assert result.in_universe_unapplied_note in text

    # The union is section 4's whole table.
    named = set(result.unexercised_scenarios) | set(result.in_universe_unapplied_scenarios)
    assert named == {"split", "halt", "spinoff", "ticker_reuse", "delisting", "revision", "rebase", "recovery"}


# ---- refusals --------------------------------------------------------------


def _zero_cost_stack() -> CostStack:
    """A stack that charges nothing, every component declared (§3.6 refuses an omission)."""
    volume = {
        "estimator": "median",
        "window": "trailing",
        "window_bars": 20,
        "refresh": "decision",
        "floor_dollars": 100_000.0,
    }
    return CostStack.from_mapping(
        {
            "spread": {"mode": "flat", "flat_bps": 0.0},
            "impact": {
                "quadratic_notional_coeff": 0.0,
                "participation_model": "linear",
                "participation_coeff": 0.0,
            },
            "dollar_volume": {"spread_bucket": volume, "participation": volume, "impact": volume},
            "commission_bps": 0.0,
            "borrow_bps_annual": 0.0,
            "cash_income_bps_annual": 0.0,
            "financing_bps_annual": 0.0,
            "short_proceeds": "collateral_no_rebate",
            "holding_cost_settlement": "per_bar",
            "trading_days_per_year": 252,
        }
    )


def test_the_canonical_engine_needs_a_declared_cost_stack(workspace: Workspace) -> None:
    """Stage 3b made the canonical engine available; it did not make it free.

    Before stage 3b this refusal named a missing module. The modules exist now,
    so the only thing that can refuse a canonical run is an undeclared cost
    policy — and it must, because a zero nobody declared is indistinguishable in
    the record from a zero that was (§3.6).
    """
    spec = load_spec(workspace.root / "specs" / "book.toml", workspace=workspace)

    with pytest.raises(EvaluationError, match="declared .*CostStack"):
        smoke_evaluate(spec, workspace=workspace, engine="canonical")
    with pytest.raises(EvaluationError, match="would ignore them"):
        smoke_evaluate(spec, workspace=workspace, canonical=CanonicalSettings(cost_stack=_zero_cost_stack()))


def test_the_smoke_path_runs_the_canonical_engine_on_shares(workspace: Workspace) -> None:
    """The fixed-share book is a different book, and the summary says which one it is."""
    spec = load_spec(workspace.root / "specs" / "book.toml", workspace=workspace)

    canonical = smoke_evaluate(
        spec,
        workspace=workspace,
        engine="canonical",
        capital=1.0,
        # A continuous grid: at a capital of 1.0 a whole share is unaffordable,
        # and §3.7 already declares that grid for the weight-space comparison.
        canonical=CanonicalSettings(cost_stack=_zero_cost_stack(), share_increment=0.0),
    )
    reproduction = smoke_evaluate(spec, workspace=workspace, capital=1.0)

    assert canonical.engine == "canonical"
    assert "fixed_share" in canonical.text()
    assert "shares are held between decisions" in canonical.text()
    assert "constant_weight" in reproduction.text()
    assert canonical.decision_dates == reproduction.decision_dates, "one decision sequence, two accountings"
    assert canonical.metrics["total_return"] != reproduction.metrics["total_return"]


def test_the_smoke_path_refuses_a_train_only_component_because_it_never_fits(workspace: Workspace) -> None:
    """Stage 2 moved this boundary, and the message moved with it.

    At stage 1 the refusal said fitted state could not be persisted before
    stage 3. Stage 2's evaluator fits per fold and writes the artifact, so that
    reason is no longer true. What is still true is that the *smoke* path does
    not fit at all: it scores unfitted components, so a train-only component
    would be scored before it was fitted. The refusal now says that and names
    the evaluation that does fit it.
    """
    spec = load_spec(workspace.root / "specs" / "train_only.toml", workspace=workspace)

    with pytest.raises(EvaluationError) as excinfo:
        smoke_evaluate(spec, workspace=workspace)
    message = str(excinfo.value)
    assert "does not fit" in message
    assert "walk_forward_evaluate" in message


def test_the_prism_evaluate_smoke_command_prints_the_summary(
    workspace: Workspace, restore_root_logger, capsys
) -> None:
    spec = str(workspace.root / "specs" / "book.toml")

    code = main(["--workspace", str(workspace.root), "evaluate", spec, "--smoke"])

    assert code == 0
    printed = capsys.readouterr().out
    assert "prism evaluate --smoke" in printed
    assert "historical_constant_weight" in printed
    assert RECORDING_STATEMENT in printed
