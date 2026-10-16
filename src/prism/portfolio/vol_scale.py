"""Constant-volatility sizing: the Barroso and Santa-Clara (2015) rule and its failure path.

The rule scales a book so that its estimated volatility is a fixed target. At a
refresh at close *t* it estimates the book's annualized volatility from the
last ``lookback_bars`` daily returns, sets ``scale = min(max_scale, target_vol /
sigma)``, and the book carries that scale from *t + 1* until its next refresh
(N1, N2). Barroso, P. and Santa-Clara, P. (2015), "Momentum has its moments",
Journal of Financial Economics 116(1), 111-120.

The estimate is taken on the *base book*: the unscaled book the rule sizes,
held at constant weights from one close to the next and gross of costs.
:func:`base_book_returns` computes its returns from a close panel and the
weights the book held at each close; a return at bar *t* uses the weights held
at the close of *t - 1* and nothing later.

The arithmetic of :func:`decide_scale` on a valid window is
``research/scripts/factor_lab.py::scale_constant_vol`` (lines 388-399 at commit
70e8332): sample standard deviation with ``ddof = 1``, annualized by the square
root of ``annualization_bars``, then the capped ratio. It is duplicated here by
value rather than imported, because ``prism`` never imports ``research``
(SPEC §7, N8), and ``tests/test_vol_scale.py`` checks the two agree.

Where the factor lab raises on a degenerate estimate, this module de-grosses
instead (N7 allows either, explicitly): a window that cannot be trusted --
too few finite returns, a decision bar that is not the panel's last bar, more
than ``max_zero_return_fraction`` of the returns exactly zero (a stale or flat
panel), or a non-finite estimate or one at or below ``MIN_SIGMA`` -- gives
``min(1, last good scale)``, or 1 when there is none. A failure can therefore
only lower exposure: never above the unscaled book, never above the scale the
book last earned. Before the book has ``lookback_bars`` returns at all the
scale is 1, recorded as warm-up rather than as a failure.

This module is numpy and pandas only (N8). The evaluator applies it per
component at each due bar (``prism.evaluate``, ``[sizing]`` in
``prism.spec``); no live path calls it.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: The one sizing rule a specification can declare (``[sizing] rule``).
CONSTANT_VOL = "constant_vol"
SIZING_RULES: tuple[str, ...] = (CONSTANT_VOL,)

#: What a scale decision can be. ``ok`` is the rule's own value; ``warmup`` is
#: the scale of 1 before the book has a full window of returns and has never
#: had a good decision; ``fallback`` is the failure path.
STATUSES: tuple[str, ...] = ("ok", "warmup", "fallback")

#: Why a decision fell back, in the order the checks run.
FALLBACK_REASONS: tuple[str, ...] = (
    "decision_bar_not_last_bar",
    "insufficient_returns",
    "zero_returns",
    "degenerate_sigma",
)

#: An annualized estimate at or below this is a numerically constant window --
#: zero up to rounding -- and is degenerate. No book that trades has a
#: volatility anywhere near it; without the floor a window of identical
#: returns would read as almost no risk and lever the book to ``max_scale``.
MIN_SIGMA = 1e-10

#: The relative slack :func:`cap_gross` allows before the ceiling binds, so a
#: book summed from tranches that sits exactly at the ceiling is not rescaled
#: by rounding in its last digit.
GROSS_TOLERANCE = 1e-12

#: The cash buffer the planned public baseline (selection set
#: ``public_baseline_v1``) keeps beside a T-bill holding of idle equity.
#: :func:`session_exposure` reports the sleeve fraction ``max(0, 1 - buffer -
#: L/E)`` at this buffer; it is a reported fact derived from the long side and
#: enters no decision.
SLEEVE_BUFFER = 0.02


class VolScaleError(ValueError):
    """A sizing rule or its inputs are malformed (N7)."""


@dataclass(frozen=True)
class VolScaleRule:
    """The constant-volatility rule's parameters, every default resolved.

    ``max_gross`` is the account's gross ceiling after scaling: when the
    scaled account target's gross exceeds it, the whole target is scaled down
    to it (:func:`cap_gross`). It defaults to ``max_scale``, which is the
    ceiling a single book of raw gross 1.0 reaches at the cap, so for such a
    book it never binds.
    """

    target_vol: float = 0.12
    lookback_bars: int = 126
    max_scale: float = 2.0
    max_gross: float | None = None
    annualization_bars: int = 252
    max_zero_return_fraction: float = 0.05
    rule: str = field(default=CONSTANT_VOL)

    def __post_init__(self) -> None:
        if self.rule not in SIZING_RULES:
            raise VolScaleError(f"sizing rule {self.rule!r} is not one of {SIZING_RULES}")
        _positive_finite("target_vol", self.target_vol)
        _integer_at_least("lookback_bars", self.lookback_bars, 2)
        _positive_finite("max_scale", self.max_scale)
        if self.max_gross is None:
            object.__setattr__(self, "max_gross", float(self.max_scale))
        _positive_finite("max_gross", self.max_gross)
        _integer_at_least("annualization_bars", self.annualization_bars, 1)
        fraction = self.max_zero_return_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) < 1.0:
            raise VolScaleError(
                f"max_zero_return_fraction must be a number in [0, 1), got {fraction!r}; at 1 a window of "
                "nothing but zeros would pass as a volatility estimate"
            )
        for name in ("target_vol", "max_scale", "max_gross", "max_zero_return_fraction"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("lookback_bars", "annualization_bars"):
            object.__setattr__(self, name, int(getattr(self, name)))

    @classmethod
    def from_mapping(cls, table: Mapping[str, Any]) -> "VolScaleRule":
        """The rule a ``[sizing]`` table declares. ``rule`` is required; unknown keys are refused."""
        if not isinstance(table, Mapping):
            raise VolScaleError(f"[sizing] is {table!r}, not a table")
        permitted = ("rule", "target_vol", "lookback_bars", "max_scale", "max_gross", "annualization_bars",
                     "max_zero_return_fraction")
        unknown = sorted(set(table) - set(permitted))
        if unknown:
            raise VolScaleError(f"[sizing] declares unknown key(s) {unknown}; permitted: {list(permitted)}")
        if "rule" not in table:
            raise VolScaleError(f"[sizing] declares no rule; the rules are {SIZING_RULES}")
        return cls(**{key: table[key] for key in permitted if key in table})

    def to_dict(self) -> dict[str, Any]:
        """The rule as plain JSON values, every default resolved (it enters ``config_id``)."""
        return {
            "rule": self.rule,
            "target_vol": self.target_vol,
            "lookback_bars": self.lookback_bars,
            "max_scale": self.max_scale,
            "max_gross": self.max_gross,
            "annualization_bars": self.annualization_bars,
            "max_zero_return_fraction": self.max_zero_return_fraction,
        }


@dataclass(frozen=True)
class ScaleDecision:
    """One refresh's scale and everything that decided it.

    ``sigma`` is the annualized estimate (NaN when none was taken),
    ``n_valid`` the finite returns in the window, ``zero_fraction`` the share
    of the window's ``lookback_bars`` returns that are exactly zero (NaN when
    the window was not read), and ``last_good`` the scale the failure path
    was bounded by.
    """

    scale: float
    sigma: float
    status: str
    reason: str | None
    n_valid: int
    zero_fraction: float
    decision_bar: str
    last_good: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_bar": self.decision_bar,
            "scale": self.scale,
            "sigma": self.sigma,
            "status": self.status,
            "reason": self.reason,
            "n_valid": self.n_valid,
            "zero_fraction": self.zero_fraction,
            "last_good": self.last_good,
        }


def base_book_returns(close: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Close-to-close returns of a book held at ``weights``, gross of costs.

    ``weights`` has a row per bar of ``close`` (same index): the weights the
    book held at that bar's close, as fractions of equity. A row that is
    entirely NaN means there was no book at that close; NaN inside a row is no
    position. The return at bar *t* is ``sum_i w[t-1, i] * (close[t, i] /
    close[t-1, i] - 1)``: weights at the prior close only (N1), so changing any
    row from *t* onward leaves every return through *t* unchanged.

    A held name without a positive finite close at *t - 1* or *t* -- halted,
    delisted, a bar the vendor did not deliver -- contributes nothing to that
    bar, as an unchanged mark would. A whole panel that stopped updating
    therefore returns exactly zero, which :func:`decide_scale`'s zero-return
    guard reads as a stale panel. The first bar, and any bar whose prior close
    held no book, is NaN.
    """
    if not isinstance(close, pd.DataFrame) or not isinstance(weights, pd.DataFrame):
        raise VolScaleError("base_book_returns takes a close DataFrame and a weights DataFrame")
    if not weights.index.equals(close.index):
        raise VolScaleError("weights must have one row per bar of the close panel, on the same index")
    extra = weights.columns.difference(close.columns)
    if len(extra):
        raise VolScaleError(f"weights name {len(extra)} sid(s) the close panel does not carry: {list(extra)[:5]}")
    # C order fixes the summation order of each bar's row: a row sum over a
    # Fortran-ordered copy of the same panel rounds differently in its last digit.
    prices = np.ascontiguousarray(close.to_numpy(dtype=float))
    held = np.ascontiguousarray(weights.reindex(columns=close.columns).to_numpy(dtype=float))
    out = np.full(len(close.index), np.nan)
    if len(close.index) < 2:
        return pd.Series(out, index=close.index, name="base_return")
    prior, now = prices[:-1], prices[1:]
    priced = np.isfinite(prior) & np.isfinite(now) & (prior > 0.0) & (now > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        moves = np.where(priced, now / np.where(priced, prior, 1.0) - 1.0, 0.0)
    prior_weights = held[:-1]
    has_book = np.isfinite(prior_weights).any(axis=1)
    contributions = np.where(np.isfinite(prior_weights), prior_weights, 0.0) * moves
    out[1:] = np.where(has_book, contributions.sum(axis=1), np.nan)
    return pd.Series(out, index=close.index, name="base_return")


def decide_scale(
    returns: Any,
    rule: VolScaleRule,
    *,
    last_good: float | None,
    decision_bar: Any,
    panel_last_bar: Any,
) -> ScaleDecision:
    """The scale a refresh at ``decision_bar`` sets, from base-book returns through it.

    ``returns`` are the base book's returns ending at the decision bar; the
    last ``rule.lookback_bars`` of them are the window. ``panel_last_bar`` is
    the last bar of the panel they were computed from; it must be the decision
    bar, or the estimate describes some earlier day. ``last_good`` is the
    scale of this book's last ``ok`` decision, or None.

    The checks run in the order of ``FALLBACK_REASONS``. A short window with no
    good decision yet is warm-up (scale 1); every other failure is the
    fallback ``min(1, last_good)``, or 1 without a last good scale.
    """
    if last_good is not None and not (math.isfinite(float(last_good)) and float(last_good) > 0.0):
        raise VolScaleError(f"last_good must be a positive finite scale or None, got {last_good!r}")
    bar = _session_day(decision_bar, "decision_bar")
    window = np.asarray(returns, dtype=float).reshape(-1)[-rule.lookback_bars :]
    finite = np.isfinite(window)
    n_valid = int(finite.sum())
    good = None if last_good is None else float(last_good)

    def fallback(reason: str, *, zero_fraction: float = math.nan, sigma: float = math.nan) -> ScaleDecision:
        status = "warmup" if reason == "insufficient_returns" and good is None else "fallback"
        scale = 1.0 if good is None else min(1.0, good)
        return ScaleDecision(
            scale=scale,
            sigma=sigma,
            status=status,
            reason=None if status == "warmup" else reason,
            n_valid=n_valid,
            zero_fraction=zero_fraction,
            decision_bar=bar,
            last_good=good,
        )

    if _session_day(panel_last_bar, "panel_last_bar") != bar:
        return fallback("decision_bar_not_last_bar")
    if n_valid < rule.lookback_bars:
        return fallback("insufficient_returns")
    zero_fraction = float(np.count_nonzero(window == 0.0)) / float(rule.lookback_bars)
    if zero_fraction > rule.max_zero_return_fraction:
        return fallback("zero_returns", zero_fraction=zero_fraction)
    # research/scripts/factor_lab.py::scale_constant_vol, by value (module docstring).
    sigma = float(np.std(window, ddof=1) * math.sqrt(rule.annualization_bars))
    if not math.isfinite(sigma) or sigma <= MIN_SIGMA:
        return fallback("degenerate_sigma", zero_fraction=zero_fraction, sigma=sigma)
    return ScaleDecision(
        scale=min(rule.max_scale, rule.target_vol / sigma),
        sigma=sigma,
        status="ok",
        reason=None,
        n_valid=n_valid,
        zero_fraction=zero_fraction,
        decision_bar=bar,
        last_good=good,
    )


def cap_gross(target: pd.Series, max_gross: float) -> tuple[pd.Series, float]:
    """Scale ``target`` down so its gross is at most ``max_gross``; never up.

    Returns the target and the factor applied (1.0 when the ceiling does not
    bind). NaN entries carry no position and stay NaN. A gross above the
    ceiling by no more than ``GROSS_TOLERANCE`` (relative) is rounding, not a
    breach, and is left alone.
    """
    _positive_finite("max_gross", max_gross)
    values = pd.Series(target, dtype=float)
    gross = float(np.nansum(np.abs(values.to_numpy(dtype=float))))
    if not math.isfinite(gross):
        raise VolScaleError(f"the target's gross is {gross}; a non-finite weight has no size to cap (N7)")
    if gross <= float(max_gross) * (1.0 + GROSS_TOLERANCE):
        return values, 1.0
    factor = float(max_gross) / gross
    return values * factor, factor


def session_exposure(weights: pd.DataFrame, *, sleeve_buffer: float = SLEEVE_BUFFER) -> pd.DataFrame:
    """Per bar: gross, the long side over equity, and the cash-sleeve fraction.

    ``weights`` are the book's held weights per bar as fractions of equity.
    ``long_over_equity`` is the sum of the positive weights; ``sleeve_fraction``
    is ``max(0, 1 - sleeve_buffer - long_over_equity)``, the equity a T-bill
    holding beside a cash buffer of ``sleeve_buffer`` would take
    (``prism.validation.hurdle.tbill_hurdle``'s ``sleeve_fraction``).
    """
    if not isinstance(weights, pd.DataFrame):
        raise VolScaleError("session_exposure takes a weights DataFrame")
    if isinstance(sleeve_buffer, bool) or not 0.0 <= float(sleeve_buffer) < 1.0:
        raise VolScaleError(f"sleeve_buffer must be in [0, 1), got {sleeve_buffer!r}")
    values = weights.to_numpy(dtype=float)
    finite = np.where(np.isfinite(values), values, 0.0)
    long_side = np.clip(finite, 0.0, None).sum(axis=1)
    return pd.DataFrame(
        {
            "gross": np.abs(finite).sum(axis=1),
            "long_over_equity": long_side,
            "sleeve_fraction": np.maximum(0.0, 1.0 - float(sleeve_buffer) - long_side),
        },
        index=weights.index,
    )


def _session_day(value: Any, name: str) -> str:
    """A bar as its ISO session date, whichever of label, date or timestamp it arrives as."""
    if isinstance(value, dt.datetime):  # pd.Timestamp included; a tz-aware stamp keeps its own date
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dt.date):
        return value.isoformat()
    try:
        return dt.date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        raise VolScaleError(f"{name} {value!r} is not a session date") from None


def _positive_finite(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise VolScaleError(f"{name} must be a finite number, got {value!r}")
    if float(value) <= 0.0:
        raise VolScaleError(f"{name} must be > 0, got {value!r}")


def _integer_at_least(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise VolScaleError(f"{name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise VolScaleError(f"{name} must be >= {minimum}, got {value!r}")
