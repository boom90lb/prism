"""The SPEC §7.7 regime step — per-cycle §7.5 telemetry for the live loop.

``run_daily_cycle`` consults this provider with the decision bar on EVERY
cycle (``prism.live.daily``, the ``regime`` seam): the handoff §8
precondition-(b) clock counts *sessions*, so non-refresh and halted cycles
record regime state too. The provider is telemetry only — it conditions
nothing until the sizing pre-registration
(``docs/sizing_preregistration.md``) ratifies and the separate
``regime_gross_scale`` action hook is armed in code; there is deliberately
no CLI path to that hook.

Blocks wired (all through the existing ``prism.regime`` surface — the
``regime.fetch`` adapters and pure feature math, nothing new is adapted):

* ``curve``     — level/slope/curvature via ``fetch_curve_state`` (FRED DGS).
* ``liquidity`` — net liquidity level + change via ``fetch_net_liquidity``
  (FRED WALCL/RRPONTSYD/WTREGEN; three-term identity by default — the
  DefiLlama fourth term rides only when a client is injected).
* ``inflation`` — real yield / breakeven / divergence via
  ``fetch_inflation_state`` (FRED DFII10 + T10YIE).
* ``vol``       — VIX level and the 1–3 month term ratio via
  ``FredClient.series`` on the registered ``fred_vix`` row (VIXCLS, VXVCLS)
  and the pure ``vix_term_slope``.

Blocks in the §7.5 contract with NO free fetch path through ``regime.fetch``
are named absent (``ABSENT_BLOCKS``), never silently missing: VRP's realized
leg needs market-proxy OHLC that ``regime.fetch`` does not adapt, VIX9D is
CBOE-only (registry-only row in ``regime.sources``), and the dollar factor
has no shipped feature math. Absence is recorded in every read.

Failure policy (N7: loud and NAMED, never silent, never fatal to the cycle):
a block that fails produces ``{"error": <exception class name>}`` in
``blocks`` plus one matching entry in ``failures`` and one loud
``logger.warning`` — class name only, no message text that could carry a URL
or key. The provider never raises out of the cycle and never returns a
silently-empty dict: every configured block appears either with values or
with a named failure. A read carrying ANY failure entry marks the session
NOT clean for the precondition-(b) 21-session clock (``docs/regime_step.md``).

Production-import-path safe (N8): pandas + ``prism.regime`` only.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from prism.regime.fetch import (
    DefiLlamaClient,
    FredClient,
    RegimeFetchError,
    fetch_curve_state,
    fetch_inflation_state,
    fetch_net_liquidity,
)
from prism.regime.liquidity import net_liquidity_change
from prism.regime.vol import vix_term_slope

logger = logging.getLogger(__name__)

#: The blocks every read reports (values or a named failure — never absent).
REGIME_BLOCKS: tuple[str, ...] = ("curve", "liquidity", "inflation", "vol")

#: §7.5 contract members with no free fetch path through ``regime.fetch`` —
#: named absent in every read rather than silently missing (N7). Building a
#: new external dependency for any of these is a separate, owner-gated change.
ABSENT_BLOCKS: dict[str, str] = {
    "vrp": "variance_risk_premium needs the market proxy's realized leg; regime.fetch adapts no equity price series",
    "vix9d_slope": "VIX9D is CBOE-only (regime.sources VOL_SOURCES); the CBOE CSV row is registry-only, unadapted",
    "dollar_factor": "no shipped feature math in prism.regime computes it",
}


class RegimeTelemetry:
    """Callable ``provider(decision_bar) -> dict`` for the §7.7 regime seam.

    ``fred`` is any ``FredClient``-shaped object (``series(series_id, start=,
    end=)``) and ``llama`` any ``DefiLlamaClient``-shaped object
    (``stablecoin_float(in_millions=)``) — injectable so every path runs
    offline on canned series (the ``regime.fetch`` testing idiom). ``llama``
    defaults to ``None``: the liquidity block reads the three-term identity,
    and the R4 stablecoin fourth term rides only when a client is injected
    (one fewer external failure surface on the precondition-(b) clock).

    ``lookback_days`` is the calendar fetch window ending at the decision bar
    (causal: ``end=decision_bar`` bounds every request); it only needs to
    cover the change horizon plus release lags. ``change_periods`` feeds
    ``net_liquidity_change`` (default 5 ≈ one week of dailies).

    The returned dict: ``{"decision_bar", "blocks", "failures", "absent"}``
    with every block of :data:`REGIME_BLOCKS` present in ``blocks`` — values
    (non-finite floats become ``None``, so every read is strict-JSON-safe) or
    ``{"error": <class name>}`` mirrored into ``failures``. Any failure entry
    = session NOT clean for the handoff §8 precondition-(b) 21-session clock.
    """

    def __init__(
        self,
        fred: FredClient,
        llama: DefiLlamaClient | None = None,
        *,
        lookback_days: int = 120,
        change_periods: int = 5,
    ) -> None:
        if lookback_days < 1:
            raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
        self._fred = fred
        self._llama = llama
        self._lookback_days = lookback_days
        self._change_periods = change_periods

    @classmethod
    def from_env(cls, **kwargs: object) -> "RegimeTelemetry":
        """Live provider from ``FRED_API_KEY`` (``FredClient.from_env``).

        A missing key raises here, at construction — the operator armed the
        regime step, so a cycle without its record must not run quietly (N7).
        The key lives in the client and travels only in its request params;
        it never appears in the telemetry dict, the ledger, or a log line.
        """
        return cls(FredClient.from_env(), **kwargs)

    def __call__(self, decision_bar: str) -> dict:
        blocks: dict[str, dict] = {}
        failures: list[dict] = []
        for name, compute in (
            ("curve", self._curve),
            ("liquidity", self._liquidity),
            ("inflation", self._inflation),
            ("vol", self._vol),
        ):
            try:
                blocks[name] = compute(decision_bar)
            except Exception as exc:  # noqa: BLE001 — telemetry never takes down the cycle
                blocks[name] = {"error": type(exc).__name__}
                failures.append({"block": name, "error": type(exc).__name__})
                logger.warning(
                    "REGIME BLOCK FAILED %s (%s) at %s — named failure recorded; this session is "
                    "NOT clean for the handoff §8 precondition-(b) 21-session clock (N7)",
                    name,
                    type(exc).__name__,
                    decision_bar,
                )
        return {
            "decision_bar": decision_bar,
            "blocks": blocks,
            "failures": failures,
            "absent": sorted(ABSENT_BLOCKS),
        }

    # ------------------------------------------------------------------ blocks

    def _window_start(self, decision_bar: str) -> str:
        return str((pd.Timestamp(decision_bar) - pd.Timedelta(days=self._lookback_days)).date())

    def _curve(self, decision_bar: str) -> dict:
        state = fetch_curve_state(self._fred, start=self._window_start(decision_bar), end=decision_bar)
        return _latest_finite(state, "curve")

    def _liquidity(self, decision_bar: str) -> dict:
        net = fetch_net_liquidity(
            self._fred, self._llama, start=self._window_start(decision_bar), end=decision_bar
        )
        frame = pd.DataFrame(
            {"net_liquidity": net, "net_liquidity_change": net_liquidity_change(net, self._change_periods)}
        )
        values = _latest_finite(frame, "liquidity")
        values["stablecoin_term"] = self._llama is not None
        return values

    def _inflation(self, decision_bar: str) -> dict:
        state = fetch_inflation_state(self._fred, start=self._window_start(decision_bar), end=decision_bar)
        return _latest_finite(state, "inflation")

    def _vol(self, decision_bar: str) -> dict:
        start = self._window_start(decision_bar)
        vix = self._fred.series("VIXCLS", start=start, end=decision_bar)
        vix3m = self._fred.series("VXVCLS", start=start, end=decision_bar)
        frame = pd.DataFrame({"vix": vix, "vix3m": vix3m, "term_ratio": vix_term_slope(vix, vix3m)})
        return _latest_finite(frame, "vol")


def _latest_finite(frame: pd.DataFrame, block: str) -> dict:
    """Last row carrying any finite value, JSON-safe, with its observation date.

    Non-finite fields in that row become ``None`` (strict JSON has no NaN);
    ``asof`` is the row's own date, so a stale read is visible as a lagging
    ``asof``, never disguised as fresh. A window with NO finite observation
    raises — an empty regime read must become a NAMED failure upstream, never
    a silently-empty block (N7).
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    finite = numeric.dropna(how="all")
    if finite.empty:
        raise RegimeFetchError(f"{block}: no finite observations in the fetch window (N7: absence is not data)")
    row = finite.iloc[-1]
    values: dict = {
        str(column): (float(value) if math.isfinite(float(value)) else None) for column, value in row.items()
    }
    values["asof"] = str(finite.index[-1].date())
    return values
