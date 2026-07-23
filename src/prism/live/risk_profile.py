"""Operator risk-profile surface (W6 draft — schema freeze lags).

Provisional product mapping from named profiles onto already-ratified pins
or *stricter* subsets. See ``docs/risk_profile_schema.md`` (DRAFT until
owner freeze / J6).

**G6 soft gate:** ``research_paper`` resolves to the certified B1 *paper*
``DailyBookConfig`` defaults (momentum book, decile_neutral, monthly cadence).
A profile-aware path under that id must remain bit-identical to a bare
certified config. Unknown profile ids fail loud (N7). Profiles never loosen
pins. De-gross arming is **not** a profile field action — sizing GO commit.

This module is legal under a DRAFT schema (code may draft against provisional
names). Shipping profile-aware *live deploy* before schema freeze is not
deserved v0.4.0 product.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Mapping

from prism.live.daily import DailyBookConfig

# Product enum — provisional until docs/risk_profile_schema.md freezes.
PROFILE_IDS: frozenset[str] = frozenset(
    {"research_paper", "conservative", "balanced", "assertive"}
)

# Certified B1 paper-loop construction defaults (prism.scripts.paper_loop
# momentum book with CLI defaults: decile 0.10, decision_every 21, OPG
# whole-shares). This is the *promotion instrument* path, not the batch
# demotion walk-forward pin (max_symbol_abs_weight 0.35 there).
CERTIFIED_B1_PAPER_CONFIG = DailyBookConfig(
    book="decile_neutral",
    decile=0.10,
    decision_every=21,
    max_gross=1.0,
    max_symbol_abs_weight=0.10,
    no_trade_band=0.0,
    max_participation=None,
    min_order_notional=1.0,
    whole_shares=True,
    adv_window_bars=20,
    vol_ewma_bars=63,
    position_size=0.0,
)

# Certified gross pin used for tighten-only checks (construction).
CERTIFIED_MAX_GROSS: float = 1.0
# Ratified crash de-gross pin (docs/sizing_preregistration.md §3) — profiles
# may reference it only as the *ceiling* when the hook is armed elsewhere.
RATIFIED_DEGROSS_G: float = 0.5


@dataclass(frozen=True)
class SleeveBand:
    """Static capital band for one sleeve. ``hi`` is a fraction of deployable capital."""

    enabled: bool
    capital_lo: float = 0.0
    capital_hi: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.capital_lo <= self.capital_hi <= 1.0:
            raise ValueError(
                f"capital band must satisfy 0 <= lo <= hi <= 1, got "
                f"[{self.capital_lo}, {self.capital_hi}]"
            )
        if self.enabled and self.capital_hi <= 0.0:
            raise ValueError("enabled sleeve requires capital_hi > 0")


@dataclass(frozen=True)
class HedgePolicy:
    """Declared multi-sleeve composition (not ad-hoc daily discretion)."""

    equity_sleeve: SleeveBand
    trend_sleeve: SleeveBand
    crypto_book: SleeveBand
    # Armed only by the sizing GO commit after handoff §8 (a)+(b) — never by profile alone.
    de_gross_armed: bool = False

    def __post_init__(self) -> None:
        total_hi = 0.0
        for band in (self.equity_sleeve, self.trend_sleeve, self.crypto_book):
            if band.enabled:
                total_hi += band.capital_hi
        if total_hi > 1.0 + 1e-12:
            raise ValueError(
                f"sum of enabled sleeve capital_hi must be <= 1.0, got {total_hi}"
            )


@dataclass(frozen=True)
class RiskProfile:
    """Resolved operator profile: construction config + hedge policy + id."""

    profile_id: str
    book_config: DailyBookConfig
    hedge: HedgePolicy
    # Explicit: profiles never freer than this g when de-gross is later armed.
    de_gross_g_ceiling: float = RATIFIED_DEGROSS_G

    def to_public_dict(self) -> dict:
        """JSON-safe surface for run-dir / claim packets (no secrets)."""
        return {
            "profile_id": self.profile_id,
            "book_config": asdict(self.book_config),
            "hedge": {
                "equity_sleeve": asdict(self.hedge.equity_sleeve),
                "trend_sleeve": asdict(self.hedge.trend_sleeve),
                "crypto_book": asdict(self.hedge.crypto_book),
                "de_gross_armed": self.hedge.de_gross_armed,
            },
            "de_gross_g_ceiling": self.de_gross_g_ceiling,
            "schema_status": "DRAFT",
        }


def _require_profile_id(profile_id: str) -> str:
    if profile_id not in PROFILE_IDS:
        known = ", ".join(sorted(PROFILE_IDS))
        raise ValueError(
            f"unknown risk profile {profile_id!r}: expected one of {{{known}}} "
            f"(silent fallback forbidden — docs/risk_profile_schema.md §1)"
        )
    return profile_id


def _tighten_max_gross(requested: float, *, certified: float = CERTIFIED_MAX_GROSS) -> float:
    if not (requested > 0.0) or not (requested <= certified + 1e-15):
        raise ValueError(
            f"profiles may only tighten max_gross relative to certified pin "
            f"{certified}: got {requested}"
        )
    return float(requested)


def resolve_risk_profile(
    profile_id: str,
    *,
    max_gross: float | None = None,
    de_gross_armed: bool = False,
) -> RiskProfile:
    """Resolve a named profile to construction + hedge policy.

    ``research_paper`` ignores ``max_gross`` overrides that would diverge from
    the certified paper config (G6). Other profiles accept tighten-only
    ``max_gross`` (defaults: conservative 0.5, balanced/assertive certified 1.0).
    ``de_gross_armed`` is accepted as a *record* of the sizing arming state —
    this function never arms the hook.
    """
    pid = _require_profile_id(profile_id)

    if pid == "research_paper":
        if de_gross_armed:
            raise ValueError(
                "research_paper is the promotion instrument: de_gross must stay "
                "unarmed (telemetry only) — docs/risk_profile_schema.md §1"
            )
        if max_gross is not None and abs(max_gross - CERTIFIED_B1_PAPER_CONFIG.max_gross) > 1e-15:
            raise ValueError(
                "research_paper max_gross is pinned to the certified B1 paper path; "
                f"got {max_gross}, expected {CERTIFIED_B1_PAPER_CONFIG.max_gross}"
            )
        return RiskProfile(
            profile_id=pid,
            book_config=CERTIFIED_B1_PAPER_CONFIG,
            hedge=HedgePolicy(
                equity_sleeve=SleeveBand(enabled=True, capital_lo=1.0, capital_hi=1.0),
                trend_sleeve=SleeveBand(enabled=False),
                crypto_book=SleeveBand(enabled=False),
                de_gross_armed=False,
            ),
            de_gross_g_ceiling=RATIFIED_DEGROSS_G,
        )

    if pid == "conservative":
        gross = _tighten_max_gross(0.5 if max_gross is None else max_gross)
        # sum(capital_hi) over enabled sleeves <= 1.0 (schema §2). Trend floor
        # is the lo when the sleeve is admitted — not optimized G4b weights.
        return RiskProfile(
            profile_id=pid,
            book_config=_with_max_gross(CERTIFIED_B1_PAPER_CONFIG, gross),
            hedge=HedgePolicy(
                equity_sleeve=SleeveBand(enabled=True, capital_lo=0.5, capital_hi=0.7),
                trend_sleeve=SleeveBand(enabled=True, capital_lo=0.2, capital_hi=0.3),
                crypto_book=SleeveBand(enabled=False),
                de_gross_armed=de_gross_armed,
            ),
        )

    if pid == "balanced":
        gross = _tighten_max_gross(CERTIFIED_MAX_GROSS if max_gross is None else max_gross)
        return RiskProfile(
            profile_id=pid,
            book_config=_with_max_gross(CERTIFIED_B1_PAPER_CONFIG, gross),
            hedge=HedgePolicy(
                equity_sleeve=SleeveBand(enabled=True, capital_lo=0.6, capital_hi=0.7),
                trend_sleeve=SleeveBand(enabled=True, capital_lo=0.3, capital_hi=0.3),
                crypto_book=SleeveBand(enabled=False),
                de_gross_armed=de_gross_armed,
            ),
        )

    # assertive
    gross = _tighten_max_gross(CERTIFIED_MAX_GROSS if max_gross is None else max_gross)
    return RiskProfile(
        profile_id=pid,
        book_config=_with_max_gross(CERTIFIED_B1_PAPER_CONFIG, gross),
        hedge=HedgePolicy(
            equity_sleeve=SleeveBand(enabled=True, capital_lo=0.5, capital_hi=0.6),
            trend_sleeve=SleeveBand(enabled=True, capital_lo=0.3, capital_hi=0.4),
            crypto_book=SleeveBand(enabled=False),
            de_gross_armed=de_gross_armed,
        ),
    )


def book_config_matches_certified_paper(config: DailyBookConfig) -> bool:
    """True iff ``config`` is field-equal to the certified B1 paper instrument."""
    certified = CERTIFIED_B1_PAPER_CONFIG
    for f in fields(DailyBookConfig):
        if getattr(config, f.name) != getattr(certified, f.name):
            return False
    return True


def assert_research_paper_bit_identity(config: DailyBookConfig) -> None:
    """Fail loud if a claimed research_paper config diverges from certified paper."""
    if not book_config_matches_certified_paper(config):
        raise AssertionError(
            "research_paper DailyBookConfig diverges from CERTIFIED_B1_PAPER_CONFIG "
            f"(got {asdict(config)}, expected {asdict(CERTIFIED_B1_PAPER_CONFIG)})"
        )


def _with_max_gross(base: DailyBookConfig, max_gross: float) -> DailyBookConfig:
    kwargs = {f.name: getattr(base, f.name) for f in fields(DailyBookConfig)}
    kwargs["max_gross"] = max_gross
    return DailyBookConfig(**kwargs)


def validate_profile_payload(payload: Mapping[str, object]) -> str:
    """Validate a claim-packet / run-dir profile id field; return the id."""
    raw = payload.get("profile_id", payload.get("profile"))
    if raw is None:
        raise ValueError("profile payload missing profile_id")
    if not isinstance(raw, str):
        raise ValueError(f"profile_id must be str, got {type(raw).__name__}")
    return _require_profile_id(raw)
