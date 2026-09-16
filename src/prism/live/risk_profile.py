"""Operator risk-profile surface (W6 — schema FROZEN 2026-07-23; §3.13 files).

Named profiles map onto already-ratified pins or *stricter* subsets. See
``docs/risk_profile_schema.md``. Field names are frozen; renames need a
dedicated commit.

**Declared profile files (§3.13).** The four ids are also *files* of the
reference project, one per id, under ``examples/reference/profiles/``:
:data:`PROFILE_FILES` is that map, and the ids stay accepted as aliases naming
the file they map to. A file carries exactly the top-level keys section 3.13
declares — ``name``, ``min_account_status``, and the optional ``cash_buffer``,
``long_only`` and ``min_regime_scale`` — plus the ``[[limits]]`` array of
section 3.7, and :func:`load_declared_profile` parses it, enforcing the
tighten-only discipline *at load* rather than in the engine: the engine enforces
the policy a project selects and holds no universal threshold.

The registry in this module stays the construction path, exactly as
``examples/*/reference.toml`` are declarations of record for books the registry
still builds. :func:`assert_declared_matches_resolved` is what keeps the two
honest, and ``tests/test_risk_profile.py`` runs it for every id.

Two controls of this surface survive the move only as a *report*: the sleeve
capital bands of :class:`HedgePolicy` and the de-gross arming flag fit none of
the four limit classes of section 3.7, so they are named in
:data:`INEXPRESSIBLE_CONTROLS` and stay resolved in code rather than being
silently dropped from the declaration.

**G6:** ``research_paper`` resolves to the certified B1 *paper*
``DailyBookConfig`` defaults (momentum book, decile_neutral, monthly cadence).
A profile-aware path under that id must remain bit-identical to a bare
certified config. Unknown profile ids fail loud (N7). Profiles never loosen
pins. De-gross arming is **not** a profile field action — sizing GO commit
after handoff §8 (a)+(b). Freeze of the schema is not a GO authorization.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from prism.live.daily import DailyBookConfig
from prism.portfolio.limits import Limit, load_limits

# Product enum — frozen with docs/risk_profile_schema.md (2026-07-23).
PROFILE_IDS: frozenset[str] = frozenset({"research_paper", "conservative", "balanced", "assertive"})

#: The reference project's declared profile files, one per id, relative to the
#: repository root (§3.13, Q7/Q19). ``research_paper`` maps onto the existing
#: ``paper.toml``, whose ``name`` field has always been ``research_paper``:
#: adding a second file with the same name would give one profile two
#: declarations, which is the failure the file map exists to prevent.
PROFILE_FILES: Mapping[str, str] = {
    "research_paper": "examples/reference/profiles/paper.toml",
    "conservative": "examples/reference/profiles/conservative.toml",
    "balanced": "examples/reference/profiles/balanced.toml",
    "assertive": "examples/reference/profiles/assertive.toml",
}

#: Controls of this surface that no limit class of section 3.7 can express, with
#: the reason and where they continue to live. Reported rather than dropped: a
#: control that disappears in a migration is a control nobody enforces.
INEXPRESSIBLE_CONTROLS: Mapping[str, str] = {
    "hedge.equity_sleeve": (
        "a capital band over sleeves, not a constraint on one account's holdings; the four §3.7 "
        "classes measure gross, per-name weight, bands, variance, turnover and participation, and "
        "none of them ranges over sleeves. Resolved in code by HedgePolicy."
    ),
    "hedge.trend_sleeve": "as hedge.equity_sleeve; resolved in code by HedgePolicy.",
    "hedge.crypto_book": "as hedge.equity_sleeve; resolved in code by HedgePolicy.",
    "hedge.de_gross_armed": (
        "an arming state, not a limit: it records that the sizing GO commit happened, and §3.7 "
        "explicitly puts the de-gross ceiling on regime_scale (min_regime_scale) rather than in the "
        "limit vocabulary. Resolved in code; the bound is declared in the file."
    ),
    "book_config.book": (
        "the book's identity — which strategy is running — declared by the strategy "
        "configuration's [strategy] block (§3.3). A limit constrains a portfolio; it does not "
        "choose one."
    ),
    "book_config.decile": (
        "a rank-to-weight parameter of the strategy configuration ([strategy.components] "
        "to_weights_params, §3.3), not a portfolio limit."
    ),
    "book_config.decision_every": "a cadence in the strategy configuration (§3.3), not a limit.",
    "book_config.no_trade_band": (
        "§3.7 names it explicitly as a control that is not a limit: it belongs to the "
        "[construction.band] block, because it changes which trades are taken, never which "
        "holdings are legal."
    ),
    "book_config.min_order_notional": (
        "the execution profile's min_notional policy dust floor (§3.7), which is a different "
        "quantity from the venue minimum on the capabilities row."
    ),
    "book_config.whole_shares": (
        "the execution profile's quantity_increment_default (§3.9); a profile limit named "
        "quantity_increment exists but is the physical venue increment, not a book setting."
    ),
    "book_config.adv_window_bars": (
        "the window of the [dollar_volume.participation] statistic in the cost stack (§3.6, D10); "
        "the profile's participation_per_name limit supplies only the cap fraction."
    ),
    "book_config.vol_ewma_bars": "a risk-model window (§3.5 risk/*.toml), not a limit.",
    "book_config.position_size": "a directional sizing parameter of the ensemble book (§3.3).",
}

#: Where the reference project lives when this package is an editable install of
#: its own checkout, which is how the live system runs it.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

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
                f"capital band must satisfy 0 <= lo <= hi <= 1, got " f"[{self.capital_lo}, {self.capital_hi}]"
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
            raise ValueError(f"sum of enabled sleeve capital_hi must be <= 1.0, got {total_hi}")


@dataclass(frozen=True)
class RiskProfile:
    """Resolved operator profile: construction config + hedge policy + id."""

    profile_id: str
    book_config: DailyBookConfig
    hedge: HedgePolicy
    # Explicit: profiles never freer than this g when de-gross is later armed.
    de_gross_g_ceiling: float = RATIFIED_DEGROSS_G
    #: The reference project's file this id names (§3.13). Recorded, not read:
    #: the registry above stays the construction path, and
    #: :func:`assert_declared_matches_resolved` is what keeps the two agreeing.
    declared_file: str = ""

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
            "declared_file": self.declared_file,
            "schema_status": "FROZEN",
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
            f"profiles may only tighten max_gross relative to certified pin " f"{certified}: got {requested}"
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
            declared_file=PROFILE_FILES[pid],
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
            declared_file=PROFILE_FILES[pid],
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
            declared_file=PROFILE_FILES[pid],
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
        declared_file=PROFILE_FILES[pid],
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


# --------------------------------------------------------------------------
# Declared profile files (§3.13)
# --------------------------------------------------------------------------

#: Exactly the top-level keys a ``profiles/*.toml`` file carries (§3.13), plus
#: the ``[[limits]]`` array of §3.7. An unknown key is refused: a mis-spelled
#: ``min_regime_scale`` that silently kept the default is the defect this
#: loader exists to prevent.
PROFILE_FILE_KEYS: frozenset[str] = frozenset(
    {"name", "min_account_status", "cash_buffer", "long_only", "min_regime_scale", "limits"}
)

ACCOUNT_STATUSES: tuple[str, ...] = ("undetermined", "measured")


@dataclass(frozen=True)
class DeclaredProfile:
    """One parsed ``profiles/*.toml`` file: the §3.13 keys and the §3.7 limits."""

    name: str
    path: str
    min_account_status: str
    limits: tuple[Limit, ...] = ()
    cash_buffer: float | None = None
    long_only: bool = False
    min_regime_scale: float | None = None
    #: What ``load_limits`` reported as ignored (a priority on an action restriction).
    ignored: tuple[str, ...] = field(default_factory=tuple)

    def limit(self, name: str) -> Limit | None:
        """The declared limit of that name, or ``None``."""
        for limit in self.limits:
            if limit.name == name:
                return limit
        return None

    def limit_value(self, name: str, default: float | None = None) -> float | None:
        """The scalar value of the declared limit of that name, or ``default``."""
        limit = self.limit(name)
        return default if limit is None else float(limit.value())

    def to_public_dict(self) -> dict[str, Any]:
        """JSON-safe record of the declaration (no secrets)."""
        return {
            "name": self.name,
            "path": self.path,
            "min_account_status": self.min_account_status,
            "cash_buffer": self.cash_buffer,
            "long_only": self.long_only,
            "min_regime_scale": self.min_regime_scale,
            "limits": [
                {
                    "name": limit.name,
                    "class": limit.cls.value,
                    "priority": limit.priority,
                    "params": dict(limit.params),
                }
                for limit in self.limits
            ],
            "ignored": list(self.ignored),
        }


def _require_float(payload: Mapping[str, Any], key: str, where: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {key} must be a number, got {value!r}")
    return float(value)


def check_tighten_only(profile: DeclaredProfile) -> None:
    """Refuse a declaration that is freer than a ratified pin (schema §1, hard rule 1).

    This is the *reference project's* rule, declared in its files and enforced when they
    are loaded; the engine holds no universal threshold and enforces whatever policy the
    project it is pointed at selects (§3.13, Q5/Q7).

    ``min_regime_scale`` is checked against :data:`RATIFIED_DEGROSS_G` in the file's own
    reading — a floor on ``regime_scale``, so a floor *above* the ratified de-gross depth
    would refuse to de-gross as far as the sizing ratification requires, which is freer,
    not tighter.
    """
    where = profile.path
    gross = profile.limit_value("max_gross")
    if gross is not None and not (0.0 < gross <= CERTIFIED_MAX_GROSS + 1e-15):
        raise ValueError(
            f"{where}: max_gross {gross} is not a tightening of the certified pin {CERTIFIED_MAX_GROSS}; "
            "profiles may only tighten (docs/risk_profile_schema.md §1)"
        )
    per_name = profile.limit_value("max_name_weight")
    certified_per_name = CERTIFIED_B1_PAPER_CONFIG.max_symbol_abs_weight
    if per_name is not None and not (0.0 < per_name <= certified_per_name + 1e-15):
        raise ValueError(
            f"{where}: max_name_weight {per_name} is not a tightening of the certified per-name pin "
            f"{certified_per_name} (docs/risk_profile_schema.md §1)"
        )
    if profile.min_regime_scale is not None and profile.min_regime_scale > RATIFIED_DEGROSS_G + 1e-15:
        raise ValueError(
            f"{where}: min_regime_scale {profile.min_regime_scale} floors the regime scale above the "
            f"ratified de-gross pin {RATIFIED_DEGROSS_G}; a profile is never freer than the ratified g "
            "(docs/sizing_preregistration.md §3)"
        )


def load_declared_profile(path: str | Path, *, tighten_only: bool = True) -> DeclaredProfile:
    """Parse one ``profiles/*.toml`` file into its §3.13 keys and §3.7 limits.

    ``tighten_only`` applies the reference project's discipline; a project with its own
    pins loads with it off and enforces its own. ``min_account_status = "measured"`` is
    refused: §3.13 requires it to carry ``min_account_usd``, the record id of the
    measurement, and the profile, broker and execution profile it applies to, and this
    loader implements none of those key names. Refusing is the honest reading — inventing
    names for an attestation nobody has written would make the attestation unverifiable.
    """
    where = str(path)
    payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(payload) - PROFILE_FILE_KEYS
    if unknown:
        raise ValueError(
            f"{where}: unknown top-level keys {sorted(unknown)}; §3.13 declares {sorted(PROFILE_FILE_KEYS)}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}: every profile declares a name")
    status = payload.get("min_account_status")
    if status not in ACCOUNT_STATUSES:
        raise ValueError(f"{where}: min_account_status must be one of {ACCOUNT_STATUSES}, got {status!r}")
    if status == "measured":
        raise ValueError(
            f"{where}: min_account_status = 'measured' additionally requires min_account_usd, the record id "
            "of the measurement, and the profile, broker and execution profile it applies to (§3.13). This "
            "loader implements 'undetermined' only; a capital-suitability claim is written by the project "
            "owner, not inferred here"
        )
    cash_buffer = _require_float(payload, "cash_buffer", where)
    if cash_buffer is not None and not (0.0 <= cash_buffer < 1.0):
        raise ValueError(f"{where}: cash_buffer must satisfy 0 <= b < 1, got {cash_buffer}")
    long_only = payload.get("long_only", False)
    if not isinstance(long_only, bool):
        raise ValueError(f"{where}: long_only must be a boolean, got {long_only!r}")
    min_regime_scale = _require_float(payload, "min_regime_scale", where)
    if min_regime_scale is not None and not (0.0 < min_regime_scale <= 1.0):
        raise ValueError(f"{where}: min_regime_scale must satisfy 0 < s <= 1, got {min_regime_scale}")
    rows = payload.get("limits") or []
    if not isinstance(rows, list):
        raise ValueError(f"{where}: [[limits]] must be an array of tables, got {type(rows).__name__}")
    limits, ignored = load_limits(rows, source=where)
    profile = DeclaredProfile(
        name=str(name),
        path=where,
        min_account_status=str(status),
        limits=limits,
        cash_buffer=cash_buffer,
        long_only=long_only,
        min_regime_scale=min_regime_scale,
        ignored=ignored,
    )
    if tighten_only:
        check_tighten_only(profile)
    return profile


def declared_profile_path(profile_id: str, *, root: str | Path | None = None) -> Path:
    """The file the id names, under ``root`` (the repository by default)."""
    pid = _require_profile_id(profile_id)
    path = Path(root) if root is not None else REPO_ROOT
    resolved = path / PROFILE_FILES[pid]
    if not resolved.is_file():
        raise FileNotFoundError(
            f"profile {pid!r} names {PROFILE_FILES[pid]}, which does not exist under {path}; the four ids "
            "are files of the reference project (§3.13) and the file is the declaration"
        )
    return resolved


def declared_profile(profile_id: str, *, root: str | Path | None = None) -> DeclaredProfile:
    """The parsed declaration for one profile id."""
    return load_declared_profile(declared_profile_path(profile_id, root=root))


def inexpressible_controls() -> Mapping[str, str]:
    """Controls of this surface no §3.7 limit class expresses, and where they live.

    Reported so the migration to declared files loses nothing silently: every entry is a
    control that still binds, resolved in code, with the reason no limit class can carry it.
    """
    return dict(INEXPRESSIBLE_CONTROLS)


def assert_declared_matches_resolved(profile_id: str, *, root: str | Path | None = None) -> DeclaredProfile:
    """Fail loud when a declared file and the resolved profile disagree.

    The registry is the construction path and the file is the declaration of record; a
    declaration that has drifted from what the code builds is worse than no declaration,
    because it reads as authoritative. Compares what the file *can* say: the id, the
    gross pin, the per-name pin, and the de-gross bound.
    """
    pid = _require_profile_id(profile_id)
    declared = declared_profile(pid, root=root)
    resolved = resolve_risk_profile(pid)
    mismatches: list[str] = []
    if declared.name != pid:
        mismatches.append(f"name {declared.name!r} != profile id {pid!r}")
    gross = declared.limit_value("max_gross")
    if gross is None or abs(gross - resolved.book_config.max_gross) > 1e-15:
        mismatches.append(f"max_gross {gross!r} != resolved {resolved.book_config.max_gross}")
    per_name = declared.limit_value("max_name_weight")
    if per_name is None or abs(per_name - resolved.book_config.max_symbol_abs_weight) > 1e-15:
        mismatches.append(f"max_name_weight {per_name!r} != resolved {resolved.book_config.max_symbol_abs_weight}")
    if declared.min_regime_scale is not None and abs(declared.min_regime_scale - resolved.de_gross_g_ceiling) > 1e-15:
        mismatches.append(
            f"min_regime_scale {declared.min_regime_scale} != resolved de_gross_g_ceiling "
            f"{resolved.de_gross_g_ceiling}"
        )
    if resolved.declared_file != PROFILE_FILES[pid]:
        mismatches.append(f"resolved declared_file {resolved.declared_file!r} != {PROFILE_FILES[pid]!r}")
    if mismatches:
        raise AssertionError(
            f"{declared.path} has drifted from resolve_risk_profile({pid!r}): " + "; ".join(mismatches)
        )
    return declared
