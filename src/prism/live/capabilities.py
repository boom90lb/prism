"""The ``capabilities`` table refreshed from a broker, and the submission gate (§3.2, §3.9).

Two jobs, and they are deliberately separate.

**The table.** :func:`refresh_capabilities` pulls
:class:`~prism.live.broker.CapabilityRow` rows for a set of sids at a declared
instant and lays them out as the ``capabilities`` table of section 3.2 —
``sid``-indexed, list columns JSON-encoded, the availability and revision fields
every table in that section carries. That frame is exactly what
``prism.portfolio.validate.capability_row`` reads, so a live batch and an
evaluated one are validated against the same shape of evidence.

**The gate.** Section 3.9 declares that capabilities and buying power are
revalidated *at submission*, and :func:`revalidate_for_submission` is that
revalidation: rows refreshed (or proven fresh against the profile's
``[capabilities] max_age_hours``), buying power read from the venue, the
account's identity checked against what the caller expected, and the whole batch
put back through ``portfolio/validate.py`` with ``require_capabilities = True``
so a sid with no dated row fails rather than passing unchecked.

The gate never *decides*. It returns a report; the caller cancels, drops, or
scales per its recovery policy (§3.7). And it holds no universal threshold: the
buying power it checks against is the number the venue reported, never a
$25,000 pattern-day-trading rule this engine invented.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from prism.live.broker import CAPABILITY_COLUMNS, AccountIdentity, Broker, CapabilityRow
from prism.portfolio.limits import Limit, LimitContext
from prism.portfolio.validate import Finding, ValidationReport, ValidationSettings, validate_batch

__all__ = [
    "CAPABILITY_FRAME_COLUMNS",
    "SubmissionCheck",
    "capabilities_frame",
    "capability_ages_hours",
    "refresh_capabilities",
    "revalidate_for_submission",
    "stale_capability_sids",
]

logger = logging.getLogger(__name__)

#: The table's columns as they appear on the sid-indexed frame (``sid`` is the index).
CAPABILITY_FRAME_COLUMNS: tuple[str, ...] = tuple(column for column in CAPABILITY_COLUMNS if column != "sid")


def capabilities_frame(rows: Iterable[CapabilityRow]) -> pd.DataFrame:
    """The section 3.2 ``capabilities`` table for ``rows``, indexed by sid.

    Two rows for one sid are refused rather than resolved here: the as-of selection
    rule of section 3.2 (latest ``valid_at`` at or before the cutoff, highest revision
    within a source, source priority across sources) belongs to the dataset view, and
    a second rule that quietly picked a winner would be a second answer to the same
    question.
    """
    materialized = list(rows)
    frame = pd.DataFrame(
        [row.to_row() for row in materialized],
        columns=list(CAPABILITY_COLUMNS),
    )
    if not materialized:
        return frame.set_index("sid")
    duplicated = frame["sid"][frame["sid"].duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"two capability rows for {sorted(set(duplicated))}; the as-of selection of §3.2 picks the "
            "row, and this frame is the selected one"
        )
    return frame.set_index("sid")


def refresh_capabilities(broker: Broker, sids: Sequence[str], *, as_of: str) -> pd.DataFrame:
    """Pull dated capability rows for ``sids`` from the broker as one table."""
    return capabilities_frame(broker.capabilities(list(sids), as_of))


def capability_ages_hours(capabilities: pd.DataFrame, *, now: str | pd.Timestamp) -> pd.Series:
    """Age in hours of each row's ``valid_at`` at ``now``.

    An unparseable or missing ``valid_at`` is infinitely old, not zero: the row's
    freshness is unknown, and unknown freshness is stale (the fail-safe direction).
    """
    reference = pd.Timestamp(now)
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    stamped = pd.to_datetime(capabilities.get("valid_at"), errors="coerce", utc=True, format="mixed")
    if stamped is None or len(capabilities.index) == 0:
        return pd.Series(dtype=float)
    ages = (reference.tz_convert("UTC") - stamped).dt.total_seconds() / 3600.0
    ages = pd.Series(ages.to_numpy(dtype=float), index=capabilities.index, dtype=float)
    return ages.fillna(float("inf"))


def stale_capability_sids(
    capabilities: pd.DataFrame, *, now: str | pd.Timestamp, max_age_hours: float
) -> tuple[str, ...]:
    """The sids whose row is older than ``max_age_hours`` at ``now``.

    ``max_age_hours = 0`` means every row must be refreshed at submission, which is
    what the synthetic profile declares; a negative value is refused rather than read
    as "never stale".
    """
    if not math.isfinite(max_age_hours) or max_age_hours < 0.0:
        raise ValueError(f"[capabilities] max_age_hours must be finite and >= 0, got {max_age_hours!r}")
    ages = capability_ages_hours(capabilities, now=now)
    stale = ages[ages > float(max_age_hours)]
    return tuple(str(sid) for sid in stale.index)


@dataclass(frozen=True)
class SubmissionCheck:
    """What the submission gate found, as a record rather than a decision."""

    report: ValidationReport
    capabilities: pd.DataFrame
    refreshed: bool
    buying_power: float
    required_buying_power: float
    account: AccountIdentity | None = None

    @property
    def ok(self) -> bool:
        """The verdict of the report this check produced."""
        return self.report.ok

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe record of the gate, for the section 3.12 decision row."""
        return {
            "report": self.report.to_dict(),
            "sids": [str(sid) for sid in self.capabilities.index],
            "refreshed": self.refreshed,
            "buying_power": self.buying_power,
            "required_buying_power": self.required_buying_power,
            "account": None if self.account is None else dataclasses.asdict(self.account),
        }


def _capability_settings(execution_profile: Any) -> tuple[float, bool, str]:
    """``(max_age_hours, revalidate_at_submission, source)`` from ``[capabilities]``."""
    if execution_profile is None:
        block: Mapping[str, Any] = {}
    elif isinstance(execution_profile, Mapping):
        block = dict(execution_profile.get("capabilities") or {})
    else:
        value = getattr(execution_profile, "capabilities", None)
        block = dict(value) if isinstance(value, Mapping) else {}
    return (
        float(block.get("max_age_hours", 0.0)),
        bool(block.get("revalidate_at_submission", True)),
        str(block.get("source", "broker")),
    )


def _required_buying_power(intents: Sequence[Any], positions: Mapping[str, float]) -> float:
    """Dollars of buying power the batch consumes at its own reference prices.

    Buys consume it; so does the part of a sale that opens or extends a short. A sale
    that closes a long does not, and its proceeds are not credited either: the cash is
    not available until the fill prints, and crediting it here would authorize a batch
    on money the account does not yet have.
    """
    required = 0.0
    for intent in intents:
        sid = str(getattr(intent, "sid", None) or getattr(intent, "symbol"))
        qty = float(getattr(intent, "qty"))
        reference = float(getattr(intent, "reference_price"))
        if qty > 0.0:
            required += qty * reference
            continue
        held = float(positions.get(sid, 0.0))
        opening_short = max(0.0, -qty - max(0.0, held))
        required += opening_short * reference
    return required


def revalidate_for_submission(
    broker: Broker,
    intents: Sequence[Any],
    *,
    execution_profile: Any,
    ctx: LimitContext,
    prices: pd.Series,
    limits: Iterable[Limit] = (),
    positions: Mapping[str, float] | None = None,
    cash: float | None = None,
    committed_cash: float = 0.0,
    capabilities: pd.DataFrame | None = None,
    as_of: str,
    now: str | pd.Timestamp | None = None,
    expect_account_id: str | None = None,
    settings: ValidationSettings | None = None,
    cost_model: Any = None,
    modeled_cost_dollars: float | None = None,
    hedge_sids: Iterable[str] = (),
    exempt_sids: Iterable[str] = (),
) -> SubmissionCheck:
    """Revalidate capabilities and buying power immediately before submission (§3.9).

    ``capabilities`` may carry the rows the batch was constructed against; they are
    reused only when the profile permits it *and* every one of them is inside
    ``max_age_hours`` at ``now``. Otherwise the rows are pulled again from the broker,
    which is the profile's ``revalidate_at_submission = true`` case and the default.

    Every failure is a finding on the returned report, never an exception, with one
    exception of its own: an input the gate cannot evaluate at all (no prices, an
    unusable profile) raises, because a gate that cannot run has not passed.
    """
    held = {str(sid): float(qty) for sid, qty in dict(positions or {}).items()}
    max_age_hours, revalidate, source = _capability_settings(execution_profile)
    reference_now = pd.Timestamp(now) if now is not None else pd.Timestamp(as_of)
    sids = [str(getattr(intent, "sid", None) or getattr(intent, "symbol")) for intent in intents]
    # Two intents for one sid is one capability row, not two: the table is keyed
    # (sid, valid_at), and pulling the same name twice would build a frame the
    # as-of selection rule has to arbitrate.
    unique_sids = list(dict.fromkeys(sids))

    findings: list[Finding] = []

    # -- capabilities -------------------------------------------------------
    rows = capabilities
    refreshed = False
    if rows is not None and revalidate:
        # `[capabilities] revalidate_at_submission = true`: the rows the batch was
        # constructed against are not the rows it is submitted against, however fresh
        # they look. This is the profile's declaration, and it is not negotiable here.
        rows = None
    elif rows is not None:
        stale = stale_capability_sids(rows, now=reference_now, max_age_hours=max_age_hours)
        if stale:
            logger.info(
                "capability rows for %s are older than %.3fh; refreshing before submission",
                sorted(stale),
                max_age_hours,
            )
            rows = None
    if rows is None:
        rows = refresh_capabilities(broker, unique_sids, as_of=as_of)
        refreshed = True
    missing = [sid for sid in sids if sid not in set(rows.index)]
    if missing:
        findings.append(
            Finding(
                check="capabilities",
                detail=(
                    f"no dated capabilities row at {as_of} for {sorted(set(missing))} from source {source!r}; "
                    "an order for a name whose capabilities are unknown is not authorized"
                ),
                fatal=True,
            )
        )
    still_stale = stale_capability_sids(rows, now=reference_now, max_age_hours=max_age_hours)
    if still_stale:
        findings.append(
            Finding(
                check="capabilities_freshness",
                detail=(
                    f"capability rows for {sorted(still_stale)} are older than the declared "
                    f"max_age_hours {max_age_hours}"
                ),
                fatal=True,
            )
        )

    # -- account identity ---------------------------------------------------
    account: AccountIdentity | None = None
    if expect_account_id is not None:
        account = broker.account_identity()
        if account.account_id != str(expect_account_id):
            findings.append(
                Finding(
                    check="account_identity",
                    detail=(
                        f"broker at {account.endpoint} reports account {account.account_id!r}, not the "
                        f"expected {str(expect_account_id)!r}; a batch is never submitted to an account "
                        "it was not decided for"
                    ),
                    fatal=True,
                )
            )

    # -- buying power -------------------------------------------------------
    available = float(broker.buying_power())
    required = _required_buying_power(intents, held)
    resolved_settings = settings or ValidationSettings.from_profile(execution_profile, require_capabilities=True)
    if required > available + resolved_settings.cash_tolerance:
        findings.append(
            Finding(
                check="buying_power",
                detail=(
                    f"the batch consumes {required:.2f} of buying power against the venue's reported "
                    f"{available:.2f} (tolerance {resolved_settings.cash_tolerance:.2f})"
                ),
                fatal=True,
            )
        )

    report = validate_batch(
        list(intents),
        limits=limits,
        ctx=ctx,
        prices=prices,
        settings=resolved_settings,
        execution_profile=execution_profile,
        capabilities=rows,
        cash=float(cash) if cash is not None else float(broker.cash()),
        committed_cash=float(committed_cash),
        positions=held,
        cost_model=cost_model,
        modeled_cost_dollars=modeled_cost_dollars,
        hedge_sids=hedge_sids,
        exempt_sids=exempt_sids,
        attributed_to="submission",
    )
    if findings:
        binding = report.binding_condition or next(
            (finding.check for finding in findings if finding.fatal), report.binding_condition
        )
        report = dataclasses.replace(report, findings=tuple(findings) + report.findings, binding_condition=binding)
    return SubmissionCheck(
        report=report,
        capabilities=rows,
        refreshed=refreshed,
        buying_power=available,
        required_buying_power=required,
        account=account,
    )
