"""``--no-submit``: rehearse the whole submitting path without submitting (SPEC §7.7).

The paper entry points had no way to be *exercised*. Every invocation that
found credentials ran a complete cycle and sent orders, so "does this command
still parse and reach its credential check?" and "trade the account" were the
same act — on 2026-09-09 a verification step meaning the former performed the
latter, and 116 market-on-open orders reached the paper venue from a throwaway
run directory. This module is the missing third state: run everything, send
nothing, and say so loudly.

Two pieces, because a rehearsal has two ways to touch the world:

* :class:`NoSubmitBroker` wraps the real adapter. Every *read* passes straight
  through to the venue — account, positions, cash, open orders, fills — so the
  book is marked, scored and constructed against the same truth a trading run
  would see, and every mapping the submitting path exercises is exercised.
  ``submit`` records the order and logs it; nothing leaves the process. Every
  other venue *write* (``cancel``) is refused loudly rather than silently
  swallowed: a rehearsal that quietly cancelled a live order would be the same
  class of defect in the other direction.

* :func:`mirror_run_dir` redirects the run directory's durable writes to a
  scratch copy. This is not tidiness. The write-ahead protocol persists the
  decided orders to ``state.json`` *before* submitting them
  (``prism.live.loop.decide_and_submit``), and a resumed pending set is
  submitted without being re-decided — so a rehearsal that wrote to the real
  run directory would leave a decision armed for the next real session to fire
  at stale prices. A rehearsal must not be able to arm a trade. The mirror is
  seeded from the real directory so settle, reconcile, the cadence anchor and
  the drawdown rail all read exactly what the real session reads; the copy is
  discarded when the session ends.

The rehearsal still takes the real run directory's writer lock. Holding it is
what makes the mirror a coherent snapshot rather than a copy of a directory
another process is mid-write on.
"""

from __future__ import annotations

import inspect
import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from prism.live.broker import (
    AccountIdentity,
    Broker,
    CapabilityRow,
    Fill,
    FillRecord,
    Order,
    OrderStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NO_SUBMIT",
    "WOULD_SUBMIT",
    "NoSubmitBroker",
    "NoSubmitSession",
    "VenueWriteRefused",
    "mirror_run_dir",
    "no_submit_session",
]

#: The token every rehearsal line carries. One grep answers "did this run trade?".
NO_SUBMIT = "NO-SUBMIT"

#: The per-order rehearsal line's prefix.
WOULD_SUBMIT = "WOULD SUBMIT"

#: Endings the mirror never copies: the writer lock belongs to this process
#: (``prism.live.lockfile.LOCK_FILENAME`` is the dotfile ``.lock``, which has no
#: suffix, hence a name match rather than a suffix one), and a ``.tmp`` file is a
#: half-written document rather than state.
_UNMIRRORED_ENDINGS = (".lock", ".tmp")


class VenueWriteRefused(RuntimeError):
    """A venue write other than ``submit`` was attempted under ``--no-submit``."""


class NoSubmitBroker(Broker):
    """Read-through broker double: real reads, recorded orders, no submission.

    Reads delegate to ``inner`` — the point of ``--no-submit`` is to exercise
    the real path, and a rehearsal against a fake account proves nothing about
    the real one. ``submit`` is the only method whose behaviour differs, and it
    differs completely: the order is logged and recorded, and ``inner.submit``
    is never called.

    ``submits_to_venue`` is ``False``, which is what keeps the submission
    announcement (``prism.live.loop``) truthful: the line that says a run traded
    is printed only by an adapter that actually transmits.
    """

    #: This adapter transmits nothing (``prism.live.broker.transmits_to_venue``).
    submits_to_venue: bool = False

    def __init__(self, inner: Broker) -> None:
        self._inner = inner
        self._recorded: list[Order] = []

    # ------------------------------------------------------------------- record

    @property
    def inner(self) -> Broker:
        """The real adapter every read passes through to."""
        return self._inner

    @property
    def recorded(self) -> tuple[Order, ...]:
        """Every order this rehearsal would have submitted, in offer order."""
        return tuple(self._recorded)

    # ------------------------------------------------------------------- writes

    def submit(self, order: Order) -> None:
        """Record and log the order. Nothing is sent to the venue.

        Returns normally, so the loop's write-ahead protocol runs its real
        course (the persisted set advances in the mirrored run directory) and
        every order in the batch is reported, rather than the first one
        aborting the rehearsal.
        """
        self._recorded.append(order)
        logger.info(
            "%s %s #%d %s %s%s shares @ ref %.4f (notional %.2f) id=%s bar=%s — NOT SENT",
            NO_SUBMIT,
            WOULD_SUBMIT,
            len(self._recorded),
            order.symbol,
            "+" if order.qty > 0 else "",
            f"{order.qty:g}",
            order.reference_price,
            abs(order.qty) * order.reference_price,
            order.client_order_id,
            order.decision_bar,
        )

    def cancel(self, client_order_id: str) -> None:
        """Refuse: a rehearsal may not cancel a real order either."""
        raise VenueWriteRefused(
            f"--no-submit refuses to cancel {client_order_id!r}: a rehearsal reads the venue and "
            "writes nothing to it. Re-run without --no-submit to act on the account"
        )

    # -------------------------------------------------------------------- reads

    def positions(self) -> dict[str, float]:
        return self._inner.positions()

    def cash(self) -> float:
        return self._inner.cash()

    def submitted_order_ids(self, since: str | None = None) -> set[str]:
        """Delegate, keeping the duck-typed ``since`` narrowing when the inner has it.

        ``prism.live.loop._known_order_ids`` inspects this signature to decide
        whether it may bound the venue's order-history scan to the pending
        decision bar. Forwarding the parameter keeps the rehearsal's venue reads
        identical to a real session's instead of quietly paging more history.
        """
        submitted = self._inner.submitted_order_ids
        try:
            supports_since = "since" in inspect.signature(submitted).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
            supports_since = False
        if since is not None and supports_since:
            return submitted(since=since)
        return submitted()

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        return self._inner.fills_for(client_order_ids)

    def account_identity(self) -> AccountIdentity:
        return self._inner.account_identity()

    def capabilities(self, sids: Sequence[str], as_of: str) -> list[CapabilityRow]:
        return self._inner.capabilities(sids, as_of)

    def buying_power(self) -> float:
        return self._inner.buying_power()

    def open_orders(self) -> list[OrderStatus]:
        return self._inner.open_orders()

    def fills_since(self, marker: str | None = None) -> tuple[list[FillRecord], str | None]:
        return self._inner.fills_since(marker)

    def server_time(self) -> str:
        return self._inner.server_time()

    def __getattr__(self, name: str) -> Any:
        """Delegate the duck-typed extras (``endpoint``, ``open_order_ids``, ...).

        Only names the wrapper does not define itself reach here, so no venue
        write can slip through: ``submit`` and ``cancel`` are defined above, and
        every other mutating call on the adapters in scope goes through one of
        them.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        inner = self.__dict__.get("_inner")
        if inner is None:  # pragma: no cover - only during a partial __init__
            raise AttributeError(name)
        return getattr(inner, name)


def mirror_run_dir(run_dir: Path | str, destination: Path | str) -> Path:
    """Copy ``run_dir``'s durable files into ``destination`` and return it.

    A flat, one-level copy: run directories hold ledgers, ``state.json``, the
    kill switch and the per-bar spin-off cache side by side, and copying all of
    them is what makes the rehearsal decide what the real session would decide.
    The writer lock and any ``.tmp`` remnant are skipped — the lock is this
    process's and a torn temp file is not state. A ``run_dir`` that does not
    exist yet mirrors to an empty directory, which is exactly the state a first
    real session would start from.
    """
    source = Path(run_dir)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        return target
    copied = 0
    for entry in sorted(source.iterdir()):
        if not entry.is_file() or entry.name.endswith(_UNMIRRORED_ENDINGS):
            continue
        shutil.copy2(entry, target / entry.name)
        copied += 1
    logger.info("%s: mirrored %d file(s) from %s into a scratch copy", NO_SUBMIT, copied, source)
    return target


class NoSubmitSession:
    """One rehearsal: where its writes go, and what it would have submitted.

    ``work_dir`` is the directory the caller must use for every durable write
    (state file, ledgers, caches). Under a rehearsal it is the scratch mirror;
    with the rehearsal off it is the real run directory unchanged, so a single
    code path serves both and the trading path cannot drift from the rehearsed
    one.
    """

    def __init__(self, run_dir: Path, work_dir: Path, *, enabled: bool) -> None:
        self.run_dir = run_dir
        self.work_dir = work_dir
        self.enabled = enabled
        self._broker: NoSubmitBroker | None = None

    def wrap(self, broker: Broker) -> Broker:
        """The broker the session must trade through: the real one, or the double."""
        if not self.enabled:
            return broker
        self._broker = NoSubmitBroker(broker)
        return self._broker

    @property
    def recorded(self) -> tuple[Order, ...]:
        """Every order the rehearsal would have submitted."""
        return () if self._broker is None else self._broker.recorded

    def report(self) -> None:
        """The closing statement: how many orders, and that none of them was sent."""
        if not self.enabled:
            return
        rule = "=" * 78
        orders = self.recorded
        logger.warning("%s", rule)
        if orders:
            notional = sum(abs(o.qty) * o.reference_price for o in orders)
            logger.warning(
                "%s: %d ORDERS WOULD HAVE BEEN SUBMITTED (%.2f notional). NOTHING WAS SENT TO THE VENUE.",
                NO_SUBMIT,
                len(orders),
                notional,
            )
        else:
            logger.warning(
                "%s: THE CYCLE DECIDED NO ORDERS. NOTHING WAS SENT TO THE VENUE.",
                NO_SUBMIT,
            )
        logger.warning(
            "%s: no order was submitted, cancelled or modified, and the durable record of %s is "
            "unchanged — every write went to a scratch copy that is now discarded.",
            NO_SUBMIT,
            self.run_dir,
        )
        logger.warning("%s", rule)


@contextmanager
def no_submit_session(run_dir: Path | str, *, enabled: bool) -> Iterator[NoSubmitSession]:
    """Open a rehearsal over ``run_dir``, or a pass-through when ``enabled`` is false.

    Enter it *inside* the run directory's writer lock: the mirror is taken here,
    and a snapshot of a directory another writer holds would be torn.
    """
    directory = Path(run_dir)
    if not enabled:
        yield NoSubmitSession(directory, directory, enabled=False)
        return
    logger.warning(
        "%s: --no-submit is set. This run resolves credentials, reads the venue and constructs the "
        "book exactly as a trading run does, then SUBMITS NOTHING. Every order it would have sent is "
        "logged below as '%s'.",
        NO_SUBMIT,
        WOULD_SUBMIT,
    )
    with tempfile.TemporaryDirectory(prefix="prism-no-submit-") as scratch:
        session = NoSubmitSession(directory, mirror_run_dir(directory, Path(scratch) / directory.name), enabled=True)
        try:
            yield session
        finally:
            session.report()
