"""Read-only account margin report: equity, both sides, margin requirements and the cushion.

One ``GET /v2/account`` through :meth:`prism.live.alpaca.AlpacaBroker.account_margin`.
No order is built, sized or sent, and no other venue endpoint is called. The
report exists so the owner can check against a real account the broker facts
every cash-holding and leverage estimate depends on: what the account's margin
requirement is with shorts and a T-bill ETF held, how far equity is above the
maintenance requirement, and, read beside the account statement, whether short
proceeds offset a margin debit.

    python -m prism.scripts.margin_report                                  # print
    python -m prism.scripts.margin_report --json                           # the row as JSON
    python -m prism.scripts.margin_report --run-dir runs/x --record        # + append {run-dir}/margin.jsonl

``--record`` appends one row to ``{run-dir}/margin.jsonl``
(:mod:`prism.live.margin`) under the run directory's writer lock, so it never
interleaves with a loop session; it refuses a directory that does not exist
rather than creating one. ``prism-doctor`` reads the newest row and warns when
the cushion ``(equity - maintenance margin) / equity`` is below 25% or the read
is stale. Nothing schedules this report: the nightly wrapper does not run it,
so a daily record needs the owner to schedule it (docs/operations.md).

Exit status 0 when the account was read, 1 when the read or the record failed.
Credentials come from the environment or ``.env`` (``APCA_API_KEY_ID``,
``APCA_API_SECRET_KEY``) and are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from prism.workspace import load_env


def _money(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):,.2f}"


def _share(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def format_report(row: Mapping[str, Any]) -> str:
    """The human-readable report of one margin row."""
    cushion = row.get("cushion")
    floor = float(row["cushion_floor"])
    if cushion is None:
        verdict = "equity is not positive: no cushion (the account is in deficit)"
    elif float(cushion) < floor:
        verdict = f"BELOW the {floor:.0%} warning floor"
    else:
        verdict = f"above the {floor:.0%} warning floor"
    gross = row["gross_over_equity"]
    table = [
        ("equity", _money(row["equity"])),
        ("cash", _money(row["cash"])),
        ("long market value", f"{_money(row['long_market_value'])}  ({_share(row['long_over_equity'])} of equity)"),
        ("short market value", f"{_money(row['short_market_value'])}  ({_share(row['short_over_equity'])} of equity)"),
        ("gross / equity", "n/a" if gross is None else f"{float(gross):.3f}"),
        ("initial margin (Reg T)", f"{_money(row['initial_margin'])}  excess {_money(row['initial_margin_excess'])}"),
        (
            "maintenance margin",
            f"{_money(row['maintenance_margin'])}  excess {_money(row['maintenance_margin_excess'])}",
        ),
        ("cushion (equity - maint)/eq", f"{_share(cushion)}  {verdict}"),
        ("Reg T buying power", _money(row["regt_buying_power"])),
        ("buying power", f"{_money(row['buying_power'])}  multiplier {float(row['multiplier']):g}"),
    ]
    lines = [
        f"account margin ({row['source']} {row['mode']}, read {row['read_at']}, New York date {row['as_of']}; "
        "read-only, no orders)",
        *(f"  {label:<29} {value}" for label, value in table),
    ]
    optional = (
        ("last equity", "last_equity"),
        ("last maintenance margin", "last_maintenance_margin"),
        ("non-marginable buying power", "non_marginable_buying_power"),
        ("SMA", "sma"),
    )
    for label, key in optional:
        if row.get(key) is not None:
            lines.append(f"  {label:<29} {_money(row[key])}")
    if row.get("balance_asof") is not None:
        lines.append(f"  previous-session snapshot as of {row['balance_asof']}")
    flags = []
    if row.get("status") is not None:
        flags.append(f"status {row['status']}")
    if row.get("shorting_enabled") is not None:
        flags.append(f"shorting {'enabled' if row['shorting_enabled'] else 'DISABLED'}")
    if flags:
        lines.append("  " + ", ".join(flags))
    return "\n".join(lines)


def _default_broker(mode: str) -> Any:
    from prism.live.alpaca import AlpacaBroker

    return AlpacaBroker.from_env(mode=mode)


def main(argv: list[str] | None = None, *, broker_factory: Callable[[str], Any] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read the Alpaca account's margin position (read-only; no orders).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory whose margin.jsonl --record appends to. It must exist; it is never created.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Append the read to {run-dir}/margin.jsonl under the run directory's writer lock.",
    )
    parser.add_argument(
        "--account",
        choices=("paper", "live"),
        default="paper",
        help="The account mode the endpoint is checked against (default paper). Reading is the only action.",
    )
    parser.add_argument("--json", action="store_true", help="Print the row as JSON instead of the table.")
    args = parser.parse_args(argv)
    if args.record and args.run_dir is None:
        parser.error("--record needs --run-dir: the record lives in the run directory")
    # Refused before any venue read: a record with nowhere to go is not a read worth making.
    if args.run_dir is not None and not args.run_dir.is_dir():
        print(f"margin-report: {args.run_dir} is not a directory; nothing was read", file=sys.stderr)
        return 1

    from prism.live.margin import MARGIN_LEDGER_FILENAME, append_margin_row, margin_row

    load_env()
    factory = broker_factory or _default_broker
    try:
        broker = factory(args.account)
        margin = broker.account_margin()
    except Exception as exc:  # noqa: BLE001 — every layer (env, endpoint, transport, payload) ends the report
        print(f"margin-report: account read failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    row = margin_row(margin, read_at=datetime.now(timezone.utc), mode=args.account, source="alpaca")
    print(json.dumps(row, sort_keys=True) if args.json else format_report(row))

    if args.record:
        from prism.live.lockfile import run_dir_lock

        path = args.run_dir / MARGIN_LEDGER_FILENAME
        try:
            with run_dir_lock(args.run_dir):
                written = append_margin_row(path, row)
        except (OSError, RuntimeError) as exc:
            print(f"margin-report: not recorded in {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(
            (
                f"recorded in {path}"
                if written
                else f"not recorded: {path} already holds a read at or after {row['read_at']}"
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
