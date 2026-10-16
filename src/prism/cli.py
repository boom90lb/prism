"""The `prism` command surface (spec §3.14).

One console script, one vocabulary: ``init``, ``data``, ``conform``,
``evaluate``, ``compare``, ``run``, ``reconcile``, ``replay``, ``check``,
``trials``, ``folds``, ``upgrade-diff``. Every command the specification
declares is parsed here from the first release, because a command that exits
saying which stage delivers it is a schedule an operator can read, while an
``invalid choice`` traceback is a surprise.

Three duties, in this order, before any command writes (§3.1): resolve the
workspace and create it, load ``.env``, then configure logging under
``workspace.logs``. Importing this module does none of them — the engine's
import purity invariant — and neither does ``prism init``, which writes a new
project directory and must not scatter a workspace across whatever directory
the operator happened to be standing in.

Exit codes: 0 success, 1 a refusal or a failure with a named cause, 2 a command
whose delivery stage has not landed (and argparse's own usage errors). A report
that could not cover everything it was asked about also exits 2, under the
`prism.live.parity_gate` convention: 1 something failed, 2 something could not
be evaluated and nothing failed, 0 everything asked about was answered.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from prism import __version__

if TYPE_CHECKING:  # pragma: no cover - imports the data layer only for the checker
    from prism.io.dataset import Dataset, DatasetView
    from prism.live.daily import DailyBookConfig
    from prism.spec import LoadedComponent, Specification
    from prism.workspace import Workspace

logger = logging.getLogger("prism.cli")

#: Exit codes. ``EXIT_LATER_STAGE`` is also argparse's usage-error code: both
#: mean "this invocation did not run", and neither is a failed run.
EXIT_LATER_STAGE = 2
EXIT_REFUSED = 1


class CommandError(RuntimeError):
    """A command refused with a cause worth printing without a traceback."""


class LaterStage(RuntimeError):
    """A declared command whose delivery stage has not landed.

    Carries the stage so the message always names it: a user reading the
    refusal learns when the capability arrives, not merely that it is absent.
    """

    def __init__(self, what: str, stage: int, delivers: str) -> None:
        super().__init__(f"{what} is not delivered until stage {stage}: {delivers}")
        self.what = what
        self.stage = stage


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The whole command surface of §3.14, deferred subcommands included."""
    parser = argparse.ArgumentParser(
        prog="prism",
        description="Prism — a cross-sectional systematic trading engine.",
    )
    parser.add_argument("--version", action="version", version=f"prism-trading {__version__}")
    parser.add_argument(
        "--workspace",
        default=None,
        help="the writable root (default: PRISM_WORKSPACE, [tool.prism].workspace, then the current directory)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG on the console")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    init = sub.add_parser("init", help="write a private-project template")
    init.add_argument("directory", help="the project directory to create")
    init.add_argument(
        "--force",
        action="store_true",
        help="overwrite the template's own files in an existing project directory",
    )

    data = sub.add_parser("data", help="datasets: fetch, ingest, synthetic fixtures, manifest, universe")
    data_sub = data.add_subparsers(dest="subcommand", metavar="<subcommand>")

    synthetic = data_sub.add_parser("synthetic", help="write the deterministic synthetic fixture")
    synthetic.add_argument("--seed", type=int, required=True, help="fixes the generator")
    synthetic.add_argument(
        "--scenarios",
        default="all",
        help="'all' or a comma-separated subset of the twelve scenario ids of §3.2",
    )
    synthetic.add_argument("--equities", type=int, default=100, help="number of synthetic equities")
    synthetic.add_argument("--etfs", type=int, default=10, help="number of synthetic ETFs")
    synthetic.add_argument("--vintage", default="synthetic_v1", help="the vintage the manifest names")

    fetch = data_sub.add_parser("fetch", help="vendor download into the tables of §3.2")
    fetch.add_argument("--source", required=True)
    fetch.add_argument("--symbols", required=True)
    fetch.add_argument("--start")
    fetch.add_argument("--end")

    ingest = data_sub.add_parser("ingest", help="local files under a declared column mapping")
    ingest.add_argument("--format", choices=("parquet", "csv"), required=True)
    ingest.add_argument("--mapping", required=True, help="the §3.2 mapping file: columns, lag, revision policy")
    ingest.add_argument("--vintage", default="local_v1", help="the vintage label the manifest carries")
    ingest.add_argument(
        "--symbol-from-filename",
        action="store_true",
        help="read the vendor symbol from the leading {SYMBOL}_{interval} of each file name",
    )
    ingest.add_argument("path", help="a file or a directory of files in the declared format")

    legacy = data_sub.add_parser("import-legacy", help="existing per-symbol caches and membership Parquet")
    legacy.add_argument("--cache-dir", help="where the caches live (default: the workspace data directory)")
    legacy.add_argument("--out", help="where the tables are written (default: the workspace data directory)")
    legacy.add_argument("--vintage", default="legacy_v1", help="the vintage label the manifest carries")
    legacy.add_argument("--index", default="sp500", help="the index id the membership files describe")
    legacy.add_argument("--interval", default="1d", help="the bar interval to import")
    legacy.add_argument("--symbols", help="a file of symbols, one per line, restricting the import")

    manifest = data_sub.add_parser("manifest", help="write manifest.json")
    manifest.add_argument("--freeze", action="store_true", help="name a vintage")

    data_sub.add_parser("status", help="coverage, provisional sids, last ingestion per source")

    universe = data_sub.add_parser("universe", help="rebuild the point-in-time membership table")
    universe.add_argument("--index", required=True)
    universe.add_argument("--asof")

    conform = sub.add_parser("conform", help="the component and dataset conformance suite of §3.3")
    conform.add_argument("spec", help="the specification to conform")

    evaluate = sub.add_parser("evaluate", help="walk-forward evaluation of a specification")
    evaluate.add_argument("spec")
    evaluate.add_argument("--registration")
    evaluate.add_argument("--arm")
    evaluate.add_argument("--capital", type=float, default=1.0, help="opening cash; recorded (default 1.0)")
    evaluate.add_argument("--engine", choices=("historical_constant_weight", "canonical"), default=None)
    evaluate.add_argument("--visibility", choices=("replay_recorded", "research_vintage"), default=None)
    evaluate.add_argument("--vintage")
    evaluate.add_argument("--freeze-scores", dest="freeze_scores")
    evaluate.add_argument("--smoke", action="store_true", help="the fixed engineering run on the synthetic fixture")

    compare = sub.add_parser("compare", help="the comparison report of §3.12")
    compare.add_argument("records", nargs="*")
    compare.add_argument("--registration")
    compare.add_argument("--repricing")
    compare.add_argument("--cost-multiplier", dest="cost_multiplier", type=float)
    compare.add_argument("--promotion-check", dest="promotion_check")
    compare.add_argument("--edge-bracketing", dest="edge_bracketing")

    trials = sub.add_parser("trials", help="evidence history")
    trials_sub = trials.add_subparsers(dest="subcommand", metavar="<subcommand>")
    trials_reconcile = trials_sub.add_parser("reconcile", help="close orphaned started rows")
    trials_reconcile.add_argument("--log")
    trials_reconcile.add_argument("--source")

    folds = sub.add_parser("folds", help="fold tables")
    folds_sub = folds.add_subparsers(dest="subcommand", metavar="<subcommand>")
    folds_import = folds_sub.add_parser("import", help="write a fold table from recorded fold windows")
    folds_import.add_argument("--from", dest="from_path", required=True)
    folds_import.add_argument("--out", required=True)

    run = sub.add_parser("run", help="one operating session")
    run.add_argument("--spec", required=True)
    run.add_argument("--account", choices=("paper", "live"), required=True)
    run.add_argument("--broker", choices=("synthetic", "alpaca"), default="synthetic")
    run.add_argument("--once", action="store_true", help="one session, then exit")
    run.add_argument("--stage", choices=("decide", "submit", "sweep"), default=None)
    run.add_argument("--break-lock", dest="break_lock", action="store_true")
    run.add_argument(
        "--reason",
        default=None,
        help="why the account lock is being broken; required with --break-lock, because §3.10 records "
        "who broke a lock, when, and why in the run directory, and a break with no stated reason "
        "records no why",
    )

    reconcile = sub.add_parser("reconcile", help="reconcile a run against the venue (§3.11)")
    reconcile.add_argument("run", help="the run directory")

    replay = sub.add_parser("replay", help="replay recorded sessions")
    replay.add_argument("run", help="the recorded session capture: a directory of ledgers with a manifest.json")
    replay.add_argument(
        "--upgrade-diff",
        dest="upgrade_diff",
        action="store_true",
        help="report what adopting one engine version in place of another would change over the sessions this "
        "capture records; the same report as `prism upgrade-diff --sessions <run>`",
    )
    _add_upgrade_diff_options(replay)

    check = sub.add_parser("check", help="health, authorization, freshness and missed-run checks")
    check.add_argument("run", help="the run directory whose durable record is read")
    check.add_argument(
        "--expect-session",
        dest="expect_session",
        action="store_true",
        help="assert a session should have been decided by now: the §3.11 freshness table is printed and a "
        "missed session, or a run directory that has decided nothing at all, is a failure rather than a note",
    )
    check.add_argument("--monitor", action="store_true", help="run the telemetry read of §3.11 instead")
    check.add_argument(
        "--asof",
        default=None,
        help="the decision clock to evaluate against (default: today). A replayed session is evaluated "
        "against its recorded decision bar, never against wall-clock time (§3.11)",
    )
    check.add_argument(
        "--account",
        choices=("paper", "live"),
        default="paper",
        help="the mode the endpoint and authorization checks are evaluated for (§3.10); `check` writes "
        "nothing and reaches no venue, so unlike `run --account` it carries a default",
    )
    check.add_argument(
        "--universe-file",
        dest="universe_file",
        default=None,
        help="the universe file the loop trades, when the run directory's checks should include it",
    )

    upgrade = sub.add_parser(
        "upgrade-diff",
        help="diff what two engine versions decided over one set of recorded sessions",
    )
    upgrade.add_argument(
        "--sessions",
        required=True,
        help="the session capture whose manifest defines the sessions under review",
    )
    _add_upgrade_diff_options(upgrade)

    return parser


def _add_upgrade_diff_options(parser: argparse.ArgumentParser) -> None:
    """The optional half of the `upgrade-diff` surface, shared with `replay`.

    Section 3.14 sketched this command as `--spec/--session/--engine`, that is,
    as one process driving another installed version in place. What stage 5
    delivered is `prism.live.replay.upgrade_diff_report`, which compares two
    *captures* of what each version decided over the same recorded sessions.
    That is the honest shape: two engine versions cannot both be imported into
    one interpreter, so the second side is always a recording made elsewhere,
    and a flag named `--engine <version>` would promise an in-process upgrade
    the mechanism never performs.
    """
    parser.add_argument(
        "--baseline",
        default=None,
        help="the capture holding what the incumbent version decided (default: the sessions capture itself)",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="the capture holding what the upgraded version decided over the same sessions. Without it "
        "every session is reported as not compared, which is the ordinary state before the upgrade has "
        "been run over these sessions",
    )
    parser.add_argument(
        "--bar",
        dest="bars",
        action="append",
        default=None,
        help="restrict the review to this decision bar (YYYY-MM-DD); repeatable. Default: every session "
        "the baseline capture covers",
    )
    parser.add_argument(
        "--weight-tolerance",
        dest="weight_tolerance",
        type=float,
        default=1e-9,
        help="target weights closer than this are the same weight (default: 1e-9)",
    )
    parser.add_argument(
        "--share-tolerance",
        dest="share_tolerance",
        type=float,
        default=1e-9,
        help="order quantities closer than this are the same order (default: 1e-9)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="write the full report here as JSON, beside the summary printed to stdout",
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run one command. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command is None:
        parser.print_help()
        return EXIT_LATER_STAGE
    handler = _HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover — argparse rejects an unknown command first
        parser.error(f"unknown command {args.command!r}")
    try:
        return handler(args)
    except LaterStage as exc:
        print(f"prism {args.command}: {exc}", file=sys.stderr)
        return EXIT_LATER_STAGE
    except Exception as exc:
        # A declared engine refusal prints its cause and exits 1; anything else
        # is a defect and keeps its traceback.
        if not isinstance(exc, _refusal_types()):
            raise
        print(f"prism {args.command}: {exc}", file=sys.stderr)
        return EXIT_REFUSED


def _refusal_types() -> tuple[type[BaseException], ...]:
    """The engine exceptions a command reports as a refusal, not as a crash.

    Imported lazily so that ``prism --help`` and ``prism init`` do not pull the
    data layer into the process.
    """
    from prism.conformance import ConformanceError
    from prism.evaluate import EvaluationError
    from prism.io.dataset import DatasetError, StageNotAvailableError
    from prism.io.events import EventError
    from prism.io.legacy_import import LegacyImportError
    from prism.io.local_files import IngestError
    from prism.io.manifest import ManifestError
    from prism.io.security_master import SecurityMasterError
    from prism.io.synthetic import SyntheticError
    from prism.live.authorization import AuthorizationRefused
    from prism.live.lockfile import LockRefused
    from prism.live.loop import PartialBatchEscalation, UnresolvedSubmission
    from prism.spec import SpecError
    from prism.template import TemplateError

    return (
        AuthorizationRefused,
        CommandError,
        ConformanceError,
        DatasetError,
        EvaluationError,
        EventError,
        IngestError,
        LegacyImportError,
        LockRefused,
        ManifestError,
        PartialBatchEscalation,
        SecurityMasterError,
        SpecError,
        StageNotAvailableError,
        SyntheticError,
        TemplateError,
        UnresolvedSubmission,
    )


# --------------------------------------------------------------------------
# Commands that stage 1 delivers
# --------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    """Write the template of §4. No workspace is resolved and none is created:
    ``prism init`` writes one new directory and nothing beside it."""
    from prism.template import ENGINE_PIN, write_template

    root = Path(args.directory).expanduser()
    written = write_template(root, force=args.force)
    print(f"wrote {len(written)} files into {root} (pinned to {ENGINE_PIN})")
    print("next: cd into it, then `prism data synthetic --seed 7 --scenarios all`")
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    if args.subcommand is None:
        raise CommandError("`prism data` needs a subcommand; try `prism data synthetic --seed 7`")
    if args.subcommand == "ingest":
        return _cmd_data_ingest(args)
    if args.subcommand == "import-legacy":
        return _cmd_data_import_legacy(args)
    if args.subcommand != "synthetic":
        raise LaterStage(
            f"`prism data {args.subcommand}`",
            3,
            "the vendor fetch path, the manifest freeze, the status read and the universe rebuild arrive "
            "with the rest of the data contract"
            + (
                "; `prism-build-universe` rebuilds the point-in-time membership table today"
                if args.subcommand == "universe"
                else ""
            ),
        )
    from prism.io.synthetic import resolve_scenarios, write_synthetic_dataset

    workspace = _open_workspace(args, "data-synthetic")
    scenarios = resolve_scenarios(args.scenarios)
    manifest = write_synthetic_dataset(
        workspace.data,
        seed=args.seed,
        scenarios=scenarios,
        n_equities=args.equities,
        n_etfs=args.etfs,
        vintage=args.vintage,
    )
    print(f"wrote the synthetic fixture into {workspace.data}")
    print(f"  data_id  {manifest.id}")
    print(f"  vintage  {manifest.vintage}")
    print(f"  as_of    {manifest.as_of}")
    print(f"  scenarios {', '.join(scenarios)}")
    print(f"  files    {len(manifest.files)}")
    cases = manifest.coverage.get("scenarios", {})
    print(f"  cases    {len(cases)} addressable by scenario id in manifest coverage['scenarios']")
    for name in sorted(cases):
        case = cases[name]
        window = f"{case['decision_bar']} -> {case['fill_bar']}"
        print(f"    {name:<13} {window}  {', '.join(sorted(case['sids'].values()))}")
    return 0


def _cmd_data_ingest(args: argparse.Namespace) -> int:
    """``prism data ingest`` (§3.2, "Local ingestion"): local files under a declared mapping."""
    from prism.io.local_files import IngestMapping, ingest_local_files

    workspace = _open_workspace(args, "data-ingest")
    mapping = IngestMapping.read(Path(args.mapping))
    target = Path(args.path)
    if target.is_dir():
        paths = sorted(target.glob(f"*.{args.format}"))
        if not paths:
            raise CommandError(f"no *.{args.format} files under {target}")
    elif target.is_file():
        paths = [target]
    else:
        raise CommandError(f"{target} is neither a file nor a directory")
    report = ingest_local_files(
        workspace.data,
        paths,
        mapping=mapping,
        fmt=args.format,
        vintage=args.vintage,
        symbol_from_filename=args.symbol_from_filename,
    )
    print(f"ingested {len(report.files)} file(s) from source {report.source} into {report.root}")
    print(f"  data_id  {report.manifest.id}")
    print(f"  vintage  {report.manifest.vintage}")
    print(f"  as_of    {report.manifest.as_of}")
    print(f"  sids     {len(report.sids)} ({len(report.minted_sids)} minted)")
    print(f"  rows     {report.rows_written} written, {report.rows_revised} revised, {report.rows_deduped} identical")
    for flag in report.gap_flags:
        print(f"  gap flag {flag}")
    for entry in report.gap_scan_skipped:
        print(f"  no gap scan for {entry['vendor_symbol']}: {entry['reason']}")
    return 0


def _cmd_data_import_legacy(args: argparse.Namespace) -> int:
    """``prism data import-legacy`` (§3.2, "Legacy import")."""
    from prism.io.legacy_import import import_legacy, legacy_summary_json

    workspace = _open_workspace(args, "data-import-legacy")
    cache_dir = Path(args.cache_dir) if args.cache_dir else workspace.data
    out = Path(args.out) if args.out else workspace.data
    symbols = None
    if args.symbols:
        text = Path(args.symbols).read_text(encoding="utf-8")
        symbols = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    report = import_legacy(
        out,
        cache_dir=cache_dir,
        vintage=args.vintage,
        index_id=args.index,
        interval=args.interval,
        symbols=symbols,
    )
    logger.info("import-legacy\n%s", legacy_summary_json(report))
    print(f"imported {len(report.symbol_to_sid)} series into {report.root}")
    print(f"  data_id  {report.manifest.id}")
    print(f"  vintage  {report.manifest.vintage}")
    print(f"  as_of    {report.manifest.as_of}")
    print(f"  sids     {len(set(report.symbol_to_sid.values()))}, every one provisional at mint")
    print(f"  rows     {report.rows_written} written, {report.rows_revised} revised, {report.rows_deduped} identical")
    print(
        f"  events   {report.events_written} written, {len(report.events_outside_session_table)} groups out of range"
    )
    print(f"  gaps     {len(report.gap_flags)} flagged")
    for entry in report.skipped_series:
        print(f"  skipped {entry['vendor_symbol']}: {entry['reason']}")
    return 0


def _cmd_conform(args: argparse.Namespace) -> int:
    """The conformance suite of §3.3 over every component of one specification."""
    from prism.conformance import conform_specification
    from prism.spec import load_spec

    workspace = _open_workspace(args, "conform")
    spec = load_spec(Path(args.spec).expanduser(), workspace=workspace)
    report = conform_specification(spec, workspace=workspace)
    print(report.text())
    return 0 if report.ok else EXIT_REFUSED


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """``--smoke`` only at stage 1: the fixed engineering run on the synthetic fixture."""
    if not args.smoke:
        raise LaterStage(
            "`prism evaluate` without --smoke",
            2,
            "a recorded evaluation needs the ResultRecord, the run-recording protocol and the "
            "trial log, which arrive with the full evaluator",
        )
    if args.engine == "canonical":
        raise LaterStage(
            "`--engine canonical`",
            3,
            "the canonical accounting engine needs execution/portfolio_state.py, "
            "execution/accounting.py and execution/simulate.py",
        )
    if args.registration or args.arm:
        raise LaterStage(
            "`--registration` / `--arm`",
            2,
            "a registration binds a run to a declared selection set, which needs the trial log",
        )
    if args.visibility == "replay_recorded":
        raise LaterStage(
            "`--visibility replay_recorded`",
            3,
            "both visibility modes arrive with the data contract; stage 1 evaluates " "research_vintage only",
        )
    if args.vintage:
        raise LaterStage(
            "`--vintage`",
            3,
            "the vintage is read from the dataset manifest at stage 1; selecting among frozen "
            "vintages arrives with the two visibility modes",
        )
    if args.freeze_scores:
        raise LaterStage(
            "`--freeze-scores`",
            4,
            "the frozen score panel is the registration's shared input, and stage 4 opens with it",
        )
    from prism.evaluate import smoke_evaluate
    from prism.spec import load_spec

    workspace = _open_workspace(args, "evaluate")
    spec = load_spec(Path(args.spec).expanduser(), workspace=workspace)
    result = smoke_evaluate(
        spec,
        workspace=workspace,
        engine=args.engine or "historical_constant_weight",
        capital=args.capital,
    )
    print(result.text())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """One operating session (§3.14), or the operator act that precedes one.

    Three paths leave here. ``--break-lock`` is not a session at all: it breaks
    one account's execution lock, on the record, and exits.
    ``--broker alpaca`` is the authorized venue session of §3.10, where the
    authorization record governs before a client exists. ``--broker synthetic``
    is the credential-free replay session over the synthetic fixture, which
    holds the run-directory lock, runs the write-ahead decide protocol, and
    writes ``LoopState`` at the schema version the tree carries.
    """
    if args.break_lock:
        return _break_execution_lock(args)
    if args.reason is not None:
        raise CommandError("--reason belongs to --break-lock; a session records its own cause")
    if args.broker != "synthetic":
        return _venue_session(args)
    if args.account == "live":
        raise CommandError(
            "`--account live --broker synthetic` names no account: a live account is a venue's, and a "
            "synthetic session reaches no venue. Run a live account with `--broker alpaca`, which reads "
            "the authorization record of §3.10 before it constructs a client, or run the synthetic "
            "session with `--account paper`"
        )
    if args.stage is not None:
        raise LaterStage(
            "`--stage decide|submit|sweep` on a synthetic session",
            5,
            "the three separately scheduled stages exchange durable state, and the synthetic driver "
            "replays a whole cycle in one process",
        )
    if not args.once:
        raise LaterStage(
            "`prism run` without --once",
            5,
            "the resident loop is the operating workflow; stage 1 runs one session and exits",
        )

    from prism.io.dataset import Dataset
    from prism.live.lockfile import run_dir_lock
    from prism.live.replay import replay_daily_cycles
    from prism.signal.base import Signal  # noqa: F401 - the factory's return annotation
    from prism.spec import instantiate_component, load_spec

    workspace = _open_workspace(args, "run")
    spec = load_spec(Path(args.spec).expanduser(), workspace=workspace)
    if len(spec.components) != 1:
        raise LaterStage(
            f"a {len(spec.components)}-component specification in `prism run`",
            5,
            "an operating session over a composed book runs through assemble; stage 1 replays "
            "the single-component daily cycle of §4",
        )
    loaded = spec.components[0]
    if loaded.spec.grid_schedule is not None:
        raise CommandError(
            f"{spec.path} declares tranches or phase_offset_bars for component {loaded.component_id!r}, and "
            "`prism run` schedules the daily cycle by elapsed sessions only; running it would trade a schedule "
            "the specification does not declare. Evaluate the specification with `prism evaluate` instead"
        )
    if spec.sizing is not None:
        raise CommandError(
            f"{spec.path} declares [sizing] rule={spec.sizing.rule!r}, and `prism run` trades the unscaled book; "
            "running it would trade a size the specification does not declare. The rule is applied by the "
            "account_state evaluator only"
        )
    config = _daily_book_config(spec, loaded)

    dataset = _open_dataset(Dataset, workspace)
    decision_date = str(dataset.manifest.as_of)[:10]
    view = dataset.view("research_vintage", decision_date, vintage=dataset.manifest.vintage)
    sids = _universe(spec, view)
    close = view.panel(sids, decision_date, "close")
    open_ = view.panel(sids, decision_date, "open")
    volume = view.panel(sids, decision_date, "volume")
    if close.empty:
        raise CommandError(
            f"the dataset at {workspace.data} carries no bars for the {len(sids)} names of "
            f"{spec.strategy.universe_index or 'the declared universe'} through {decision_date}"
        )

    # A synthetic session declares the account id ``synthetic`` so it can never
    # contend with a real account's lock (§3.14).
    account_id = "synthetic"
    run_dir = workspace.run_dir(account_id)
    component_spec = loaded.spec

    def factory() -> "Signal":
        return instantiate_component(component_spec, workspace=workspace, where=str(spec.path)).component

    try:
        with run_dir_lock(run_dir):
            results, broker = replay_daily_cycles(
                factory,
                close,
                volume,
                open_,
                config,
                run_dir,
                start_bar=decision_date,
                end_bar=decision_date,
            )
    except RuntimeError as exc:
        raise CommandError(str(exc)) from exc
    except ValueError as exc:
        raise CommandError(str(exc)) from exc

    result = results[-1]
    targets = result.target_weights.dropna()
    print(f"session {result.decision_bar} on {len(close.columns)} names, account {account_id} (synthetic broker)")
    print(f"  book        {config.book} from to_weights={component_spec.to_weights}")
    print(f"  equity      {result.equity:,.2f}")
    print(f"  targets     {int((targets != 0.0).sum())} non-zero, gross {float(targets.abs().sum()):.4f}")
    print(f"  orders      {len(result.submitted_orders)} decided at the close, filled at the next open")
    print(f"  state       {run_dir / 'state.json'}")
    print(f"  cash        {broker.cash():,.2f}")
    return 0


def venue_broker(mode: str, *, time_in_force: str = "opg") -> Any:
    """The venue client ``--broker alpaca`` runs against — the construction seam.

    A module-level function, so the whole authorized session can be exercised
    offline against a broker double. The seam sits *above* the controls and
    never inside them: the authorization record is loaded before this is
    called, and the binding is verified against whatever client it returns, so
    nothing a caller substitutes here can weaken a refusal.

    The mode is an argument, never an inheritance (§3.10). ``--account
    paper|live`` selects it; ``APCA_API_BASE_URL`` may set an endpoint but never
    the mode, and :func:`prism.live.authorization.resolve_endpoint` checks the
    resolved endpoint against that mode's allowlist where the client is
    constructed — before any request exists to send.
    """
    from prism.live.alpaca import AlpacaBroker

    return AlpacaBroker.from_env(mode=mode, time_in_force=time_in_force)


def _venue_session(args: argparse.Namespace) -> int:
    """``--broker alpaca``: the authorized venue session of §3.10 and §3.14.

    Authorization comes first, and it is not conditional on the mode. The
    record for the named account is read from that account's run directory
    before a client is constructed, so a session with no record — state S4b,
    which permits nothing — makes no venue call and reads no credential. Only
    then is the client built, its account id read once, and the binding
    verified: mode, endpoint, account id, ``config_id``, engine version, build
    identity and execution host, each compared exactly. The account-keyed
    execution lock follows from the record inside every loop entry point
    (:func:`prism.live.loop.execution_lock_spec`), which is why nothing here
    takes it by hand.

    ``config_id`` comes from the specification this process resolved, never
    from the record: comparing the record with itself would check nothing. That
    is the mechanism behind "merging code never authorizes a live
    configuration" (§3.10).

    Two of the three stages of §3.14 run here. ``submit`` re-offers the durable
    pending-order record and reads nothing else; ``sweep`` completes it. The
    ``decide`` stage, and the whole cycle with no ``--stage``, need the book
    constructed over a live dataset view, which this command does not yet do —
    they refuse by name rather than approximately.
    """
    from prism.live.authorization import (
        AUTHORIZATION_FILENAME,
        AuthorizationGate,
        AuthorizationRefused,
        ProcessIdentity,
        load_authorization,
    )
    from prism.live.lockfile import run_dir_lock
    from prism.live.loop import (
        LiveLoopContext,
        UnresolvedSubmission,
        held_execution_lock,
        pending_order_recovery,
        submit_pending,
        sweep_pending,
    )
    from prism.live.state import StateStore
    from prism.spec import load_spec

    if not args.once:
        raise LaterStage(
            "`prism run` without --once",
            5,
            "the resident loop is the operating workflow; a stage runs one invocation and exits",
        )
    if args.stage not in ("submit", "sweep"):
        raise LaterStage(
            "`prism run --broker alpaca` without `--stage submit|sweep`" if args.stage is None else "`--stage decide`",
            5,
            "the decide stage constructs the book over a live dataset view through assemble; the submit "
            "and sweep stages, which read only the durable pending-order record, run today",
        )

    workspace = _open_workspace(args, "run")
    run_dir = workspace.run_dir(args.account)
    record = load_authorization(run_dir)
    if record is None:
        raise AuthorizationRefused(
            f"no authorization record for the {args.account} account at "
            f"{run_dir / AUTHORIZATION_FILENAME}: state S4b permits nothing and every broker write is "
            "refused, C0 included. Refusing before a client is constructed and before a credential is "
            "read (§3.10)"
        )
    spec = load_spec(Path(args.spec).expanduser(), workspace=workspace)
    broker = venue_broker(args.account, time_in_force="day" if args.stage == "sweep" else "opg")
    identity = ProcessIdentity.current(
        mode=args.account,
        endpoint=broker.endpoint,
        account_id=broker.account_identity().account_id,
        config_id=spec.config_id(),
    )
    gate = AuthorizationGate(record, identity, account=record.account_name)
    gate.verify_binding(observed_endpoint=broker.endpoint)
    broker.bind_authorization(gate)
    ctx = LiveLoopContext(
        store=StateStore(run_dir / "state.json"),
        broker=broker,
        fills_ledger=run_dir / "fills.jsonl",
        equity_ledger=run_dir / "equity.jsonl",
        unfilled_ledger=run_dir / "unfilled.jsonl",
        authorization=gate,
    )
    print(f"account {record.account_name} ({record.account_id}) at {record.endpoint}, mode {record.mode}")
    print(f"  record      {run_dir / AUTHORIZATION_FILENAME}")
    print(f"  scope       {gate.scope().to_dict()}")

    # Both locks are held, as §3.10 says they are: the run directory's, because
    # `state.json` and the ledgers are a second thing two writers can tear, and
    # the account-keyed one, which every loop entry point takes from the record.
    with run_dir_lock(run_dir):
        if args.stage == "sweep":
            swept = sweep_pending(ctx)
            print(f"  sweep       {len(swept)} residual order(s) submitted")
            return 0

        with held_execution_lock(ctx):
            state = ctx.store.load()
            open_orders = [] if state is None else state.open_pending_orders
            if state is None or not open_orders:
                print("  submit      nothing pending; the decide stage writes the record this stage offers")
                return 0
            recovery = pending_order_recovery(ctx, state)
            if not recovery.resolved:
                raise UnresolvedSubmission(recovery.refusal(), recovery)
            outcome = submit_pending(ctx, state, open_orders, gate=gate)
    print(f"  submit      {len(outcome.submitted)} of {len(open_orders)} order(s) offered")
    if outcome.rejected:
        for client_order_id, why in outcome.rejected:
            print(f"  rejected    {client_order_id}: {why}")
        print(f"  unoffered   {len(outcome.unsubmitted)} order(s); response {outcome.response}")
        return EXIT_REFUSED
    return 0


def _break_execution_lock(args: argparse.Namespace) -> int:
    """``--break-lock`` (§3.10): break one account's execution lock, on the record.

    A non-stale lock is never broken automatically. This is the manual
    override, and it is an override rather than a hazard only because it leaves
    a record: who broke it, when, and why are appended to
    ``<run_dir>/lock_breaks.jsonl`` **before** the break takes effect, together
    with the record being displaced. It breaks the lock and exits — an operator
    act is not a trading session, and running one immediately after the other
    in a single process would hide which of the two an operator asked for.
    """
    from prism.live import ledgers
    from prism.live.authorization import (
        AUTHORIZATION_FILENAME,
        AuthorizationRefused,
        load_authorization,
    )
    from prism.live.lockfile import account_lock_path, break_account_lock, utc_now_text

    if not args.reason or not args.reason.strip():
        raise CommandError(
            "`--break-lock` requires `--reason`: §3.10 records who broke a lock, when, and why, and a "
            "break with no stated reason records no why"
        )
    workspace = _open_workspace(args, "run")
    synthetic = args.broker == "synthetic"
    run_dir = workspace.run_dir("synthetic" if synthetic else args.account)
    if synthetic:
        # §3.14: a synthetic session declares the account id `synthetic`, so it
        # can never contend with a real account's lock — including here.
        account_id = "synthetic"
    else:
        record = load_authorization(run_dir)
        if record is None:
            raise AuthorizationRefused(
                f"no authorization record at {run_dir / AUTHORIZATION_FILENAME}: the record is what names "
                "the account whose lock this would break, so there is no lock to name (§3.10)"
            )
        account_id = record.account_id
    path = account_lock_path(account_id)
    ledger = run_dir / "lock_breaks.jsonl"

    def record_the_break(displaced: Any) -> None:
        # Written and fsynced before the lock changes: a crash between here and
        # the rename leaves the account of an attempted break, not a lock that
        # became free with no explanation.
        ledgers.append_rows(
            ledger,
            [
                {
                    "broken_at": utc_now_text(),
                    "broken_by": _who_is_running(),
                    "reason": args.reason.strip(),
                    "account_id": account_id,
                    "lock_path": str(path),
                    "displaced": None if displaced is None else displaced.to_mapping(),
                }
            ],
        )

    _, displaced = break_account_lock(
        account_id,
        reason=args.reason.strip(),
        broken_by=_who_is_running(),
        before_break=record_the_break,
    )
    print(f"broke the execution lock for account {account_id!r}")
    print(f"  lock        {path}")
    print(f"  displaced   {'an unreadable record' if displaced is None else displaced.describe()}")
    print(f"  recorded    {ledger}")
    return 0


def _who_is_running() -> str:
    """``user@host:pid`` — the "who" of §3.10's lock-break record."""
    import getpass
    import os
    import socket

    try:
        user = getpass.getuser()
    except (KeyError, OSError):  # a container with no passwd entry and no USER
        user = os.environ.get("USER") or "unknown"
    return f"{user}@{socket.gethostname()}:{os.getpid()}"


# --------------------------------------------------------------------------
# Commands whose stage has not landed
# --------------------------------------------------------------------------


def _cmd_compare(args: argparse.Namespace) -> int:
    raise LaterStage(
        "`prism compare`",
        2,
        "a comparison reads ResultRecords, and the record and the trial log arrive with the " "full evaluator",
    )


def _cmd_trials(args: argparse.Namespace) -> int:
    if args.subcommand is None:
        raise CommandError("`prism trials` needs a subcommand; the vocabulary declares `reconcile`")
    raise LaterStage(
        "`prism trials reconcile`",
        2,
        "there is no trial log to reconcile until the run-recording protocol writes one",
    )


def _cmd_folds(args: argparse.Namespace) -> int:
    if args.subcommand is None:
        raise CommandError("`prism folds` needs a subcommand; the vocabulary declares `import`")
    raise LaterStage(
        "`prism folds import`",
        2,
        "frozen fold tables are imported from recorded fold windows with the full evaluator",
    )


def _cmd_reconcile(args: argparse.Namespace) -> int:
    raise LaterStage(
        "`prism reconcile`",
        5,
        "reconciliation against a venue's fills is part of the operating workflow",
    )


def _cmd_replay(args: argparse.Namespace) -> int:
    """`prism replay <run>`: the recorded sessions, and what a version change would do to them.

    `--upgrade-diff` is the delivered read, and it is the same report as
    `prism upgrade-diff --sessions <run>`. Driving the daily cycle forward over
    a book is a different job with a different input set (a book, a universe
    and local bar panels, none of which a run directory carries), and it stays
    where it already lives, in `prism.scripts.replay_loop`.
    """
    if not args.upgrade_diff:
        raise CommandError(
            "`prism replay <run>` reads a recorded capture, and the only read it performs on one is the "
            "upgrade diff: pass --upgrade-diff. Driving the daily cycle forward over local bar panels is "
            "`python -m prism.scripts.replay_loop --help`, which needs a book and a universe that a run "
            "directory does not carry"
        )
    return _upgrade_diff(
        sessions=args.run,
        baseline=args.baseline,
        candidate=args.candidate,
        bars=args.bars,
        weight_tolerance=args.weight_tolerance,
        share_tolerance=args.share_tolerance,
        json_path=args.json_path,
    )


#: The three §3.11 freshness checks an offline `prism check` cannot evaluate,
#: with what does evaluate each. They are printed rather than omitted: a report
#: that silently listed one of four checks would read as one check passing, and
#: §3.11's whole point is that the four are separate questions.
_UNEVALUATED_OFFLINE: tuple[tuple[str, str], ...] = (
    (
        "data-freshness",
        "not evaluated offline: each source is compared with its declared publication deadline, at the "
        "declared coverage of eligible securities, by prism.live.freshness.check_data_freshness at "
        "write-ahead time; a shortfall is S1 for the affected sids",
    ),
    (
        "broker-state-freshness",
        "not evaluated offline: the age of the last positions-and-cash read is established at write-ahead "
        "time by prism.live.freshness.check_broker_state against max_broker_state_age_seconds (S2)",
    ),
    (
        "clock-health",
        "not evaluated offline: the host clock is compared with Broker.server_time by "
        "prism.live.freshness.check_clock against max_clock_skew_seconds, and no server time available is "
        "itself the fault (S8)",
    ),
)


def _cmd_check(args: argparse.Namespace) -> int:
    """The §3.11 health, authorization, freshness and missed-run surface.

    Offline and credential-free, because the alerting path cannot depend on the
    same venue reachability whose absence it has to report. Two of the four
    freshness checks — broker-state age and clock skew — need a venue and are
    named as unevaluated rather than counted as passes; the other two, data
    freshness against the session table's expected sessions and the table's own
    remaining coverage, are answered here.

    ``--expect-session`` is the missed-run monitor's trigger (§3.11), and §3.11
    requires that trigger to be independent of the trading scheduler: a check
    the trading scheduler runs cannot detect that scheduler's own failure. It
    is what an external dead-man's service invokes, so it exits nonzero when a
    session the exchange held was never decided — and, unlike the plain health
    surface, when the run directory has decided nothing at all.
    """
    from prism.scripts.doctor import CheckResult, check_missed_sessions, run_checks

    if args.monitor:
        raise LaterStage(
            "`prism check --monitor`",
            5,
            "the monitoring read is driven by policies/monitoring.toml, whose schema and reader "
            "registry are the monitoring half of this stage; `prism.scripts.paper_monitor` runs the "
            "anytime read today",
        )

    run_dir = Path(args.run).expanduser()
    if not run_dir.is_dir():
        raise CommandError(f"{run_dir} is not a run directory; `prism check` reads a directory's durable record")
    as_of = date.today() if args.asof is None else _asof_date(args.asof)

    workspace = _open_workspace(args, "check")
    universe_file = None if args.universe_file is None else Path(args.universe_file).expanduser()
    # A session question the bundled table cannot answer surfaces as the
    # `missed-sessions` row's own FAIL, naming the table and the date, rather
    # than as an exception here: the other checks still have answers worth
    # printing, and a report that died on one row would hide them.
    results = list(
        run_checks(
            run_dir=run_dir,
            universe_file=universe_file,
            data_dir=workspace.data,
            network=False,
            today=as_of,
            mode=args.account,
        )
    )

    if args.expect_session:
        # The §3.11 table, all four rows, in one place. The scheduler row is the
        # one this command can answer offline and it replaces the health
        # surface's copy rather than being printed beside it; the other three
        # are named, not omitted.
        results = [result for result in results if result.name != "missed-sessions"]
        liveness = check_missed_sessions(run_dir, today=as_of)
        if liveness.status == "WARN":
            # `--expect-session` is the caller asserting a session should have
            # been decided. A run directory with no NAV row has not decided one,
            # and reporting that as a note is how a dead loop stays green.
            liveness = CheckResult(
                liveness.name,
                "FAIL",
                liveness.detail + " — and --expect-session asserts one should have been",
            )
        results.extend(CheckResult(name, "WARN", detail) for name, detail in _UNEVALUATED_OFFLINE)
        results.append(liveness)

    print(f"run-dir  {run_dir}")
    print(f"as-of    {as_of.isoformat()}" + ("" if args.asof else "  (today; --asof pins a recorded clock)"))
    width = max(len(result.name) for result in results)
    for result in results:
        print(f"{result.status:4}  {result.name:<{width}}  {result.detail}")
    failed = [result for result in results if result.status == "FAIL"]
    warned = [result for result in results if result.status == "WARN"]
    print(f"\n{len(results) - len(failed) - len(warned)} pass, {len(warned)} warn, {len(failed)} fail")
    return EXIT_REFUSED if failed else 0


def _asof_date(text: str) -> date:
    """``--asof`` as a date, refusing anything that is not one."""
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise CommandError(f"--asof {text!r} is not an ISO date (YYYY-MM-DD)") from exc


def _cmd_upgrade_diff(args: argparse.Namespace) -> int:
    """`prism upgrade-diff --sessions <capture>`: the §3.14 upgrade review."""
    return _upgrade_diff(
        sessions=args.sessions,
        baseline=args.baseline,
        candidate=args.candidate,
        bars=args.bars,
        weight_tolerance=args.weight_tolerance,
        share_tolerance=args.share_tolerance,
        json_path=args.json_path,
    )


def _upgrade_diff(
    *,
    sessions: str,
    baseline: str | None,
    candidate: str | None,
    bars: Sequence[str] | None,
    weight_tolerance: float,
    share_tolerance: float,
    json_path: str | None,
) -> int:
    """Build, print and score one upgrade diff. Shared by `replay` and `upgrade-diff`.

    Offline and credential-free by construction: every side of the comparison
    is a directory of recorded ledgers, and no venue is reached to produce one.

    The exit code follows `prism.live.parity_gate`, which is the convention
    every gate in this repository already reports under: 1 if any compared
    session decided differently, 2 if nothing failed but at least one requested
    session could not be compared, 0 only when every requested session was
    compared and agreed. A difference outranks an incomparable session because
    a difference is a finding, while an incomparable session is the absence of
    one, and 0 is reserved for the single case where the report actually
    covered what it was asked about.
    """
    import json

    from prism.live.replay import upgrade_diff_report

    sessions_dir = Path(sessions).expanduser()
    if not sessions_dir.is_dir():
        raise CommandError(f"{sessions_dir} is not a session capture directory")
    try:
        report = upgrade_diff_report(
            sessions_dir,
            baseline=None if baseline is None else Path(baseline).expanduser(),
            candidate=None if candidate is None else Path(candidate).expanduser(),
            bars=None if bars is None else [str(bar) for bar in bars],
            weight_tolerance=weight_tolerance,
            share_tolerance=share_tolerance,
        )
    except FileNotFoundError as exc:
        # A capture without its manifest is a user error with a readable cause,
        # not a defect: it is reported as a refusal rather than as a traceback.
        raise CommandError(str(exc)) from exc
    print(report.summary())
    if json_path is not None:
        out = Path(json_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    if report.differing:
        return EXIT_REFUSED
    if not report.complete:
        return EXIT_LATER_STAGE
    return 0


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": _cmd_init,
    "data": _cmd_data,
    "conform": _cmd_conform,
    "evaluate": _cmd_evaluate,
    "compare": _cmd_compare,
    "trials": _cmd_trials,
    "folds": _cmd_folds,
    "run": _cmd_run,
    "reconcile": _cmd_reconcile,
    "replay": _cmd_replay,
    "check": _cmd_check,
    "upgrade-diff": _cmd_upgrade_diff,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _open_workspace(args: argparse.Namespace, run_name: str) -> "Workspace":
    """The three entry-point duties of §3.1, in order, once per command."""
    from prism.logging_utils import configure_logging
    from prism.workspace import load_env, resolve_workspace

    workspace = resolve_workspace(args.workspace).ensure()
    load_env()
    configure_logging(workspace.logs, verbose=args.verbose, run_name=run_name)
    logger.debug("workspace %s", workspace.root)
    return workspace


def _open_dataset(dataset_cls: type["Dataset"], workspace: "Workspace") -> "Dataset":
    """Open the workspace dataset, naming the fixture command when it is absent."""
    manifest = workspace.data / "manifest.json"
    if not manifest.is_file():
        raise CommandError(
            f"no dataset manifest at {manifest} — write the credential-free fixture first with "
            "`prism data synthetic --seed 7 --scenarios all`"
        )
    return dataset_cls.open(workspace.data)


def _universe(spec: "Specification", view: "DatasetView") -> list[str]:
    """The sids the specification's universe resolves to at this decision date."""
    if spec.strategy.universe_index:
        sids = sorted(view.membership(spec.strategy.universe_index))
        if not sids:
            raise CommandError(
                f"index {spec.strategy.universe_index!r} has no members at {view.decision_date} in "
                "this dataset; `prism data synthetic` writes synthetic_100 and synthetic_etf_10"
            )
        return sids
    if spec.strategy.universe_list:
        return list(spec.strategy.universe_list)
    raise CommandError(f"{spec.path} declares neither universe_index nor universe_list")


def _daily_book_config(spec: "Specification", loaded: "LoadedComponent") -> "DailyBookConfig":
    """The specification's rank-to-weight rule and profile limits as a book config.

    The rule supplies its own caps and the profile supplies its limits; the
    binding value is the smaller of the two, which is the down-only convention
    the construct already applies (a low-gross book stays low-gross).
    """
    from prism.live.daily import DailyBookConfig
    from prism.spec import resolve_to_weights_params

    rule = loaded.spec.to_weights
    params = resolve_to_weights_params(rule, loaded.spec.to_weights_params, where=f"{spec.path} to_weights_params")
    limits = _profile_limits(spec.policy.profile)
    execution = _read_toml(spec.policy.execution)

    max_gross = min(float(params["max_gross"]), float(limits.get("max_gross", float("inf"))))
    max_name = min(
        float(params["max_symbol_abs_weight"]),
        float(limits.get("max_name_weight", float("inf"))),
    )
    participation = limits.get("participation_per_name")
    fractional = execution.get("fractional", {})
    if not isinstance(fractional, dict) or "allowed" not in fractional:
        raise CommandError(f"{spec.policy.execution} declares no [fractional] allowed = true|false")
    common: dict[str, Any] = {
        "max_gross": max_gross,
        "max_symbol_abs_weight": max_name,
        "whole_shares": not bool(fractional["allowed"]),
        "min_order_notional": float(execution.get("min_notional", 1.0)),
        "max_participation": None if participation is None else float(participation),
        "decision_every": int(loaded.spec.update_every_bars),
    }
    if rule == "decile_long_short":
        return DailyBookConfig(book="decile_neutral", decile=float(params["decile"]), **common)
    if rule == "sign_inverse_vol":
        return DailyBookConfig(book="inverse_vol", vol_ewma_bars=int(params["vol_ewma_bars"]), **common)
    raise LaterStage(
        f"to_weights = {rule!r} in `prism run`",
        5,
        "the live daily cycle constructs the decile and inverse-vol books; the remaining rules "
        "reach an operating session through the composed book that assemble drives",
    )


def _profile_limits(path: Path) -> dict[str, float]:
    """``[[limits]]`` of a profile as ``name -> value`` (§3.13)."""
    table = _read_toml(path)
    limits: dict[str, float] = {}
    for entry in table.get("limits", []):
        if not isinstance(entry, dict) or "name" not in entry or "value" not in entry:
            raise CommandError(f"{path}: every [[limits]] entry needs a name and a value")
        limits[str(entry["name"])] = float(entry["value"])
    return limits


def _read_toml(path: Path) -> dict[str, Any]:
    import tomllib

    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CommandError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise CommandError(f"{path} cannot be read: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover — `python -m prism.cli`
    raise SystemExit(main())
