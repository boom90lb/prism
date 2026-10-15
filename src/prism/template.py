"""The private-project template `prism init` writes (spec §4).

A project is a directory of *inputs* the engine reads by path and never writes:
the strategy module, the specification, the profiles, the cost stack, the
policies, and the fold table. The engine's own outputs go to the workspace
(§3.1), which for this template is the project directory itself, since the
template declares no ``[tool.prism].workspace`` and ``resolve_workspace`` then
falls through to the current directory.

Why the files live here as text rather than as package data on disk: the wheel
would otherwise ship a second ``pyproject.toml`` and an importable
``strategies`` package inside ``prism/``, and the import-closure probe of
``tests/test_import_hygiene.py`` walks and imports every module under the
package. Text constants keep the template out of the package's import graph and
make the pin below a single literal derived from ``prism.__version__``.

Every block is the one section 4 shows. Where section 4 elides a block for
length — the ``[validation]`` and ``[rounding]`` tables of the execution
profile — the file here carries it at the reference values of section 3.7, as
section 4 says the shipped file does.
"""

from __future__ import annotations

from pathlib import Path

from prism import __version__

#: The distribution the template pins, and the pin itself. One literal, derived
#: from the engine's own version so a template can never advertise a release
#: that was never built; ``tests/test_template.py`` pins it to the built wheel.
ENGINE_DISTRIBUTION = "prism-trading"
ENGINE_PIN = f"{ENGINE_DISTRIBUTION}=={__version__}"

#: Directories `prism init` creates whether or not it writes a file into them.
#: The four empty ones are the input directories later stages fill:
#: ``calibrations/`` and ``risk/`` at stage 4, ``registrations/`` when the first
#: selection set is opened, and ``data/frozen/`` when a score panel is frozen.
TEMPLATE_DIRECTORIES: tuple[str, ...] = (
    "strategies",
    "specs",
    "profiles",
    "costs",
    "calibrations",
    "risk",
    "policies",
    "mappings",
    "registrations",
    "folds",
    "trials",
    "runs",
    "data/frozen",
)

#: Directories that get a ``.gitkeep`` so an empty input directory survives a
#: clone. ``trials/`` and ``runs/`` deliberately get none: they are workspace
#: output, the template ``.gitignore`` excludes them, and a tracked placeholder
#: inside an ignored directory only invites someone to commit evidence rows.
_GITKEEP_DIRECTORIES: tuple[str, ...] = ("calibrations", "risk", "registrations", "data/frozen")


PYPROJECT_TOML = f"""\
# The private project. `pip install -e .` is one supported way to make
# `strategies` importable; the other is the engine's own sys.path rule, which
# puts the workspace root in front of sys.path before importing a component
# (spec §4), and neither path requires the other.
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-strategy"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["{ENGINE_PIN}"]

[tool.hatch.build.targets.wheel]
packages = ["strategies"]
"""


MY_SIGNAL_PY = '''\
"""The example component: a trailing total return, consumed by rank order.

Every field above ``__init__`` is a declaration the engine reads before it runs
the component (spec §3.3). ``prism conform`` refuses a component that omits one,
and the declarations enter ``config_id``, so changing one mints a new identity
rather than quietly changing what a recorded result means.
"""

import pandas as pd
from prism.signal.base import Signal


class TrailingReturnRank(Signal):
    component_id = "trailing"
    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    fitted_state = "none"
    price_scale_invariant = True
    estimated_decay_bars = None
    horizon_bars = 21
    update_every_bars = 21
    valid_for_bars = 21
    refit_every_bars = 1
    training_window_bars = 0

    def __init__(self, lookback_bars: int = 126) -> None:
        self.lookback_bars = lookback_bars

    @property
    def required_history(self) -> int:
        return self.lookback_bars + 1

    def fit(self, close, volume=None):
        return self

    def score(self, close, volume=None):
        # A panel too short to score raises; a *name* too short to score inside
        # a long enough panel is NaN, which the division below produces on its
        # own. That split is conformance check 4 (spec section 3.3): a missing
        # opinion is never a number, and a fabricated 0.0 would read as a
        # position instruction.
        if len(close) < self.required_history:
            raise ValueError(
                f"a {self.lookback_bars}-bar trailing return needs {self.required_history} bars, "
                f"got {len(close)}"
            )
        return close.iloc[-1] / close.iloc[-1 - self.lookback_bars] - 1.0
'''


BOOK_TOML = """\
# The long/short reference configuration.
[strategy]
name = "trailing_rank"
universe_index = "synthetic_100"
combination = "position_space"

[[strategy.components]]
id = "trailing"
path = "strategies.my_signal:TrailingReturnRank"
weight = 1.0
horizon_bars = 21
update_every_bars = 21
valid_for_bars = 21
to_weights = "decile_long_short"

[strategy.components.params]
lookback_bars = 126

[strategy.components.to_weights_params]
decile = 0.10
max_gross = 1.0
max_symbol_abs_weight = 0.10

[construction]
kind = "heuristic"

[policy]
profile = "profiles/paper.toml"
costs = "costs/retail_v1.toml"
execution = "profiles/execution_synthetic.toml"
monitoring = "policies/monitoring.toml"
recovery = "policies/recovery.toml"
"""


ETF_CORE_TOML = """\
# A clearly illustrative long-only ETF configuration for smaller accounts. It
# is a separate configuration with its own evidence and carries no
# capital-suitability claim: `min_account_status` in profiles/etf_small.toml
# stays `undetermined`. It exists so the template, the rank-to-weight rules,
# and the whole-share profile are exercised at the small end of the range.
[strategy]
name = "etf_core_long_only"
universe_index = "synthetic_etf_10"
combination = "position_space"

[[strategy.components]]
id = "trailing"
path = "strategies.my_signal:TrailingReturnRank"
weight = 1.0
horizon_bars = 21
update_every_bars = 21
valid_for_bars = 21
to_weights = "top_k_equal_weight"

[strategy.components.params]
lookback_bars = 252

[strategy.components.to_weights_params]
k = 5
max_gross = 0.98
max_symbol_abs_weight = 0.25

[construction]
kind = "heuristic"

[policy]
profile = "profiles/etf_small.toml"
costs = "costs/retail_v1.toml"
execution = "profiles/execution_synthetic.toml"
monitoring = "policies/monitoring.toml"
recovery = "policies/recovery.toml"
"""


PAPER_PROFILE_TOML = """\
# The migration of the current paper profile, carrying its certified values
# only: gross 1.0, per-name 0.10, no participation cap, no no-trade band, no
# cash buffer. It is the fixture the parity test replays against, so it must
# not acquire new limits.
name = "research_paper"
min_account_status = "undetermined"

[[limits]]
name = "max_gross"
class = "preference"
priority = 1
value = 1.0

[[limits]]
name = "max_name_weight"
class = "preference"
priority = 2
value = 0.10
"""


ETF_SMALL_PROFILE_TOML = """\
# `long_only = true` is the profile's statement, not the rank-to-weight rule's:
# a negative target under this profile is a validation failure, never a short
# order, so a long-only profile refuses a long/short specification instead of
# silently truncating it.
name = "etf_small_illustrative"
cash_buffer = 0.02
min_account_status = "undetermined"
long_only = true

[[limits]]
name = "max_gross"
class = "preference"
priority = 1
value = 1.0

[[limits]]
name = "max_name_weight"
class = "preference"
priority = 2
value = 0.25

[[limits]]
name = "participation_per_name"
class = "physical"
priority = 0
value = 0.05

[[limits]]
name = "quantity_increment"
class = "physical"
priority = 0
value = 1.0
"""


EXECUTION_SYNTHETIC_TOML = """\
# The whole-share, no-fractional, single-session counterpart of an
# `execution_alpaca.toml`. `capabilities.source = "synthetic"` reads the
# per-sid dated capability rows the fixture writer produced instead of polling
# a broker, so the example needs no credential.
quantity_increment_default = 1.0
min_notional = 1.0
price_convention = "next_open"
order_types = ["market_on_open"]
default_order_type = "market_on_open"
sessions = ["regular"]

[fractional]
allowed = false
time_in_force = ["day"]
short_allowed = false

[capabilities]
source = "synthetic"
max_age_hours = 0
revalidate_at_submission = false

[fills]
model = "full_next_open"
max_participation_of_bar_volume = 0.0
partial_fill_fraction = 1.0
reject_probability = 0.0
missing_open_policy = "reference_price"

# Tolerances are declared, not implied (spec §3.7). `mode = "report_only"`
# exists for the stage 5 replay parity gate and for nothing else.
[validation]
limit_tolerance = 1.0e-6
cash_tolerance = 0.01
cost_tolerance_bps = 0.1
mode = "blocking"

# Batch rounding and repair (spec §3.7). The `historical_constant_weight`
# engine forces `zero_crossing_clamp = false`, `apply_dust_rule = false`, and a
# continuous increment, because that is what the reproduction engine did; the
# `canonical` engine uses these values.
[rounding]
tie_rule = "half_to_even"
zero_crossing_clamp = true
apply_dust_rule = true
fractional_increment = 0.001
off_increment_holding = "leave_residual"
scale_schedule = [0.75, 0.5, 0.25]
repair_max_moves = 200
repair_min_decrement = 1.0e-6
min_implemented_gross_fraction = 0.80
"""


RETAIL_COSTS_TOML = """\
# Every component of this file is an assumption: no fill-based calibration
# record exists yet, and these are the ExecutionConfig defaults plus the
# pre-registered bucket schedule. `class = "calibrated"` requires a named
# calibration record, and the first one is a stage 5 reconciliation output.
commission_bps = 1.0
borrow_bps_annual = 50.0
cash_income_bps_annual = 0.0
financing_bps_annual = 0.0
short_proceeds = "collateral_no_rebate"
holding_cost_settlement = "per_bar"
trading_days_per_year = 252

# `bound = "inclusive_lower"`: each step is an inclusive lower bound on the
# declared liquidity statistic, steps descending. `unknown_liquidity =
# "widest_step"`: a missing, NaN, or non-positive statistic takes the last
# step, never the cheapest.
[spread]
mode = "bucket"
statistic = "dollar_volume.spread_bucket"
bound = "inclusive_lower"
unknown_liquidity = "widest_step"
schedule = [[500000000.0, 1.0], [100000000.0, 2.0], [25000000.0, 5.0], [0.0, 10.0]]

[impact]
quadratic_notional_coeff = 10.0
participation_model = "sqrt"
participation_coeff = 0.0
statistic = "dollar_volume.impact"

# The three uses of a dollar-volume statistic each name their own estimator,
# window, refresh cadence, and floor. `window = "formation"` is legal only with
# `refresh = "fold"` under an evaluation that declares a fold table.
[dollar_volume.spread_bucket]
estimator = "median"
window = "trailing"
window_bars = 20
refresh = "decision"
floor_dollars = 0.0
source = "close_times_volume"

[dollar_volume.participation]
estimator = "median"
window = "trailing"
window_bars = 20
refresh = "decision"
floor_dollars = 0.0
base = "current_equity"
source = "close_times_volume"

[dollar_volume.impact]
estimator = "same_bar"
window = "trailing"
window_bars = 1
refresh = "bar"
floor_dollars = 100000.0
source = "close_times_volume"

[source.commission]
class = "assumption"
reference = "src/prism/config.py::ExecutionConfig.commission_bps"

[source.spread]
class = "assumption"
reference = "docs/r2_design.md section 3"

[source.impact]
class = "assumption"
reference = "src/prism/config.py::ExecutionConfig.slippage_coeff"

[source.borrow]
class = "assumption"
reference = "src/prism/config.py::ExecutionConfig.borrow_rate_bps_annual"

[source.financing]
class = "assumption"
reference = "no rate established; zero is a stated placeholder"

[source.cash_income]
class = "assumption"
reference = "no rate established; zero is a stated placeholder"
"""


MONITORING_POLICY_TOML = """\
# Monitoring is telemetry. `response = "report"` is the only supported value in
# this version: a deterioration detector that changes exposure needs its own
# specification, and the loader refuses any other value rather than implying one.
evidence_window_periods = 21
statistic = "mean_net_return"
alpha = 0.05
hurdle = 0.0
opt_horizon = 252
min_periods = 6
reset_policy = "never"
readers = ["anytime", "paper_monitor"]
response = "report"
report_every_session = true
"""


RECOVERY_POLICY_TOML = """\
# Every response key is a closed enumeration (spec §3.4, §3.7). This file's
# content enters `config_id` through `policy_ids.recovery` in the authorization
# record, which is why it carries a declared schema rather than an inferred one.
on_infeasible = "cancel"
on_sign_contradiction = "hold"
on_blocked_obligation = "escalate"
on_deadline_under_kill_switch = "escalate"
on_rounding_failure = "scale_batch"
on_partial_batch = "cancel_accepted"
preference_response = "record"
bound_slack = 1.0e-6
recovery_entry_margin = 0.002
component_liquidate_priority = 1

[escalation]
write_alert = true
exit_code = 3
require_acknowledgement = true
"""


SMOKE_FOLDS_TOML = """\
# The two-fold table `prism evaluate --smoke` runs on. The windows are sessions
# of the bundled table, inside the synthetic fixture's span (2026-01-02 to
# 2027-12-31): a 251-session formation window rolled forward by one test window
# of 126 and 125 sessions. A smoke run is a fixed engineering run, not a small
# evaluation, so this table is shipped rather than derived.
id = "smoke_v1_folds"
source = "prism data synthetic (the bundled session table, 502 sessions)"
timezone = "America/New_York"
convention = "inclusive_session_dates"

[[fold]]
index = 0
formation_start = 2026-01-02
formation_end = 2026-12-31
test_start = 2027-01-04
test_end = 2027-07-06

[[fold]]
index = 1
formation_start = 2026-07-07
formation_end = 2027-07-06
test_start = 2027-07-07
test_end = 2027-12-31
"""


TWELVEDATA_MAPPING_TOML = """\
# The single declaration point for a local ingest: the column names, the source
# lag, the price basis, and the revision policy. `prism data ingest` reads it;
# `lag_bars` is copied into the manifest and the views read it from there, so a
# lag can never differ between two readers of the same source.
schema = 1

[source]
id = "twelvedata"
kind = "vendor_api"
lag_bars = 0
revision_policy = "dedupe_identical"
price_basis = "split_adjusted_at_ingest"
timezone = "America/New_York"
gap_flag_sessions = 15
redistributable = false

[columns]
ts = "datetime"
open = "open"
high = "high"
low = "low"
close = "close"
volume = "volume"

[identity]
vendor_symbol = "symbol"
match = "ticker_interval_overlap"
"""


GITIGNORE = """\
# Engine output. The workspace layout (spec §3.1) is written into this
# directory when PRISM_WORKSPACE is not set elsewhere; none of it is project
# input, and `trials/` in particular is append-only selection evidence that
# must never arrive through a merge.
data/*
!data/frozen/
results/
state/
logs/
trials/
runs/

# Credentials are read from .env by command entry points, never at import.
.env

__pycache__/
*.py[cod]
.venv/
dist/
"""


ENV_EXAMPLE = """\
# Copy to .env. Command entry points load it after argument parsing; importing
# the engine reads no credential and creates no directory.
#
# The section 4 example needs none of these: `prism data synthetic` generates
# its own fixture and `--broker synthetic` places no order with a venue.

# Vendor bar downloads (`prism data fetch --source twelvedata`).
# TWELVEDATA_API_KEY=

# Paper or live trading through the Alpaca adapter (stage 5).
# ALPACA_API_KEY=
# ALPACA_SECRET_KEY=
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""


CI_YML = """\
# Private CI: offline by construction. The job installs the engine, copies the
# project into a throwaway workspace, and runs the stage 1 exit sequence there
# against the synthetic fixture. No credential is read, no market data is
# fetched, and no live account is touched.
#
# The copy is what keeps engine output out of the checkout: a specification
# resolves its policy files and imports its `strategies` package against the
# workspace root, so the workspace has to be the project — and a CI job must
# never append a run-recording row to the project's own trials/, which is
# append-only selection evidence. Copying satisfies both.
name: prism

on:
  push:
  pull_request:

jobs:
  offline-smoke:
    runs-on: ubuntu-latest
    env:
      PRISM_WORKSPACE: ${{ runner.temp }}/prism-workspace
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
      - name: install the pinned engine
        # Until the distribution is published, point pip at a locally built
        # wheel instead: python -m pip install --find-links <dir> .
        run: python -m pip install --upgrade pip && python -m pip install .
      - name: offline smoke
        run: |
          mkdir -p "$PRISM_WORKSPACE"
          cp -R . "$PRISM_WORKSPACE"
          cd "$PRISM_WORKSPACE"
          prism data synthetic --seed 7 --scenarios all
          prism conform specs/book.toml
          prism conform specs/etf_core.toml
          prism evaluate specs/book.toml --engine historical_constant_weight --smoke
          prism evaluate specs/etf_core.toml --engine historical_constant_weight --smoke
          prism run --spec specs/book.toml --account paper --once --broker synthetic
"""


#: Every file the template writes, by path relative to the project directory.
TEMPLATE_FILES: dict[str, str] = {
    "pyproject.toml": PYPROJECT_TOML,
    "strategies/my_signal.py": MY_SIGNAL_PY,
    "specs/book.toml": BOOK_TOML,
    "specs/etf_core.toml": ETF_CORE_TOML,
    "profiles/paper.toml": PAPER_PROFILE_TOML,
    "profiles/etf_small.toml": ETF_SMALL_PROFILE_TOML,
    "profiles/execution_synthetic.toml": EXECUTION_SYNTHETIC_TOML,
    "costs/retail_v1.toml": RETAIL_COSTS_TOML,
    "policies/monitoring.toml": MONITORING_POLICY_TOML,
    "policies/recovery.toml": RECOVERY_POLICY_TOML,
    "folds/smoke.toml": SMOKE_FOLDS_TOML,
    "mappings/twelvedata_daily.toml": TWELVEDATA_MAPPING_TOML,
    ".gitignore": GITIGNORE,
    ".env.example": ENV_EXAMPLE,
    ".github/workflows/ci.yml": CI_YML,
}


class TemplateError(RuntimeError):
    """`prism init` cannot write the template where it was pointed."""


def write_template(destination: Path | str, *, force: bool = False) -> list[Path]:
    """Write the template into ``destination`` and return the files written.

    Refuses a destination that already holds any of the template's files unless
    ``force``: a project is a directory of hand-edited inputs, and overwriting
    a strategy module or a cost stack without being asked is the one failure
    mode an init command must not have. ``force`` overwrites the template's own
    files and still leaves everything else in the directory untouched.
    """
    root = Path(destination).expanduser()
    if root.exists() and not root.is_dir():
        raise TemplateError(f"{root} exists and is not a directory")
    if not force:
        clashes = sorted(name for name in TEMPLATE_FILES if (root / name).exists())
        if clashes:
            raise TemplateError(
                f"{root} already holds {len(clashes)} template file(s) — first: {clashes[0]}; "
                "point `prism init` at a new directory, or pass --force to overwrite the "
                "template's own files"
            )
    root.mkdir(parents=True, exist_ok=True)
    for name in TEMPLATE_DIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in _GITKEEP_DIRECTORIES:
        keep = root / name / ".gitkeep"
        keep.write_text("", encoding="utf-8")
        written.append(keep)
    for name, content in TEMPLATE_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return sorted(written)
