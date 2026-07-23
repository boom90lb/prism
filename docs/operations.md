# Operations notes

Durable operational facts for running the pipeline on real data — the
"what will actually bite you" companion to the README. The owner's
deployment checklist lives in `docs/human_review_handbook.md` §3.

## Path to deployment (owner acts; agents cannot fund or arm)

These are the open items between the v0.3.4 checkpoint and an honest
v0.4.0. Record each result durably (ops log or dated section here) before
any deployment claim.

| Step | Owner decision | Evidence to attach |
|---|---|---|
| **Micro-account (A3)** | Fund ≤ $2k of cost-calibration equity (SPEC §10 carve-out) | Cap, funding date, venue account id — no secrets in git |
| **Venue check (i)** | Fractional short acceptance and per-name fractionability | API probe receipt; on failure, whole-share ≥ $100k is the only mode |
| **Venue check (ii)** | Day-order versus auction fill quality from the paper stream | Paper fill telemetry summary |
| **Capital mode** | Whole-share auction ≥ $100k, **or** fractional day orders if both venue checks pass | Chosen mode written down |
| **Gate G0** | ≥ 1 full rebalance cycle of real fills; spread tables recalibrated from them | Fill ledger + recalibrated buckets |
| **Regime clock** | ≥ 21 consecutive clean paper sessions with regime telemetry (`regime.jsonl`); any fail-loud event restarts the count | Session count + date of last event |
| **De-gross arming** | Dedicated commit, only after sizing pins and the regime clock both hold | Commit SHA |
| **First real order** | Named risk profile + a sleeve at its evidence bar + live URL set intentionally | Claim packet with profile id |

Paper is the default: `APCA_API_BASE_URL` unset means the paper endpoint
(`docs/security.md` §2.5). Risk profiles:
`python -m prism.scripts.paper_loop --profile research_paper ...`
(schema frozen — `docs/risk_profile_schema.md`).

Remaining ops hardening called out by the program: boot-resilient nightly
scheduling (the paper stream has already shown sleep/gap risk), a filesystem
health check in the nightly wrapper, and a one-page deploy runbook covering
capital mode × risk profile × venue × custody × kill switch.

## Data vendor (Twelve Data)

- The key goes in a gitignored `.env`: `TWELVEDATA_API_KEY=...`
  (loaded by `src/prism/config.py`).
- **Interval strings.** The vendor expects `1day`/`1week`/`1month`; the
  project keeps the short forms (`1d`, `1wk`, `1mo`) everywhere and
  normalizes only at the request boundary (`DataLoader._to_vendor_interval`).
- **The free tier bites.** ~8 requests/min and 800/day — pre-warm the cache
  before a multi-symbol run. The `/dividends` endpoint returns 403 for most
  symbols on this tier, so cross-symbol total return is inconsistent there:
  run backtests with `--no_dividends` for a uniform price-return comparison,
  or supply a higher-tier key.
- **Dividend caching is failure-aware.** A failed fetch is never cached, and
  a genuinely-empty "no dividends" answer expires after 7 days — so a tier
  outage cannot poison later runs and a tier upgrade heals on its own.

## Which ensemble members actually contribute

For the legacy research backtest: `xgboost` is the default forecast member;
`arima` and `prophet` are opt-in diagnostics only; the forecast `lstm` was
removed outright; the RL policy members run if requested but see the
performance note below. For a forecast-only backtest, train with
`--models xgboost`.

## Per-bar prediction cost

The backtest calls `predict` once per bar over the test range (~1,600
bars/symbol on a multi-year run):

- **JAX members re-trace on most bars** (input shape variation defeats the
  compile cache). A 4-symbol forecast+RL backtest can take ~80 minutes,
  almost all of it retracing; the same universe forecast-only runs in
  ~15–20 minutes. Don't pay for members you don't need.
- **statsmodels warns once per bar** and routes some warnings through
  logging, so expect chatty stderr regardless of `PYTHONWARNINGS`. Redirect
  output to a file and skip `--verbose`.

## Reproducing a run from scratch

```bash
echo "TWELVEDATA_API_KEY=..." > .env

python -m research.scripts.training \
  --symbols AAPL,MSFT,GOOG,AMZN \
  --start_date 2020-01-01 --end_date <today> \
  --horizon 5 --n_splits 5 --models xgboost

PYTHONWARNINGS=ignore python -m research.scripts.backtest \
  --training_run runs/<run_name> --no_dividends
```

Participation impact is opt-in (`--adv_impact_coeff`, `--adv_floor_dollars`);
dollar volume comes from the close × volume bars already on disk. Artifacts
land under `runs/` (per-fold models) and `results/` (backtest report); both
are gitignored, as is `logs/`.

## The synthetic-versus-real lesson

Every verification before the first real-data run used synthetic
random-walk prices, which masked three real bugs (vendor interval string,
ensemble member persistence, predict-frame column set). When changing
data/model/backtest plumbing, prefer a real or schema-faithful fixture;
`tests/test_real_pipeline_regressions.py` guards these specific paths.
