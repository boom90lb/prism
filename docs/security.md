# Security and credentials

> **Status: in force (drafted 2026-07-18).** How credentials are held,
> transported and logged by anyone running Prism against their own vendor and
> broker accounts. The project's other rules are in SPEC.md; this file states
> operational rules only. Terms are explained in the [glossary](glossary.md).

## 1. Credential inventory

| Credential | Env var | Transport | Notes |
|---|---|---|---|
| Twelve Data key | `TWELVEDATA_API_KEY` | query param (vendor requirement) | redaction rule applies (section 2, rule 4) |
| Alpaca key pair | `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` | request headers only | paper endpoint by default (section 2, rule 5) |
| FRED key | `FRED_API_KEY` | query param (vendor requirement) | redacted at `FredClient` (`src/prism/regime/fetch.py`) |
| Alpha Vantage key | secret of the separate `prism-observatory` repository | query param | lives in that repository's Actions secrets, never in this one |

**Prism never touches bank credentials.** Funding, ACH linkage, and account
numbers live at the broker; the only thing this codebase ever holds is broker
API keys. Any doc or discussion phrased as "connect a routing/account number"
is wrong by construction.

## 2. Rules

1. **Env-only.** Credentials enter through the process environment or a
   gitignored `.env` (loaded by `prism.workspace.load_env`, called from each
   command's `main` -- never at import of `prism`; the research stack's
   `research/config.py` calls it at its own import). `PRISM_DISABLE_DOTENV=1` switches
   discovery off (the test suite sets it in `tests/conftest.py`; operators who
   export credentials themselves can too); an explicit path still loads. Never CLI arguments
   (shell history, `ps`), never tracked files, never code literals.
2. **Fail loudly on absence (invariant N7).** Every `from_env` constructor raises when a
   required credential is missing (`alpaca.py`, `alpaca_data.py`,
   `regime/fetch.py`). No silent degraded mode for live paths; the loader's
   keyless warning applies to cached-only research reads.
3. **Headers over query params.** Keys travel in request headers wherever the
   vendor allows (Alpaca). Query-param transport exists only where the vendor
   requires it (Twelve Data, FRED), which is what makes rule 4 binding.
4. **Redaction.** Any log or exception line that can embed a request
   URL passes through a redactor before logging: `DataLoader._redact`
   (`src/prism/io/loader.py`), `FredClient.series`
   (`src/prism/regime/fetch.py:99` precedent). `requests.HTTPError` text
   embeds the full `?apikey=…` URL — that is the leak vector. Enforced by the
   credential-hygiene tests in `tests/test_data_loader.py`.
5. **Paper-first.** `AlpacaClient.from_env` resolves to the paper endpoint
   unless `APCA_API_BASE_URL` is explicitly set (`src/prism/live/alpaca.py`).
   Pointing the loop at a live account is a deliberate, explicit act.
6. **Rotation on exposure.** A key that has ever appeared in a log, artifact,
   or terminal scrollback is rotated, not shrugged at. Exposure through a
   pre-redaction log line counts.
7. **No secrets in tracked artifacts.** `results/` JSON, docs, and universe
   files carry no credentials. `logs/` is untracked but treated as sensitive
   anyway: it is included in off-box backups (an operator concern), so rule 4
   must hold *before* lines reach disk, not be cleaned up after.
8. **Trial logs name no machine.** A new trial-log row records `host` as a
   label: `PRISM_HOST_LABEL` when it is set, else a random `host-<hex>` id
   created once in `$XDG_CONFIG_HOME/prism/host_id`
   (`prism.validation.trials.host_label`). A finished row's record path is
   relative to the workspace. Reconciliation treats both the label and the
   machine's hostname as this host, so rows written before the label existed,
   which carry the hostname and absolute paths, still reconcile. Those rows
   stay as written, because ledgers are append-only (`SPEC.md` section 4).

## 3. Incident record

- **2026-07 — Twelve Data key in fetch-error logs.** Loader error paths logged
  `str(exception)`, which embeds the full request URL including `apikey=`,
  during vendor-outage and rate-limit events. Fixed by the redaction rule
  (section 2, rule 4: `DataLoader._redact`, test-enforced); the exposed key was
  rotated 2026-07-18. Closed. This incident is why redaction is stated as a
  rule and not a style preference.

- **2026-07-18 — WSL disk-flush failure wrote zero-byte git objects.** During
  a commit, a flush failure produced four zero-byte loose
  objects (one commit plus its tree/blob set, destroyed before push). Repair:
  purge the empty objects, rebuild the index (`rm .git/index` +
  `git read-tree HEAD` — a stale index referencing purged blobs breaks
  `git reset`), expire reflogs, recommit from worktree text, `fsck` clean.
  Consequences adopted: the paper-loop ledgers under `runs/` — the one
  artifact class that cannot be regenerated, only re-accumulated — are
  snapshotted to a dedicated backup repository and pushed off-box to a
  private remote (`prism-artifacts`) by the nightly sync
  (`ops/artifacts_sync.sh` since 2026-09-30, replacing the operator-local
  `~/bin/prism_artifacts_sync.sh`; wired into the nightly runner and
  unconditional on loop exit status — a red loop is exactly when the ledger
  copy matters most); the `ratify/*` tags and in-tree timestamp
  proofs this sentence once described were retired in the 2026-09-02
  consolidation and no longer exist (`docs/erratum_2026-10-14.md` E7). Closed 2026-07-19: first off-box push verified.
