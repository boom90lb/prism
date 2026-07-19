# Security & credential doctrine

> **Status: active doctrine (drafted 2026-07-18).** Governs how credentials
> are held, transported, and logged by anyone running prism against their own
> vendor and broker accounts. Agent conduct lives in AGENTS.md; research law
> in SPEC.md. This file states operational invariants only.

## 1. Credential inventory

| Credential | Env var | Transport | Notes |
|---|---|---|---|
| Twelve Data key | `TWELVEDATA_API_KEY` | query param (vendor requirement) | redaction invariant applies (§2.4) |
| Alpaca key pair | `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` | request headers only | paper endpoint by default (§2.5) |
| FRED key | `FRED_API_KEY` | query param (vendor requirement) | redacted at `FredClient` (`src/prism/regime/fetch.py`) |
| Alpha Vantage key | observatory repo secret | query param | lives in prism-observatory Actions secrets, never in this repo |

**Prism never touches bank credentials.** Funding, ACH linkage, and account
numbers live at the broker; the only thing this codebase ever holds is broker
API keys. Any doc or discussion phrased as "connect a routing/account number"
is wrong by construction.

## 2. Invariants

1. **Env-only.** Credentials enter through the process environment or a
   gitignored `.env` (loaded in `src/prism/config.py`). Never CLI arguments
   (shell history, `ps`), never tracked files, never code literals.
2. **Fail-loud on absence (N7).** Every `from_env` constructor raises when a
   required credential is missing (`alpaca.py`, `alpaca_data.py`,
   `regime/fetch.py`). No silent degraded mode for live paths; the loader's
   keyless warning applies to cached-only research reads.
3. **Headers over query params.** Keys travel in request headers wherever the
   vendor allows (Alpaca). Query-param transport exists only where the vendor
   requires it (Twelve Data, FRED) — which is what makes invariant 4 binding.
4. **Redaction invariant.** Any log or exception line that can embed a request
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
   anyway: it is included in off-box backups (operator concern), so invariant
   4 must hold *before* lines reach disk, not be cleaned after.

## 3. Incident record

- **2026-07 — Twelve Data key in fetch-error logs.** Loader error paths logged
  `str(exception)`, which embeds the full request URL including `apikey=`,
  during vendor-outage and rate-limit events. Fixed by the §2.4 redaction
  invariant (`DataLoader._redact`, test-enforced); exposed key rotated
  2026-07-18. Closed. This incident is why §2.4 is stated as an invariant
  and not a style preference.
