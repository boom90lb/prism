# Security policy

Prism holds broker and data-vendor API keys and can send orders to a broker, so a security bug here
can cost someone money or leak their credentials. Please report one privately.

## How to report

Do not open a public issue, discussion or pull request for a security problem.

Email **boom90lb@alum.mit.edu**. If this repository's Security tab offers "Report a vulnerability",
you can use that instead. Please include:

- what the problem is and where it is (file and line, command, commit or release);
- the steps that reproduce it;
- what it exposes or allows, as far as you know.

Do not put working credentials, account numbers or anyone's private data in the report. If you
found a credential in this repository or its history, say where it is and do not use it.

Reports are handled on a best-effort basis; there is no guaranteed response time.

## What to report here

Examples of problems that belong in a private report:

- a credential (an Alpaca, Twelve Data or FRED key, or an alert or heartbeat URL) written to a log
  line, an exception message, a recorded result, a built wheel or sdist, or a tracked file;
- a way for the engine to send, change or cancel a broker order without the explicit steps the
  documentation requires: the paper endpoint is the default, `--submit` is needed to send orders,
  and a live account needs a matching local authorization record (`docs/contributing.md`
  section 3);
- a `.env` or other credential file being read when the documentation says it will not be, for
  example while `PRISM_DISABLE_DOTENV=1` is set;
- a packaging or dependency problem that makes installing or importing the package run code it
  should not.

Open an ordinary issue instead for strategy performance, trading losses, disagreement with a
recorded result, and bugs that involve no credential, order path or install integrity.

## Supported versions

Security fixes are made on the `main` branch and go into the next release. Earlier tags and
releases are not patched.

## How credentials are handled

The rules for holding, sending and logging credentials, and the record of past incidents, are in
[`docs/security.md`](docs/security.md). In short: credentials come only from the environment or a
gitignored `.env`, never from command-line arguments or tracked files; keys travel in request
headers wherever the vendor allows it; any URL that carries a key is redacted before it is logged;
and the broker client uses the paper endpoint unless a live account is chosen explicitly.
