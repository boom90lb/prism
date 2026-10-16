"""Suite-wide guards.

``PRISM_DISABLE_DOTENV=1`` switches off ``.env`` discovery
(``prism.workspace.load_env``) for every test and every subprocess a test
spawns, so running the suite inside a deployment checkout never loads that
checkout's broker credentials. A test that exercises discovery opts back in
with ``monkeypatch.delenv("PRISM_DISABLE_DOTENV", raising=False)``.

``PRISM_HOST_LABEL`` gives the trial-log rows tests write a fixed host label,
so no test creates ``~/.config/prism/host_id`` for the account running the
suite (``prism.validation.trials.host_label``). A test of the generated label
removes the variable and points ``XDG_CONFIG_HOME`` at its own directory.
"""

import os


def pytest_configure(config):
    os.environ.setdefault("PRISM_DISABLE_DOTENV", "1")
    os.environ.setdefault("PRISM_HOST_LABEL", "pytest")
