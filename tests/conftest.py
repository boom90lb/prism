"""Suite-wide guards.

``PRISM_DISABLE_DOTENV=1`` switches off ``.env`` discovery
(``prism.workspace.load_env``) for every test and every subprocess a test
spawns, so running the suite inside a deployment checkout never loads that
checkout's broker credentials. A test that exercises discovery opts back in
with ``monkeypatch.delenv("PRISM_DISABLE_DOTENV", raising=False)``.
"""

import os


def pytest_configure(config):
    os.environ.setdefault("PRISM_DISABLE_DOTENV", "1")
