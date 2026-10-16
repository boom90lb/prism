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

Every project ``prism init`` writes names its component package
``strategies``. Tests build such projects in their own temporary directories,
so the package is dropped from ``sys.modules`` around each test: otherwise the
next test resolves ``strategies.my_signal`` to an earlier test's deleted file.
"""

import os
import sys

import pytest

_TEMPLATE_PACKAGE = "strategies"


def pytest_configure(config):
    os.environ.setdefault("PRISM_DISABLE_DOTENV", "1")
    os.environ.setdefault("PRISM_HOST_LABEL", "pytest")


def _drop_template_package() -> None:
    for name in [n for n in sys.modules if n == _TEMPLATE_PACKAGE or n.startswith(_TEMPLATE_PACKAGE + ".")]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _isolate_template_package():
    _drop_template_package()
    yield
    _drop_template_package()
