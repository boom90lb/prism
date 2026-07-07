"""SPEC N8 import-hygiene contract: importing any core `prism` module must not
pull a research heavyweight (JAX/torch/mlflow/...) into the process.

The probe runs in a subprocess with a controlled sys.path (repo src/ only, cwd
excluded via -P) so the result reflects the *import closure*, not the contents
of the dev venv — the heavy libraries ARE installed here; the contract is that
importing prism never touches them. Excluding the cwd also makes any accidental
core -> research module-level import fail loudly instead of resolving against
the repo checkout.

`prophet` and `matplotlib` joined the forbidden set with the R1 vocabulary
convergence: the legacy forecaster stack (research.models.prophet was the
only prophet importer; its matplotlib arrived transitively) is quarantined
under research/, so N8's prophet/matplotlib clause now binds the whole
production closure, live/ included.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOTS = {
    "jax",
    "flax",
    "distrax",
    "optax",
    "torch",
    "transformers",
    "gymnasium",
    "mlflow",
    "prophet",
    "matplotlib",
    "research",
}

_PROBE = """
import importlib, json, pkgutil, sys
import prism
names = [m.name for m in pkgutil.walk_packages(prism.__path__, "prism.")]
for name in names:
    importlib.import_module(name)
print(json.dumps(sorted({m.split(".")[0] for m in sys.modules})))
"""


def test_prism_import_closure_is_research_free():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    # -P keeps the cwd off sys.path (pytest's `pythonpath` ini does not
    # propagate to subprocesses, and `python -c` would otherwise prepend cwd).
    proc = subprocess.run(
        [sys.executable, "-P", "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
    )
    assert proc.returncode == 0, f"import probe failed:\n{proc.stderr}"
    roots = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    leaked = roots & FORBIDDEN_ROOTS
    assert not leaked, (
        f"core import closure pulled forbidden modules: {sorted(leaked)} (SPEC N8)"
    )
