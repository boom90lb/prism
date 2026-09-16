"""Prism — a cross-sectional systematic trading engine.

v0.3.0 re-founds the project on the score -> residualize -> construct -> execute
spine (see SPEC.md); the per-symbol forecaster ensemble is demoted to one
plug-in signal node. Package, distribution, and repository are all named
``prism`` (src-layout, ``src/prism/``) as of the 2026-07-05 publication
rename (SPEC.md §12); v0.4.0 renames the *distribution* to ``prism-trading``
for the installed private-project workflow, leaving the import name ``prism``
unchanged. ``__version__`` is the distribution version: ``tests/test_template.py``
pins the two together and to the version the project template installs.
"""

__version__ = "0.4.0"
