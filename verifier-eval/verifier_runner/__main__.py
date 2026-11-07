"""CLI entry point.

Run with ``python -m verifier_runner <subcommand> ...`` (or
``python verifier-eval/__main__.py`` from the repo root, less common).

Subcommands:
    ingest-gandalf-trial   Translate one Harbor trial → Verdict JSON
    ingest-gandalf-job     Translate every trial under a jobs/<run> dir
    score                  Score a run's verdicts vs. ground truth
"""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
