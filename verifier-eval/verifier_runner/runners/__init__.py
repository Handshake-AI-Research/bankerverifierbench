"""Per-verifier runners.

Each module exposes a translator from the verifier's native output format
into a canonical :class:`verifier_runner.verdict.Verdict`. Translators are
pure (no I/O beyond reading the verifier's already-produced output files).

Future additions: each runner may also expose an ``invoke()`` function that
runs the verifier from scratch — for Gandalf this means shelling out to
Harbor; for the others it means calling their Python libraries directly.
"""
