"""Uniform runner interface.

Every verifier runner exposes:

* ``capabilities()`` — declares what artifacts the runner needs. The
  orchestrator checks these against an :class:`InputBundle` and skips
  (with a recorded reason) when a required artifact is unavailable.

* ``invoke(bundle, model)`` — runs the verifier and returns a
  :class:`Verdict`. Pure side-effect-free with respect to the eval
  framework; the runner may make network calls, spawn subprocesses,
  etc., but it doesn't write to ``results/``.

Runners do not depend on each other. Adding a new verifier means
writing one new module here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    """What an :class:`InputBundle` must contain for this runner to work."""

    needs_workspace: bool = False
    needs_trajectory: bool = False
    # Whether a Harbor placeholder trajectory ("verify-only" / nop agent)
    # is acceptable as input. Runners that grade trajectory text need a
    # real trajectory; runners that ignore trajectory or grade only the
    # final workspace can accept the placeholder.
    accepts_placeholder_trajectory: bool = True


class SkipReason(str):
    """String-typed reason codes for orchestrator skip decisions."""


SKIP_NO_WORKSPACE = "no workspace artifact (run `stage` first?)"
SKIP_NO_TRAJECTORY = "no trajectory artifact"
SKIP_PLACEHOLDER_TRAJECTORY = "trajectory is Harbor placeholder"
SKIP_SYNTHETIC_TRAJECTORY = "trajectory is a synthetic BVB placeholder; only Gandalf is permitted"
SKIP_GANDALF_NO_INFO_JSON = "no verifier/info.json in staged trial dir (Harbor verification didn't run?)"
SKIP_NO_REWARDKIT_OUTPUT = (
    "no verifier/{reward,reward-details}.json in staged trial dir "
    "(was the cohort staged with --verifier-kind rewardkit?)"
)
