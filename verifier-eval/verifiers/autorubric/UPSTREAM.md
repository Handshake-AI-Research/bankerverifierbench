# AutoRubric — provenance

## Source

PyPI package [`autorubric`](https://pypi.org/project/autorubric/), pinned to:

```
autorubric==1.0.1
```

(latest as of 2026-05-01; previously also available as 0.3.2 and 1.0.0).

Docs: https://autorubric.org/docs/

## Why this layout

AutoRubric is a pure pip package — there's no source to vendor. We
isolate it in its own uv project so:

* dependencies (LiteLLM transitively) don't conflict with AAAJ or
  archipelago, both of which also pin LiteLLM to specific versions;
* the version is pinned in source; reproducible across laptops via
  `uv sync` from this directory.

## What the runner needs

The `verifier_runner.runners.autorubric` runner shells out to a
`worker.py` (in this directory, in a follow-up commit) running in this
venv. Worker imports:

```python
from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    LLMConfig,
    Rubric,
)
from autorubric.graders import CriterionGrader
```

If autorubric ever bumps a major version that breaks one of those
import paths, pin a working version here and document the bump.
