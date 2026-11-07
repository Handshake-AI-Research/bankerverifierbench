PYTHON ?= uv run python
BVB_RELEASE_DIR ?= $(CURDIR)/bvb-data/BankerVerifierBench
BVB_OUTPUT_DIR ?= $(CURDIR)/datasets/gandalf
BVB_REWARDKIT_OUTPUT_DIR ?= $(CURDIR)/datasets/rewardkit
BVB_COHORT := bvb
BVB_TASK_IDS ?=
RESULTS_BUNDLE ?= results/paper
SMOKE_DIR := $(CURDIR)/datasets/smoke
VERIFIER_SMOKE_DIR := $(CURDIR)/datasets/verifier-smoke
TOOL_SMOKE_DIR := $(CURDIR)/datasets/tool-smoke
VERIFIER_SMOKE_TASK ?= btb-68259d37
VERIFIER_SMOKE_MODEL ?= gemini/gemini-3-flash-preview
SMOKE_VERIFIERS ?= gandalf,aaaj,archipelago,autorubric
SMOKE_STAMP := $(shell date -u +%Y%m%dT%H%M%SZ)
VERIFIER_SMOKE_RUN_ID ?= bvb-verifier-smoke-$(VERIFIER_SMOKE_TASK)-$(SMOKE_STAMP)
REWARDKIT_SMOKE_RUN_ID ?= bvb-rewardkit-claude-smoke-$(VERIFIER_SMOKE_TASK)-$(SMOKE_STAMP)
DAYTONA_CONTEXT_DIR ?= $(CURDIR)/bvb-cache/daytona-snapshot-context
DAYTONA_SNAPSHOT ?=
DAYTONA_CACHE_MAX_BYTES ?= 2147483648
RUN_ID ?=
PREDICTIONS ?=

comma := ,
empty :=
space := $(empty) $(empty)
HOST_SMOKE_VERIFIERS = $(filter-out gandalf,$(subst $(comma),$(space),$(SMOKE_VERIFIERS)))
HOST_SMOKE_MODELS = $(subst $(space),$(comma),$(foreach v,$(HOST_SMOKE_VERIFIERS),$(v)=$(VERIFIER_SMOKE_MODEL)))
LOAD_ENV := set -a; [ ! -f "$(CURDIR)/.env" ] || . "$(CURDIR)/.env"; set +a
VERIFIER_RUNNER := PYTHONPATH="verifier-eval" $(PYTHON) -m verifier_runner --release-dir "$(BVB_RELEASE_DIR)"

.PHONY: verify-results fetch validate materialize rewardkit-materialize smoke tool-smoke test score verifier-bootstrap gandalf-smoke verifier-smoke rewardkit-claude-smoke analysis daytona-snapshot runtime-snapshots

# No credentials, no downloads.
verify-results:
	python3 scripts/bvb_public_results.py check --bundle "$(RESULTS_BUNDLE)"
	python3 scripts/bvb_rollout_generation.py check

# Downloads the dataset and shared tools; loads optional HF_TOKEN from .env.
fetch:
	@$(LOAD_ENV); \
	$(PYTHON) -m adapters.bvb.fetch --with-tools --output-dir "$(BVB_RELEASE_DIR)"

validate:
	$(PYTHON) -c "from adapters.bvb.prerequisites import validate_release; release, result = validate_release(__import__('pathlib').Path('$(BVB_RELEASE_DIR)'), require_tools=True); print(result)"

materialize: validate
	$(PYTHON) -m adapters.bvb.run_adapter --release-dir "$(BVB_RELEASE_DIR)" --output-dir "$(BVB_OUTPUT_DIR)" $(if $(strip $(BVB_TASK_IDS)),--task-ids $(BVB_TASK_IDS),)

rewardkit-materialize: validate
	$(PYTHON) -m adapters.bvb.run_adapter --release-dir "$(BVB_RELEASE_DIR)" --verifier-kind rewardkit --output-dir "$(BVB_REWARDKIT_OUTPUT_DIR)" $(if $(strip $(BVB_TASK_IDS)),--task-ids $(BVB_TASK_IDS),)

smoke: validate
	rm -rf "$(SMOKE_DIR)"
	$(PYTHON) -m adapters.bvb.run_adapter --release-dir "$(BVB_RELEASE_DIR)" --output-dir "$(SMOKE_DIR)"
	$(PYTHON) scripts/smoke_bvb.py --release-dir "$(BVB_RELEASE_DIR)" --generated-dir "$(SMOKE_DIR)/$(BVB_COHORT)"

test:
	BVB_TEST_RELEASE="$(BVB_RELEASE_DIR)" $(PYTHON) -m unittest discover -s tests -t .
	$(PYTHON) verifier-eval/scripts/validate_scoring.py "$(BVB_RELEASE_DIR)"
	$(PYTHON) verifier-eval/scripts/validate_rewardkit_runner.py

score:
	@test -n "$(PREDICTIONS)" || (echo "PREDICTIONS=<file.jsonl> is required" >&2; exit 2)
	$(PYTHON) -m adapters.bvb.cli score --release-dir "$(BVB_RELEASE_DIR)" --predictions "$(PREDICTIONS)"

# Builds both local images and checks real MCP exports as the judge user;
# Docker only, no model calls.
tool-smoke: validate
	rm -rf "$(TOOL_SMOKE_DIR)"
	$(PYTHON) -m adapters.bvb.run_adapter --release-dir "$(BVB_RELEASE_DIR)" --output-dir "$(TOOL_SMOKE_DIR)/gandalf" --task-ids $(VERIFIER_SMOKE_TASK)
	$(PYTHON) -m adapters.bvb.run_adapter --release-dir "$(BVB_RELEASE_DIR)" --verifier-kind rewardkit --output-dir "$(TOOL_SMOKE_DIR)/rewardkit" --task-ids $(VERIFIER_SMOKE_TASK)
	$(PYTHON) scripts/smoke_bvb_container_tools.py --release-dir "$(BVB_RELEASE_DIR)" \
		--task-dir "$(TOOL_SMOKE_DIR)/gandalf/$(BVB_COHORT)/$(VERIFIER_SMOKE_TASK)" \
		--task-dir "$(TOOL_SMOKE_DIR)/rewardkit/$(BVB_COHORT)/$(VERIFIER_SMOKE_TASK)"

verifier-bootstrap:
	$(if $(strip $(HOST_SMOKE_VERIFIERS)),cd verifier-eval && ./scripts/bootstrap.sh $(HOST_SMOKE_VERIFIERS),@echo "No host-side verifier selected; nothing to bootstrap")

# One task in local Docker via Harbor; makes paid model calls.
gandalf-smoke: SMOKE_VERIFIERS := gandalf
gandalf-smoke: verifier-smoke

verifier-smoke: validate verifier-bootstrap
	rm -rf "$(VERIFIER_SMOKE_DIR)/gandalf"
	$(PYTHON) -m adapters.bvb.run_adapter --release-dir "$(BVB_RELEASE_DIR)" --output-dir "$(VERIFIER_SMOKE_DIR)/gandalf" --task-ids $(VERIFIER_SMOKE_TASK)
	@$(LOAD_ENV); \
	$(VERIFIER_RUNNER) stage --tasks "$(VERIFIER_SMOKE_TASK)" --staging-id "$(VERIFIER_SMOKE_RUN_ID)" --dataset-dir "$(VERIFIER_SMOKE_DIR)/gandalf/$(BVB_COHORT)" --n-concurrent 1 && \
	$(VERIFIER_RUNNER) evaluate --tasks "$(VERIFIER_SMOKE_TASK)" --staging-id "$(VERIFIER_SMOKE_RUN_ID)" --dataset-dir "$(VERIFIER_SMOKE_DIR)/gandalf/$(BVB_COHORT)" --run-id "$(VERIFIER_SMOKE_RUN_ID)" --verifier "$(SMOKE_VERIFIERS)" $(if $(HOST_SMOKE_MODELS),--model "$(HOST_SMOKE_MODELS)",) --no-score

# Optional compatibility check with RewardKit's Claude Code agent (ANTHROPIC_API_KEY).
# The measured RewardKit conditions use Codex with GPT models on Daytona instead.
rewardkit-claude-smoke: validate
	rm -rf "$(VERIFIER_SMOKE_DIR)/rewardkit"
	$(PYTHON) -m adapters.bvb.run_adapter --release-dir "$(BVB_RELEASE_DIR)" --verifier-kind rewardkit --output-dir "$(VERIFIER_SMOKE_DIR)/rewardkit" --task-ids $(VERIFIER_SMOKE_TASK)
	@$(LOAD_ENV); \
	$(VERIFIER_RUNNER) stage --verifier-kind rewardkit --tasks "$(VERIFIER_SMOKE_TASK)" --staging-id "$(REWARDKIT_SMOKE_RUN_ID)" --dataset-dir "$(VERIFIER_SMOKE_DIR)/rewardkit/$(BVB_COHORT)" --n-concurrent 1 && \
	$(VERIFIER_RUNNER) evaluate --verifier-kind rewardkit --tasks "$(VERIFIER_SMOKE_TASK)" --staging-id "$(REWARDKIT_SMOKE_RUN_ID)" --dataset-dir "$(VERIFIER_SMOKE_DIR)/rewardkit/$(BVB_COHORT)" --run-id "$(REWARDKIT_SMOKE_RUN_ID)" --verifier rewardkit --no-score

analysis:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required for strict score replay" >&2; exit 2)
	$(VERIFIER_RUNNER) score --run-id "$(RUN_ID)"

# Daytona plus provider keys for full runs; launch conditions with `uv run bvb run`.
daytona-snapshot: validate
	@$(LOAD_ENV); \
	$(PYTHON) -m adapters.bvb.daytona --release-dir "$(BVB_RELEASE_DIR)" --context-dir "$(DAYTONA_CONTEXT_DIR)" --snapshot-name "$(DAYTONA_SNAPSHOT)" --cache-max-bytes "$(DAYTONA_CACHE_MAX_BYTES)" --create

runtime-snapshots: validate
	@$(LOAD_ENV); \
	$(PYTHON) -m adapters.bvb.runtime_snapshots --release-dir "$(BVB_RELEASE_DIR)" --kind gandalf-llm --create && \
	$(PYTHON) -m adapters.bvb.runtime_snapshots --release-dir "$(BVB_RELEASE_DIR)" --kind rewardkit --create
