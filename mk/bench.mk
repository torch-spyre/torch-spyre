# Thin wrappers over the spyre-bench CLI. Settings are BENCH_* variables the CLI also reads from the
# environment; nothing here depends on the architecture, card count or queue. Advisory only.
#   make bench-validate | bench-list | bench-explain CASE=... | bench-plan | bench-run | bench-verdict
#        bench-smoke | bench-report | bench-nightly-smoke

BENCH ?= spyre-bench
BENCH_DIR ?= benchmarks
BENCH_CENTRAL ?=
BENCH_BASE ?=
BENCH_BASE_DIR ?=
BENCH_REQUIRED ?=
BENCH_OPTIONAL ?=
BENCH_PURPOSE ?= on-demand
BENCH_ENVIRONMENT ?= one-device
BENCH_REUSE ?= never
BENCH_INGEST ?= 0
BENCH_INVOKER ?= shell
BENCH_ARCH ?=
BENCH_CARDS ?=
BENCH_QUEUE ?=
BENCH_SELECTION ?=
BENCH_CASES ?=
BENCH_TIER ?=
BENCH_SUITES ?=
BENCH_SUBJECT ?=
BENCH_SUBJECT_ID ?=
BENCH_PLAN ?=
BENCH_SET ?=
BENCH_CLEAN ?=
BENCH_MAX_WAIT ?=
BENCH_MAX_HOLD ?=
BENCH_LEASE_FILE ?=
BENCH_RUN_ID ?=
BENCH_PREFIX ?=
BENCH_QUIET ?= $(if $(CI),1,0)
BENCH_ENGINE_REF ?= main
BENCH_RESULTS_DIR ?= bench-results
BENCH_PLAN_FILE ?= $(BENCH_RESULTS_DIR)/plan/plan.json
BENCH_STUB ?= python3 $(BENCH_DIR)/tools/bench_stub.py
CASE ?=

export BENCH_PURPOSE BENCH_ENVIRONMENT BENCH_SELECTION BENCH_CASES BENCH_TIER BENCH_SUITES BENCH_SUBJECT \
	BENCH_SUBJECT_ID BENCH_REUSE BENCH_PLAN BENCH_SET BENCH_INGEST BENCH_INVOKER BENCH_ARCH BENCH_CARDS \
	BENCH_QUEUE BENCH_CLEAN BENCH_MAX_WAIT BENCH_MAX_HOLD BENCH_LEASE_FILE BENCH_RUN_ID BENCH_QUIET

_bench_install = uv pip install 'spyre-bench @ git+https://github.com/torch-spyre/torch-spyre@$(BENCH_ENGINE_REF)\#subdirectory=extensions/spyre-bench'
_bench_central = $(if $(BENCH_CENTRAL),--central '$(BENCH_CENTRAL)')
# Soft mode (BENCH_OPTIONAL non-empty, BENCH_REQUIRED empty) is for callers that must not fail without the engine.
_bench_soft = $(if $(BENCH_REQUIRED),,$(BENCH_OPTIONAL))
# $(call _bench_if,VERB,COMMAND): run COMMAND if the engine has VERB; otherwise say how to install it and exit 2,
# or exit 0 in soft mode.
_bench_if = if command -v $(firstword $(BENCH)) >/dev/null 2>&1 && $(BENCH) --help 2>&1 | grep -q '^  $(1) '; then \
		$(2); \
	else \
		echo "bench: '$(1)' skipped: '$(BENCH)' is not installed or has no such verb. Install spyre-bench: $(_bench_install)" >&2; \
		test -n "$(_bench_soft)" || { echo "bench: set BENCH_OPTIONAL=1 to skip silently when the engine is absent" >&2; exit 2; }; \
	fi
_bench_plan_cmd = $(BENCH) plan --catalog '$(BENCH_DIR)' $(_bench_central) --out '$(BENCH_PLAN_FILE)'

.PHONY: bench-validate bench-list bench-explain bench-plan bench-run bench-verdict bench-smoke bench-report \
	bench-nightly-smoke

bench-validate: ## Validate benchmarks/ (BENCH_BASE=<ref> or BENCH_BASE_DIR=<dir holding benchmarks/ at the merge base> adds the identity diff); skipped with a notice when the engine lacks it unless BENCH_REQUIRED is set.
	@$(call _bench_if,validate,$(BENCH) validate '$(BENCH_DIR)' $(_bench_central)$(if $(BENCH_BASE), --base-ref '$(BENCH_BASE)')$(if $(BENCH_BASE_DIR), --base-dir '$(BENCH_BASE_DIR)'))

bench-list: ## List cases and Selections of benchmarks/. Hardware-free.
	@$(call _bench_if,list,$(BENCH) list --catalog '$(BENCH_DIR)' $(_bench_central))

bench-explain: ## Show one case with its ids and resolved request: make bench-explain CASE=<name>.
	@test -n "$(CASE)" || { echo "usage: make bench-explain CASE=<case name>" >&2; exit 2; }
	@$(call _bench_if,explain,$(BENCH) explain '$(CASE)' --catalog '$(BENCH_DIR)' $(_bench_central))

bench-plan: ## Write the plan to BENCH_PLAN_FILE (default under BENCH_RESULTS_DIR) from the BENCH_* request. Hardware-free.
	@$(call _bench_if,plan,mkdir -p '$(dir $(BENCH_PLAN_FILE))' && $(_bench_plan_cmd))

bench-run: ## Plan (unless BENCH_PLAN=plan.json), then run on the devices the job holds; BENCH_INGEST=1 ingests.
	@$(call _bench_if,run,$(if $(BENCH_PLAN),:,mkdir -p '$(dir $(BENCH_PLAN_FILE))' && $(_bench_plan_cmd)) && mkdir -p '$(BENCH_RESULTS_DIR)' && $(BENCH_PREFIX) $(BENCH) run --catalog '$(BENCH_DIR)' $(_bench_central) --plan '$(or $(BENCH_PLAN),$(BENCH_PLAN_FILE))' --out '$(BENCH_RESULTS_DIR)')

bench-verdict: ## Verdict words for the run (BENCH_RUN_ID, else run.json stored_run_ids); fails when no stored id is known or no rows exist.
	@$(call _bench_if,verdict,id='$(BENCH_RUN_ID)'; [ -n "$$id" ] || id=$$(python3 $(BENCH_DIR)/tools/stored_run_id.py '$(BENCH_RESULTS_DIR)') || exit 2; $(BENCH) verdict --run-id "$$id" --json > '$(BENCH_RESULTS_DIR)/verdict.json' && python3 $(BENCH_DIR)/tools/stored_run_id.py --check-verdict '$(BENCH_RESULTS_DIR)/verdict.json')

bench-smoke: ## Selection 'smoke': one case, one repeat, never ingested.
	@$(MAKE) --no-print-directory bench-run BENCH_SELECTION=smoke BENCH_CASES= BENCH_TIER= BENCH_PURPOSE=on-demand \
		BENCH_REUSE=never BENCH_SET=run.repeats=1 BENCH_INGEST=0

bench-report: ## Summarise BENCH_RESULTS_DIR (counts and validity; no values when BENCH_QUIET=1).
	@$(call _bench_if,report,$(BENCH) report '$(BENCH_RESULTS_DIR)')

bench-nightly-smoke: ## The nightly's plan -> run -> verdict -> report with a card-free stub; nothing is measured or ingested.
	@$(MAKE) --no-print-directory bench-run bench-verdict bench-report BENCH='$(BENCH_STUB)' BENCH_REQUIRED=1 \
		BENCH_INGEST=0 BENCH_PREFIX= BENCH_RUN_ID=stub BENCH_PURPOSE=nightly BENCH_SELECTION=$(or $(BENCH_SELECTION),ops-core)
