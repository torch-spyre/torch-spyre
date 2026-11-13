# Thin wrappers over the spyre-bench CLI. Settings are BENCH_* variables the CLI also reads from the
# environment; nothing here depends on the architecture, card count or queue. Advisory only.
#   make bench-validate | bench-list | bench-explain CASE=... | bench-plan | bench-run | bench-smoke | bench-report

BENCH ?= spyre-bench
BENCH_DIR ?= benchmarks
BENCH_CENTRAL ?=
BENCH_BASE_DIR ?=
BENCH_REQUIRED ?=
BENCH_PURPOSE ?= on-demand
BENCH_ENVIRONMENT ?= local
BENCH_REUSE ?= never
BENCH_INGEST ?= 0
BENCH_ARCH ?=
BENCH_CARDS ?=
BENCH_QUEUE ?=
BENCH_SELECTION ?=
BENCH_CASES ?=
BENCH_TIER ?=
BENCH_SUBJECT_ID ?=
BENCH_PLAN ?=
BENCH_SET ?=
BENCH_QUIET ?= $(if $(CI),1,0)
BENCH_ENGINE_REF ?= main
BENCH_RESULTS_DIR ?= $(if $(RESULTS_DIR),$(RESULTS_DIR),results)/bench
CASE ?=

export BENCH_PURPOSE BENCH_ENVIRONMENT BENCH_SELECTION BENCH_CASES BENCH_TIER BENCH_SUBJECT_ID BENCH_REUSE \
	BENCH_PLAN BENCH_SET BENCH_INGEST BENCH_ARCH BENCH_CARDS BENCH_QUEUE BENCH_QUIET

# Install hint: the package lives in torch-spyre#6298 until that PR is merged, then on main.
_bench_install = uv pip install 'spyre-bench @ git+https://github.com/torch-spyre/torch-spyre@$(BENCH_ENGINE_REF)\#subdirectory=extensions/spyre-bench'
_bench_central = $(if $(BENCH_CENTRAL),--central '$(BENCH_CENTRAL)')
# $(call _bench_if,VERB,COMMAND): run COMMAND if the engine has VERB; otherwise print a notice and
# exit 0, or exit 2 when BENCH_REQUIRED is set to any non-empty value (including 0).
_bench_if = if command -v $(BENCH) >/dev/null 2>&1 && $(BENCH) --help 2>&1 | grep -q '^  $(1) '; then \
		$(2); \
	else \
		echo "bench: '$(1)' skipped: '$(BENCH)' is not installed or has no such verb ($(_bench_install))" >&2; \
		test -z "$(BENCH_REQUIRED)"; \
	fi

.PHONY: bench-validate bench-list bench-explain bench-plan bench-run bench-smoke bench-report

bench-validate: ## Validate benchmarks/ (BENCH_BASE_DIR=<dir holding benchmarks/ at the merge base> adds the identity diff); skipped with a notice when the engine lacks it unless BENCH_REQUIRED is set.
	@$(call _bench_if,validate,$(BENCH) validate '$(BENCH_DIR)' $(_bench_central)$(if $(BENCH_BASE_DIR), --base-dir '$(BENCH_BASE_DIR)'))

bench-list: ## List cases and Selections of benchmarks/. Hardware-free.
	@$(call _bench_if,list,$(BENCH) list --catalog '$(BENCH_DIR)' $(_bench_central))

bench-explain: ## Show one case with its ids and resolved request: make bench-explain CASE=<name>.
	@test -n "$(CASE)" || { echo "usage: make bench-explain CASE=<case name>" >&2; exit 2; }
	@$(call _bench_if,explain,$(BENCH) explain '$(CASE)' --catalog '$(BENCH_DIR)' $(_bench_central))

bench-plan: ## Write plan.json under BENCH_RESULTS_DIR from the BENCH_* request. Hardware-free.
	@$(call _bench_if,plan,mkdir -p '$(BENCH_RESULTS_DIR)' && $(BENCH) plan --catalog '$(BENCH_DIR)' $(_bench_central) --out '$(BENCH_RESULTS_DIR)/plan.json')

bench-run: ## Run the request (or BENCH_PLAN=plan.json) on the devices the job holds; BENCH_INGEST=1 ingests.
	@$(call _bench_if,run,mkdir -p '$(BENCH_RESULTS_DIR)' && $(BENCH) run --catalog '$(BENCH_DIR)' $(_bench_central) --out '$(BENCH_RESULTS_DIR)')

bench-smoke: ## Selection 'smoke': one case, one repeat, never ingested.
	@$(MAKE) --no-print-directory bench-run BENCH_SELECTION=smoke BENCH_CASES= BENCH_TIER= BENCH_PURPOSE=on-demand \
		BENCH_REUSE=never BENCH_SET=run.repeats=1 BENCH_INGEST=0

bench-report: ## Summarise BENCH_RESULTS_DIR (counts and validity; no values when BENCH_QUIET=1).
	@$(call _bench_if,report,$(BENCH) report '$(BENCH_RESULTS_DIR)')
