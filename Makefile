# HELP
# This will output the help for each task
.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

PYTEST_ARGS ?= -v
TEST_CONFIGS ?= tests/configs/torch_spyre_tests

# TEST_TYPE selects which suite subset to run. These tier names ARE the
# test_suite_config.labels vocabulary directly -- there is no alias layer,
# and a config only runs under a tier if it explicitly carries that label
# (configs with no labels field run under nothing):
#   smoke            — fast sanity checks (~4 suites)
#   unit             — all functional tests, excludes special-purpose hardware
#   integration      — device-layer surfaces flex and deeptools/dxp_standalone
#                       exercise most: streams, job launch plans, codegen,
#                       LX/scratchpad planning, tensor layout, allocator/GC,
#                       D2D copies (used as the default in integration-tests.yaml,
#                       triggered by those upstream repos)
#   regression       — everything (unit + LX-planning) under TEST_CONFIGS;
#                      default for `make tests`
#   trunk            — everything torch-spyre's four push-to-main workflows
#                      cover across tests/configs/ (torch_spyre_tests,
#                      distributed_tests, model_ops_tests, upstream_tests,
#                      upstream_tests_beta), not just TEST_CONFIGS -- so
#                      `make tests TEST_TYPE=trunk` matches what actually
#                      runs on a push to main. Filtered by the trunk label,
#                      same as every other tier -- no directory list to
#                      maintain here.
#   perf             — spyre-perf-suite benchmark (shells out, not a pytest
#                      config suite); writes report.xml into RESULTS_DIR
#   suite_<group>    — all configs inside the <group>/ sub-directory
#                      (e.g. suite_inductor, suite_tensors)
#   <label>          — any arbitrary label defined in test_suite_config.labels
#
# Empty / unset defaults to "regression" (all configs under TEST_CONFIGS
# labeled for full functional coverage).
TEST_TYPE ?= regression

# Where TEST_TYPE=perf writes its benchmark report. Flat /tmp/results so the CI
# ClickHouse push step (ingest_xml.py globs *.xml non-recursively) finds it
# alongside every other suite's JUnit XML, with no per-suite subdirectory.
RESULTS_DIR ?= /tmp/results

# Path to the OOT config checker script (relative to repo root)
CHECK_SCRIPT  := tests/scripts/check_oot_configs.py

# Path to the config filter script (relative to repo root)
FILTER_SCRIPT := tests/oot_framework/utils/filter_configs.py

# Config directory to scan (override to narrow/broaden the scope)
CHECK_CONFIGS ?= tests/configs/torch_spyre_tests

# Optional: scope checks to one test file. Unset = auto-discover all.
TEST_FILE ?=

# Internal: only pass --test-file when TEST_FILE is set
_TEST_FILE_ARG := $(if $(TEST_FILE),--test-file $(TEST_FILE),)

# ---------------------------------------------------------------------------
# Developer tooling
# ---------------------------------------------------------------------------

.PHONY: setup
setup: ## Reinstall torch-spyre into the active venv (uv sync --all-extras --reinstall-package torch-spyre)
	uv sync --all-extras --active --inexact --reinstall-package torch-spyre -v

.PHONY: precommit
precommit: ## Run all pre-commit hooks against every file
	pre-commit run --all-files

# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

.PHONY: tests
tests: ## Run torch spyre tests. Narrow scope with TEST_TYPE=smoke|unit|integration|regression|trunk|perf|suite_<group>. TEST_CONFIGS may point at a config directory (filtered by TEST_TYPE) or a single config yaml file (run directly); ignored when TEST_TYPE=trunk (scans tests/configs/ directly, filtered by the trunk label).
# TEST_TYPE=perf is a benchmark mode, not a pytest-config suite: it does not
# run the OOT config machinery below. It shells out to the installed
# spyre-perf-suite console script (a wheel dependency of the dev image) and
# writes report.xml into RESULTS_DIR. Keeping it a mode of `tests` lets CI call
# it through the same `make tests TEST_TYPE=...` entry point as every other
# suite, so no new Makefile target or Jenkins wiring is needed.
ifeq ($(TEST_TYPE),perf)
	@mkdir -p "$(RESULTS_DIR)"
	spyre-perf-suite --no-experimental --stacks torch-spyre \
		--report "$(RESULTS_DIR)/report.txt"
	@test -f "$(RESULTS_DIR)/report.xml" || \
		{ echo "ERROR: spyre-perf-suite did not emit $(RESULTS_DIR)/report.xml" >&2; \
		  exit 1; }
else ifeq ($(TEST_TYPE),trunk)
	$(eval _PATHS := $(shell python3 $(FILTER_SCRIPT) --config-dir tests/configs --test-type trunk --format paths))
	@if [ -z "$(_PATHS)" ]; then \
		echo "ERROR: no configs matched TEST_TYPE=trunk under tests/configs" >&2; \
		exit 1; \
	fi
	@TORCH_SPYRE_TEST_TYPE="$(TEST_TYPE)" bash tests/run_test.sh $(_PATHS) $(PYTEST_ARGS)
else ifneq ($(wildcard $(TEST_CONFIGS)/.),)
	$(eval _PATHS := $(shell python3 $(FILTER_SCRIPT) \
		--config-dir $(TEST_CONFIGS) \
		--test-type "$(TEST_TYPE)" \
		--format paths))
	@if [ -z "$(_PATHS)" ]; then \
		echo "ERROR: no configs matched TEST_TYPE=$(TEST_TYPE) under $(TEST_CONFIGS)" >&2; \
		exit 1; \
	fi
	@TORCH_SPYRE_TEST_TYPE="$(TEST_TYPE)" bash tests/run_test.sh $(_PATHS) $(PYTEST_ARGS)
else
	@if [ ! -f "$(TEST_CONFIGS)" ]; then \
		echo "ERROR: TEST_CONFIGS not found (expected a directory or a config file): $(TEST_CONFIGS)" >&2; \
		exit 1; \
	fi
	@TORCH_SPYRE_TEST_TYPE="$(TEST_TYPE)" bash tests/run_test.sh $(TEST_CONFIGS) $(PYTEST_ARGS)
endif


# ---------------------------------------------------------------------------
# OOT config checks (duplicates + missing + dead patterns)
# ---------------------------------------------------------------------------
 
.PHONY: check-all-configs
check-all-configs: ## Check OOT configs for duplicates, missing tests, and dead patterns. Oveeride with make check-all-configs TEST_FILE=tests/test_launch_jobplan.py for specific test file
	@python $(CHECK_SCRIPT) --config-dir $(CHECK_CONFIGS) $(_TEST_FILE_ARG)
 

.PHONY: clean
clean: ## Remove auto-generated OOT wrappers, conftest files, merged configs, and __pycache__ under tests/
	@find tests/ -name '*__oot_wrapper.py' -delete
	@find tests/ -name '__oot_conftest_*.py' -delete
	@find tests/ -name '_oot_merged_config_*.yaml' -delete
	@find tests/ -name '_spyre_merged_config_*.yaml' -delete
	@find tests/ -name '*.markers.json' -delete
	@find tests/ -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf torch_spyre.egg-info
	@rm -rf tests/oot_framework/oot_framework.egg-info
	@echo "Cleaned auto-generated files under tests/"
