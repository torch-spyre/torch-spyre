# HELP
# This will output the help for each task
.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

PYTEST_ARGS ?= -v
TEST_CONFIGS ?= tests/configs/torch_spyre_tests

# TEST_TYPE selects which suite subset to run:
#   smoke            — fast sanity checks (~4 suites)
#   core             — all functional tests, excludes special-purpose hardware
#   device_critical  — device-layer surfaces flex and deeptools/dxp_standalone
#                       exercise most: streams, job launch plans, codegen,
#                       LX/scratchpad planning, tensor layout, allocator/GC,
#                       D2D copies (used as the default in integration-tests.yaml,
#                       triggered by those upstream repos)
#   full             — everything (core + LX-planning); default for `make tests`
#   suite_<group>    — all configs inside the <group>/ sub-directory
#                      (e.g. suite_inductor, suite_tensors)
#   <label>          — any arbitrary label defined in test_suite_config.labels
# Empty / unset defaults to "full" (all configs under TEST_CONFIGS).
TEST_TYPE ?= full

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
tests: ## Run torch spyre tests. Narrow scope with TEST_TYPE=smoke|core|full|suite_<group>. TEST_CONFIGS may point at a config directory (filtered by TEST_TYPE) or a single config yaml file (run directly).
ifneq ($(wildcard $(TEST_CONFIGS)/.),)
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
