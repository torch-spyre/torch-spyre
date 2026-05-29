# HELP
# This will output the help for each task
.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

PYTEST_ARGS ?= -v
TEST_CONFIGS ?= tests/configs/torch_spyre_tests

# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

.PHONY: tests
tests: ## Run all torch spyre tests by default (except distributed). Override with: make tests TEST_CONFIGS="tests/configs/..."
	@bash tests/run_test.sh $(TEST_CONFIGS) $(PYTEST_ARGS)

.PHONY: clean
clean: ## Remove auto-generated OOT wrappers, conftest files, merged configs, and __pycache__ under tests/
	@find tests/ -name '*__oot_wrapper.py' -delete
	@find tests/ -name '__oot_conftest_*.py' -delete
	@find tests/ -name '_oot_merged_config_*.yaml' -delete
	@find tests/ -name '_spyre_merged_config_*.yaml' -delete
	@find tests/ -name '*.markers.json' -delete
	@find tests/ -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned auto-generated files under tests/"
