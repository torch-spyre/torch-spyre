# HELP
# This will output the help for each task
.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

PYTEST_ARGS ?= -v
TEST_CONFIGS ?= tests/configs/module_tests tests/configs/torch_spyre_tests

# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

.PHONY: tests
tests: ## Run all torch spyre tests + module tests by default. Override with: make tests TEST_CONFIGS="tests/configs/torch_spyre_tests/inductor"
	@bash tests/run_test.sh $(TEST_CONFIGS) $(PYTEST_ARGS)