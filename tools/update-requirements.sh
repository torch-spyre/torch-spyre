#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd ${ROOT_DIR}

# Refresh the uv.lock file, based on pyproject.toml
uv lock

# Regenerate requirement files for convenience
UV_EXPORT_OPTIONS="--format requirements.txt --no-hashes --no-install-project"

uv export $UV_EXPORT_OPTIONS > requirements/run.txt
uv export $UV_EXPORT_OPTIONS --extra lint --no-emit-package torch > requirements/lint.txt
uv export $UV_EXPORT_OPTIONS --extra lint --extra build > requirements/build.txt
uv export $UV_EXPORT_OPTIONS --all-extras > requirements/dev.txt
