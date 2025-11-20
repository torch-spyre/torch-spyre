#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cleanup() {
    rm ${ROOT_DIR}/requirements/constraints.txt || true
}

trap cleanup EXIT SIGINT SIGTERM

cd ${ROOT_DIR}

cp requirements/dev.txt requirements/constraints.txt
COMPILE_OPTIONS="--emit-index-url --build-constraints requirements/constraints.txt"

uv pip compile pyproject.toml $COMPILE_OPTIONS > requirements/run.txt
uv pip compile pyproject.toml $COMPILE_OPTIONS --extra lint --no-emit-package torch > requirements/lint.txt
uv pip compile pyproject.toml $COMPILE_OPTIONS --extra lint --extra build > requirements/build.txt
uv pip compile pyproject.toml $COMPILE_OPTIONS --all-extras > requirements/dev.txt
