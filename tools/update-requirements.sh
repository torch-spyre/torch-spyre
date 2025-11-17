#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd ${ROOT_DIR}

uv pip compile pyproject.toml --emit-index-url > requirements/run.txt
uv pip compile pyproject.toml --emit-index-url --extra lint --no-emit-package torch > requirements/lint.txt
uv pip compile pyproject.toml --emit-index-url --extra lint --extra build > requirements/build.txt
uv pip compile pyproject.toml --emit-index-url --extra lint --extra build --extra test > requirements/all.txt
uv pip compile pyproject.toml --emit-index-url --extra lint --extra build --extra test --no-emit-package torch > requirements/dev.txt

