#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd ${ROOT_DIR}

uv pip compile pyproject.toml --extra test --extra lint --emit-index-url --no-emit-package torch > requirements/dev.txt
uv pip compile pyproject.toml --extra test --extra lint --emit-index-url > requirements/all.txt
