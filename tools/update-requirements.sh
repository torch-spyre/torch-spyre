#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd ${ROOT_DIR}

uv pip compile pyproject.toml --extra test --extra lint > requirements/dev.txt
uv pip compile pyproject.toml --extra lint > requirements/lint.txt
uv pip compile pyproject.toml --extra test > requirements/test.txt