#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW_REQUIREMENTS=$(mktemp)

cleanup() {
    rm ${ROOT_DIR}/requirements/constraints.txt || true
}

trap cleanup EXIT SIGINT SIGTERM

cd ${ROOT_DIR}

cp requirements/dev.txt requirements/constraints.txt
COMPILE_OPTIONS="--emit-index-url --build-constraints requirements/constraints.txt"

uv pip compile pyproject.toml $COMPILE_OPTIONS --all-extras > "${NEW_REQUIREMENTS}"

if ! diff -q "requirements/dev.txt" "${NEW_REQUIREMENTS}" > /dev/null 2>&1; then
    {
        echo "⚠️  WARNING: requirements/dev.txt is out of sync with pyproject.toml"
        echo ""
        echo "    This is due to a change in the PR"
        echo "    This diff shows the missing update:"
        echo ""
        echo "    $(diff requirements/dev.txt ${NEW_REQUIREMENTS})"
        echo ""
        echo "    To update the requirement files, run:"
        echo "    ./tools/update-requirements.sh"
        echo ""
    } >&2
    exit 1
fi
