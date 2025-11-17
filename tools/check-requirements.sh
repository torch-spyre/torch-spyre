#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW_REQUIREMENTS=$(mktemp)

cd ${ROOT_DIR}
uv pip compile pyproject.toml --emit-index-url --extra lint --extra build --extra test --no-emit-package torch > "${NEW_REQUIREMENTS}"

if ! diff -q "requirements/dev.txt" "${NEW_REQUIREMENTS}" > /dev/null 2>&1; then
    {
        echo "⚠️  WARNING: requirements/dev.txt may be out of sync with pyproject.toml"
        echo ""
        echo "    This could be due to a change in the PR or to a new patch version of a library being available"
        echo "    This diff shows the potentially missing update:"
        echo ""
        echo "    $(diff requirements/dev.txt ${NEW_REQUIREMENTS})"
        echo ""
        echo "    If you would like to update requirements/dev.txt, run:"
        echo "    ./tools/update-requirements.sh"
        echo ""
        echo "    If version change is not related to this PR and you prefer to skip the update for now,"
        echo "    please pin the full package version (including patch version) in the pyproject.toml."
        echo ""
    } >&2
    exit 1
fi
