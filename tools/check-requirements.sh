#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TMP_REQUIREMENT_DIR=$(mktemp -d)

cleanup() {
    rm -rf "${TMP_REQUIREMENT_DIR}" &> /dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

cd "$ROOT_DIR"

echo "🔍 Checking requirements files..."

# Check that uv.lock is up to date with pyproject.toml
echo ""
echo "📦 Checking uv.lock..."
if ! uv lock --check; then
    echo ""
    echo "❌ uv.lock is out of sync with pyproject.toml"
    echo ""
    echo "Dry-run:"
    uv lock --dry-run
    echo ""
    echo "To fix, run: ./tools/update-requirements.sh"
    exit 1
fi
echo "  ✅ uv.lock is in sync with pyproject.toml"

# Regenerate requirement files and check they match committed versions
echo ""
echo "📦 Checking requirements/*.txt files..."

UV_EXPORT_OPTIONS="--format requirements.txt --no-hashes --no-install-project"

check_requirements_file() {
    local export_args="$1"
    local file="$2"
    local name="$3"

    echo "  Checking $name..."

    # Generate to temp file
    uv export $UV_EXPORT_OPTIONS $export_args > "${TMP_REQUIREMENT_DIR}/${name}"

    # Compare with committed version
    if ! diff -q "$file" "${TMP_REQUIREMENT_DIR}/${name}" > /dev/null 2>&1; then
        echo ""
        echo "    ❌ $file is out of sync"
        echo ""
        echo "    Differences:"
        diff "$file" "${TMP_REQUIREMENT_DIR}/${name}" || true
        echo ""
        return 1
    fi
}

ERRORS=0

check_requirements_file "" "requirements/run.txt" "run" || ERRORS=$((ERRORS + 1))
check_requirements_file "--extra lint --no-emit-package torch" "requirements/lint.txt" "lint" || ERRORS=$((ERRORS + 1))
check_requirements_file "--extra lint --extra build" "requirements/build.txt" "build" || ERRORS=$((ERRORS + 1))
check_requirements_file "--all-extras" "requirements/dev.txt" "dev" || ERRORS=$((ERRORS + 1))

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "❌ $ERRORS requirements file(s) out of sync"
    echo ""
    echo "To fix, run: ./tools/update-requirements.sh"
    exit 1
fi

echo ""
echo "✅ All requirements files are in sync"