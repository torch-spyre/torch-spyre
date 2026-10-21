#!/bin/bash
# Install mlir_ktdp (ktir-mlir-frontend) into the active environment.
#
# The OpSpec->KTIR emitter imports mlir_ktdp lazily, so torch-spyre builds, installs
# and tests without it -- only the golden-MLIR tests and TORCH_SPYRE_KTIR=1 need it.
# It is therefore NOT declared in pyproject.toml: see tools/ktir-mlir-frontend.pin
# for why, and edit that file to bump the commit.
#
# It cannot be a plain `pip install git+...`: ktir-mlir-frontend is a
# scikit-build-core/CMake project that must configure against a matching LLVM/MLIR,
# which its own scripts/setup_mlir.py downloads and caches.  Hence a script rather
# than a dependency entry.
#
#   tools/install-ktir.sh                 # clone into a temp dir, build, install
#   tools/install-ktir.sh ~/src/ktir      # reuse/refresh a checkout you keep
#
# Needs GIT_PAT or GITHUB_TOKEN on the first run only, for the LLVM artifact.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_FILE="${ROOT_DIR}/tools/ktir-mlir-frontend.pin"
REPO="https://github.com/torch-spyre/ktir-mlir-frontend.git"

# The pin file is commentary plus one SHA; take the first non-comment line.
REV="$(grep -vE '^\s*(#|$)' "$PIN_FILE" | head -1 | tr -d '[:space:]')"
if [[ -z "${REV}" ]]; then
    echo "error: no commit found in ${PIN_FILE}" >&2
    exit 1
fi

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo "error: mlir_ktdp needs Python >= 3.12; this environment is $(python -V)" >&2
    exit 1
fi

# A caller-supplied checkout is reused so the LLVM cache and CMake build survive
# across bumps; without one, a temp clone is made and removed.
if [[ $# -ge 1 ]]; then
    SRC="$1"
    mkdir -p "$(dirname "${SRC}")"
    if [[ -d "${SRC}/.git" ]]; then
        git -C "${SRC}" fetch --quiet origin
    else
        git clone --quiet "${REPO}" "${SRC}"
    fi
else
    SRC="$(mktemp -d)"
    trap 'rm -rf "${SRC}"' EXIT
    git clone --quiet "${REPO}" "${SRC}"
fi

git -C "${SRC}" checkout --quiet "${REV}"
echo "ktir-mlir-frontend at $(git -C "${SRC}" rev-parse --short HEAD)"

# Downloads the LLVM artifact pinned in cmake/llvm-hash.txt (cached under
# ~/.cache/ktir-mlir), or reuses an existing build if MLIR_DIR is already set.
if [[ -z "${MLIR_DIR:-}" ]]; then
    MLIR_DIR="$(cd "${SRC}" && python scripts/setup_mlir.py)"
fi
echo "MLIR_DIR=${MLIR_DIR}"

CMAKE_ARGS="-DMLIR_DIR=${MLIR_DIR}" python -m pip install "${SRC}"

# The emitter needs these specific dialect bindings, not just the package.
python -c "from mlir_ktdp.dialects import arith, func, ktdp, linalg, scf, tensor" \
    && echo "✅ mlir_ktdp installed; dialect bindings importable"
