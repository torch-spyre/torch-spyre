---
name: upgrade-pytorch-version
description: "Upgrade PyTorch dependency version across torch-spyre and torch-spyre-docs. Updates pyproject.toml, CI workflow comments, documentation, and build scripts; rebuilds downstream C++ extensions (vllm etc.). Use when asked to bump, upgrade, or update the PyTorch version dependency. Accepts target version as argument (e.g., 'upgrade-pytorch-version 2.13')."
---

# Upgrading PyTorch Version in torch-spyre

This skill walks through upgrading the pinned PyTorch version across the
torch-spyre ecosystem. It covers all files that reference the version, in
the correct order.

The user specifies the **target version** (e.g., `2.13`). The current
version is auto-detected from `pyproject.toml`. Skipping versions is
fully supported — the upgrade goes directly from whatever is current to
the specified target.

> **Scope:** This skill covers the torch-spyre repo and its companion
> torch-spyre-docs repo. The PyTorch source tree itself must already be
> checked out and built at the target version before running this procedure.

---

## Determining OLD and NEW

**NEW** (target version): Provided by the user as the skill argument.
If no argument is given, ask the user which PyTorch version to target.

**OLD** (current version): Auto-detect from `pyproject.toml`:
```bash
grep 'torch~=' $TORCH_SPYRE_DIR/pyproject.toml | head -1 | sed 's/.*torch~=\([0-9.]*\).*/\1/' | cut -d. -f1,2
```

Example: if pyproject.toml has `torch~=2.11.0` and the user says `2.13`,
then `OLD=2.11` and `NEW=2.13`. We skip `2.12` entirely — that's fine.

**Important:** `OLD` and `NEW` refer to major.minor only (e.g., `2.11`).
The full constraint uses `$OLD.0` / `$NEW.0` (e.g., `2.11.0`).

---

## Inputs

| Variable | How to determine |
|----------|-----------------|
| `NEW` | User-provided argument (e.g., `2.13`) |
| `OLD` | Auto-detect from `pyproject.toml` (current `torch~=X.Y.0`) |
| `TORCH_SPYRE_DIR` | `$DTI_PROJECT_ROOT/torch-spyre` (or locate via git root) |
| `DOCS_DIR` | `$DTI_PROJECT_ROOT/torch-spyre-docs` |
| `PYTORCH_DIR` | `$DTI_PROJECT_ROOT/pytorch` |

If `DTI_PROJECT_ROOT` is not set, look for the torch-spyre checkout by
finding the repo root of the current working directory or check common
locations (`~/dt-inductor`, `~/torch-spyre`).

---

## Prerequisites

Before making changes, verify:

1. **Target version is valid** — `release/$NEW` branch exists:
   ```bash
   cd $PYTORCH_DIR && git branch -r | grep "release/$NEW"
   ```

2. **PyTorch source is at target version:**
   ```bash
   cd $PYTORCH_DIR && git describe --tags
   ```
   Should show `v$NEW.0` or a release candidate thereof. If not, inform
   the user that `$PYTORCH_DIR` needs to be checked out and built at
   `release/$NEW` first.

3. **PyTorch was built from source** (libtorch, headers, libgomp shim):
   ```bash
   ls $PYTORCH_DIR/torch/lib/libtorch.so
   ls $PYTORCH_DIR/torch/lib/libgomp.so.1   # see Step 9 pitfall
   ```

4. **Version jump is intentional** — if `NEW` is more than one minor
   version ahead of `OLD` (e.g., `2.10` → `2.13`), confirm with the user:
   "Upgrading from $OLD to $NEW (skipping intermediate versions). Proceed?"

If any prerequisite is not met, stop and inform the user what is missing.

---

## Step 1: `pyproject.toml` — Version Dependencies

**File:** `$TORCH_SPYRE_DIR/pyproject.toml`

There are **three** sections that declare `torch` as a dependency. Each
has an active line and a commented alternative (used by the LOCAL_PYTORCH
build mode). Update all six lines:

```
# Active dependency (3 locations):
"torch~=$OLD.0"  →  "torch~=$NEW.0"

# Commented alternative (3 locations):
#    "torch>=$OLD.0"  →  #    "torch>=$NEW.0"
```

The three sections are:
- `[build-system] requires`
- `[project] dependencies`
- `[project.optional-dependencies] build`

**Important:** The commented `#    "torch>=..."` lines MUST exist and match
the version. They are toggled by `build-torch-spyre.sh` when building with
a local PyTorch source tree.

---

## Step 2: `pyproject.toml` — Comments

**File:** `$TORCH_SPYRE_DIR/pyproject.toml`

In the `[tool.pytest.ini_options] filterwarnings` section:

```
# PyTorch $OLD.0+: ...  →  # PyTorch $NEW.0+: ...
# https://github.com/pytorch/pytorch/blob/v$OLD.0/...  →  .../blob/v$NEW.0/...
```

Note: If an intermediate version was skipped, the comment URL may reference
a version older than `$OLD` (e.g., we're at 2.11 but the URL still says
v2.10.0 from a previous incomplete upgrade). Always replace whatever
version is currently there with `v$NEW.0`.

---

## Step 3: Eager-op Registrations (upstream/main and later)

As of upstream commit `faad75c` ("Simplify codegen (#1875)"), the
top-level `codegen/` directory was **removed** from torch-spyre. Eager
ATen ops are now registered by the hand-written module
`torch_spyre/ops/eager.py`, which iterates over op overloads dynamically
via `register_torch_compile_kernel()`. There is no longer any generated
`torch_spyre/codegen_ops.py`, no `codegen/inputs/Declarations.yaml`,
and no `codegen/inputs/RegistrationDeclarations.h` to keep in sync with
PyTorch.

**Action for modern upgrades:** no codegen-metadata step is needed.
Skip to the next step.

**When `eager.py` may need updates:** if PyTorch renames, adds, or
changes the signature of a specific ATen op that torch-spyre registers,
hand-edit `torch_spyre/ops/eager.py`. Read the upstream PyTorch release
notes or grep `aten::<name>` in
`$PYTORCH_DIR/aten/src/ATen/native/native_functions.yaml` to confirm.

### Legacy: PyTorch ≤ 2.10 codegen pipeline (for historical reference)

Before `faad75c`, the repo contained a top-level `codegen/` pipeline
that consumed PyTorch build outputs (`Declarations.yaml`,
`RegistrationDeclarations.h`) to generate `torch_spyre/codegen_ops.py`
at install time. The PT 2.11 upgrade on the pre-faad75c branch required
copying those two files from the PyTorch build tree into
`codegen/inputs/` and patching two bugs:

1. **`codegen/gen.py`** used a hardcoded `schemas[19:]` to skip the
   `RegistrationDeclarations.h` frontmatter. PT 2.11 shipped only 3
   frontmatter lines, silently dropping 16 real entries → `IndexError`
   in `template_tools.py:generate_replacements`. Fix: content-based
   filter (`if '"schema"' in line`) rather than line-count.
2. **`codegen/utils/template_tools.py`** mapped C++ types via
   substring replacement. New `aten::bmm.dtype` / `baddbmm.dtype` /
   `addmm.dtype` ops used `at::ScalarType`; `"Scalar"` substring-matched
   inside `"ScalarType"`, producing invalid `complexType` identifier
   → `NameError` at import. Fix: add `"ScalarType": "torch.dtype"`
   entry BEFORE `"Scalar"` in the ordered dict.

These pitfalls are now irrelevant on upstream/main but documented here
so anyone maintaining an older release branch knows what to look for.

---

## Step 4: `.github/workflows/upstream_tests.yaml` — Comments

**File:** `$TORCH_SPYRE_DIR/.github/workflows/upstream_tests.yaml`

The code logic in this workflow is **version-agnostic** (it reads
`pyproject.toml` dynamically). Only the **comments** contain hardcoded
version examples. Update them:

```yaml
# Old examples:
# pyproject.toml declares:  torch~=$OLD.0
# This step extracts "$OLD.0", strips the patch -> "$OLD", and builds
# the branch name "release/$OLD".
# Extract the version constraint, e.g. "$OLD.0" from "torch~=$OLD.0"
# version = torch_dep.removeprefix('torch~=')   # e.g. "$OLD.0"
# print('.'.join(version.split('.')[:2]))        # --> "$OLD"

# Replace all $OLD with $NEW in these comment lines.
```

Note: Search broadly for the OLD version — if a version was previously
skipped, some comments may still reference an even older version.

---

## Step 5: Internal Docs and Skills in torch-spyre

### `.claude/skills/project-overview/SKILL.md`

**File:** `$TORCH_SPYRE_DIR/.claude/skills/project-overview/SKILL.md`

Replace all occurrences:
```
torch~=$OLD.0  →  torch~=$NEW.0
```

### `docs/source/getting_started/installation.md`

**File:** `$TORCH_SPYRE_DIR/docs/source/getting_started/installation.md`

The requirements table has a PyTorch row:
```
| PyTorch | ~= $OLD.0 |
→
| PyTorch | ~= $NEW.0 |
```

### `.github/workflows/upstream_tests_beta.yaml` (if present)

Mirror the same comment-only updates as Step 4 (`upstream_tests.yaml`).
The "beta" workflow has the same dynamic-version-derivation logic with
its own copy of the `torch~=X.Y.0` example comments.

---

## Step 6: `torch-spyre-docs` Scripts

### `scripts/checkout-pytorch-src.sh`

**File:** `$DOCS_DIR/scripts/checkout-pytorch-src.sh`

```bash
# Comment (match any version number, not just $OLD):
# This script automates the initial checkout of PyTorch X.Y.
→
# This script automates the initial checkout of PyTorch $NEW.

# Branch:
git clone git@github.com:pytorch/pytorch.git -b release/X.Y
→
git clone git@github.com:pytorch/pytorch.git -b release/$NEW
```

### `scripts/build-torch-spyre.sh`

**File:** `$DOCS_DIR/scripts/build-torch-spyre.sh`

Update all sed patterns in the LOCAL_PYTORCH block. There are two groups:

1. **Forward substitution** (commenting out pinned, enabling flexible):
   ```
   torch~=$OLD.0  →  torch~=$NEW.0
   torch>=$OLD.0  →  torch>=$NEW.0
   ```

2. **Trap/revert** (restoring original state on exit):
   ```
   torch~=$OLD.0  →  torch~=$NEW.0
   torch>=$OLD.0  →  torch>=$NEW.0
   ```

**Important:** The `>=` version on BOTH sides of the trap sed must be
`$NEW.0` (not `$NEW` with a typo like `2.1.0`). A mismatch causes the
trap to silently mangle `pyproject.toml` on every `--local-pytorch`
build, leaving a bad commented version that gradually drifts from the
active dep. Diff the script carefully.

---

## Step 7: `torch-spyre-docs` Documentation

### `docs/dev_install.md`

**File:** `$DOCS_DIR/docs/dev_install.md`

```
## Install or Build PyTorch $OLD
→
## Install or Build PyTorch $NEW

...unmodified PyTorch $OLD, which is installed...
→
...unmodified PyTorch $NEW, which is installed...
```

### `docs/profiling_tools.md`

**File:** `$DOCS_DIR/docs/profiling_tools.md`

Update the kineto-spyre wheel URL and version references:
```
torch-$OLD.0  →  torch-$NEW.0
```

**Warning:** The kineto-spyre wheel for the new version may not yet be
published. If unsure, update the URL pattern but flag to the user that
they should verify the wheel exists at
`https://github.com/IBM/kineto-spyre/releases`.

---

## Step 8: Lock Files (Manual — Requires Network)

`uv.lock` and `requirements/*.txt` cannot be text-edited — they must be
regenerated. The repo ships `tools/update-requirements.sh` which does
both in the correct order with the exact `uv export` flags the project
uses (including `--no-emit-package torch` for `lint.txt`):

```bash
cd $TORCH_SPYRE_DIR
./tools/update-requirements.sh
```

Do not run `uv lock` / `uv export` commands by hand — the script is
the source of truth; running the commands yourself risks flag drift.

After running, commit `uv.lock` and the four `requirements/*.txt`
files. A pre-commit hook (`tools/check-requirements.sh`) will flag any
divergence.

**Prerequisite:** PyTorch `$NEW.0` wheels must be available on the CPU
index (`https://download.pytorch.org/whl/cpu`). If they are not yet
published, skip this step and inform the user — the active dep in
`pyproject.toml` is enough for `--local-pytorch` builds, lock
regeneration can wait for the wheel.

---

## Step 9: Rebuild Downstream C++ Extensions

**Any Python package with a compiled `*.so` that links against
`libtorch` / `libc10` must be rebuilt against PyTorch `$NEW`.** PyTorch
does not maintain C++ ABI stability across minor releases, so extensions
built for `$OLD` will fail to load with undefined-symbol errors against
`$NEW`.

Typical downstream extensions in a torch-spyre environment:

| Extension | Location | Rebuild command |
|-----------|----------|-----------------|
| vllm `_C.abi3.so` | `$VLLM_SRC/vllm/_C.abi3.so` | `cd $VLLM_SRC && uv pip install -e . --torch-backend=auto` |
| torch-spyre `_C.so` / `_hooks.so` | `$TORCH_SPYRE_DIR/torch_spyre/` | `$DOCS_DIR/scripts/build-torch-spyre.sh` |
| torchvision, torchaudio, etc. | venv site-packages | Reinstall from source or matching 2.11 wheel |

**Symptom of a stale extension:**
```
ImportError: .../extension.so: undefined symbol: _ZN3c10...
# or at op lookup time:
AttributeError: '_OpNamespace' '_C' object has no attribute 'gelu_fast'
```

The undefined symbol almost always starts with `_ZN3c10` (mangled
`c10::...`) or `_ZN5torch...` (mangled `torch::...`). If you see this
pattern, the extension that provides that symbol was built against a
different libtorch.

**Diagnostic** — list extensions with unresolved symbols:
```bash
for so in $(find $VENV/lib/python*/site-packages -name "*.so" 2>/dev/null); do
    sym=$(nm -D --undefined-only "$so" 2>/dev/null | grep -E "_ZN3c10|_ZN5torch" | head -1)
    [ -n "$sym" ] && echo "$so: $sym"
done
```

Loading any one of them against the new libtorch will fail if the
symbol is absent in `$NEW`.

### Known Pitfall: vllm CPU build fails with `Could not find OPEN_MP using the following names: gomp`

`vllm/cmake/cpu_extension.cmake` deliberately uses `NO_DEFAULT_PATH` and
only searches for libgomp inside PyTorch's lib directory
(`site-packages/torch.libs/libgomp-<hash>.so.1.0.0` or
`site-packages/torch/lib/libgomp*.so*`). This is fine for binary
PyTorch wheels (which bundle libgomp) but **breaks with a source-built
PyTorch**, which does not ship libgomp — it expects to link against the
system toolchain's libgomp.

**Symptom:**
```
CMake Error at cmake/cpu_extension.cmake:38 (find_library):
  Could not find OPEN_MP using the following names: gomp
```

**Fix:** symlink the system libgomp into PyTorch's lib dir so the
CMake glob `libgomp*.so*` finds it. **Point at the actual runtime
library** (`libgomp.so.1`), NOT the gcc-toolset sibling `libgomp.so`
which is just a linker script (ASCII text), nor `.a` (static archive):

```bash
# Find the real runtime libgomp (must be a real .so file, not a text script)
SYS_GOMP=$(find / -name "libgomp.so.1" -xtype f 2>/dev/null | head -1)
# Typical location: /usr/lib64/libgomp.so.1

# Create BOTH symlinks — CMake's find_library(NAMES gomp) looks for
# `libgomp.so`, while the probe globs `libgomp*.so*`:
ln -sf "$SYS_GOMP" $PYTORCH_DIR/torch/lib/libgomp.so.1
ln -sf "$SYS_GOMP" $PYTORCH_DIR/torch/lib/libgomp.so

# Verify CMake can stat() through the symlinks (EXISTS in CMake follows
# symlinks and returns FALSE if the target is missing):
cmake -E echo "$(cmake -DP=$PYTORCH_DIR/torch/lib/libgomp.so.1 -P /dev/stdin <<'CM'
if(EXISTS "${P}")
  message(STATUS "ok: ${P}")
else()
  message(FATAL_ERROR "broken symlink: ${P}")
endif()
CM
)"
```

**Pitfall we hit:** The gcc-toolset path `/opt/rh/gcc-toolset-14/.../libgomp.so`
looks plausible but is an 82-byte linker script; `libgomp.so.1` does NOT
exist alongside it. Symlinking to that path creates a broken symlink —
the CMake probe returns it, then `if(EXISTS "${PATH}")` in
`vllm_prepare_torch_gomp_shim` returns FALSE (because CMake follows
symlinks), the shim is skipped, and `find_library` eventually fails
with "Could not find OPEN_MP". Always verify the target with
`stat` / `ls -L` before assuming it works.

**Recurrence:** If PyTorch is rebuilt from source (a clean `torch/lib/`
gets produced), re-create both symlinks. Consider adding this step to
`build-pytorch.sh` post-build so it is automatic.

---

## Step 10: Verification

After all changes:

1. **Grep for stale references** (should return nothing outside lock files):
   ```bash
   grep -rn "$OLD" $TORCH_SPYRE_DIR/ --include="*.toml" --include="*.yaml" \
     --include="*.yml" --include="*.md" --include="*.sh" --include="*.py" \
     | grep -v ".git/" | grep -v __pycache__ | grep -v "uv.lock" | grep -v "requirements/"
   ```

2. **Same for docs repo:**
   ```bash
   grep -rn "$OLD" $DOCS_DIR/ --include="*.toml" --include="*.yaml" \
     --include="*.yml" --include="*.md" --include="*.sh" --include="*.py" \
     | grep -v ".git/"
   ```

3. **Also check for any version between OLD and NEW** (catches leftovers
   from skipped versions). For example, if upgrading from 2.10 to 2.13:
   ```bash
   grep -rn "2\.11\|2\.12" $TORCH_SPYRE_DIR/ --include="*.toml" \
     --include="*.yaml" --include="*.yml" --include="*.md" \
     --include="*.sh" --include="*.py" \
     | grep -v ".git/" | grep -v __pycache__ | grep -v "uv.lock" | grep -v "requirements/"
   ```

4. **Rebuild torch-spyre** (C++ extension + eager op registrations):
   ```bash
   export LOCAL_PYTORCH=1
   $DOCS_DIR/scripts/build-torch-spyre.sh --local-pytorch
   ```

5. **Run tests:**
   ```bash
   cd $TORCH_SPYRE_DIR
   python -m pytest tests/_inductor/test_inductor_ops.py -v
   ```

---

## Output Summary

After completing all steps, report:

```
✅ Completed (upgraded $OLD → $NEW):
   - pyproject.toml: active deps (3) + commented alternatives (3) + comments (2)
   - upstream_tests.yaml: comments updated
   - upstream_tests_beta.yaml (if present): comments updated
   - SKILL.md (project-overview): version references updated
   - docs/source/getting_started/installation.md: version table updated
   - checkout-pytorch-src.sh: comment + branch updated
   - build-torch-spyre.sh: sed patterns updated (both forward + trap)
   - dev_install.md: version references updated
   - profiling_tools.md: kineto wheel URL updated
   - libgomp shim symlinks created under $PYTORCH_DIR/torch/lib/

⏳ Remaining (manual/network-dependent):
   - [ ] Regenerate uv.lock + requirements/*.txt via `./tools/update-requirements.sh` (needs wheels on index)
   - [ ] Verify kineto-spyre wheel exists for $NEW
   - [ ] Rebuild torch-spyre + any downstream C++ extensions (vllm, ...)
   - [ ] Run test suite
   - [ ] Check torch_spyre/_monkey_patch.py & torch_spyre/ops/eager.py
         for API drift against the new PyTorch (Dynamo guards, ATen ops)
```

---

## Files That Do NOT Need Changes

| File | Reason |
|------|--------|
| `torch_spyre/ops/eager.py` | Iterates op overloads dynamically; usually version-independent — only touch if an ATen op renamed/changed |
| `torch_spyre/ops/fallbacks.py` | CPU-fallback registrations |
| `.github/workflows/upstream_tests.yaml` (logic) | Reads pyproject.toml dynamically for release branch |
| `scripts/build-pytorch.sh` | Builds whatever is in the pytorch dir |
| `scripts/checkout-required-src.sh` | No version references |
| `requirements/*.txt`, `uv.lock` | Auto-generated by `./tools/update-requirements.sh`; must be regenerated, not edited |

**Removed upstream (as of `faad75c`):** top-level `codegen/` directory
including `codegen/gen.py`, `codegen/inputs/*.yaml`/`*.h`,
`codegen/utils/template_tools.py`, `codegen/templates/*`. If you still
see these files in your checkout, you are on an older branch — see the
"Legacy" note under Step 3.

---

## Potential Breakage

After upgrading, watch for:

1. **Op signature changes** — ATen ops may have changed arguments or
   overloads. `torch_spyre/ops/eager.py` skips ops that don't take a
   Tensor or that include `"dtype"` in the overload name, but any other
   signature shift may surface as a `TypeError` at runtime.
2. **New ATen ops we don't yet register** — If the new PyTorch adds an
   op that torch-spyre needs to support, add it via
   `register_torch_compile_kernel(...)` in `torch_spyre/ops/eager.py`
   or `@register_fallback` in `torch_spyre/ops/fallbacks.py`.
3. **Removed / renamed ops** — Grep `torch_spyre/` for direct references
   to `aten.<name>` and confirm each still exists in the new PyTorch's
   `native_functions.yaml`.
4. **Inductor API changes** — `torch._inductor` internals may have changed.
   Watch for import errors or API mismatches in `torch_spyre/_inductor/`.
5. **Deprecation warnings** — New deprecations may need filterwarnings entries
   in `pyproject.toml`.
5a. **Dynamo / guard API changes** — `torch._C._dynamo.guards.GuardManager`
   method signatures occasionally gain/lose arguments. Monkey-patches in
   `torch_spyre/_monkey_patch.py` (e.g. `TENSOR_MATCH` override calling
   `add_lambda_guard`) may need updating. Example: PyTorch 2.11 added a
   required `user_stack` parameter to `add_lambda_guard` — calls with the
   old 2-arg form fail at runtime with
   `TypeError: add_lambda_guard(): incompatible function arguments`.
   Fix: pass `guard.user_stack` (or `None`) as the third argument. Check
   signatures in `torch/_C/_dynamo/guards.pyi` after the upgrade and
   cross-reference against torch-spyre's patches.
6. **Skipped-version accumulation** — If skipping versions, multiple breaking
   changes may compound. Consider reading PyTorch release notes for each
   skipped version to anticipate issues.
7. **Legacy codegen drift (pre-`faad75c` branches only)** — Old release
   branches that still contain the top-level `codegen/` pipeline may
   hit `IndexError` or `NameError` in `codegen_ops.py` generation due
   to changes in `RegistrationDeclarations.h` frontmatter or new C++
   types in schemas. See "Legacy" note under Step 3. On
   upstream/main and later this is a non-issue.
8. **Downstream C++ extensions must be rebuilt** — PyTorch's C++ ABI changes
   across minor versions. Any extension linked against `libtorch` /
   `libc10` (vllm `_C.abi3.so`, torchvision, torchaudio, custom kernels)
   must be rebuilt against the new PyTorch. See dedicated section below.

---

## Reference: GitHub Issue Template

When opening the upgrade issue:
```
Title: Upgrade to PyTorch $NEW
Labels: dependencies, good first issue

Body:
Upgrading to PT $NEW should be simple: update the dependency in
pyproject.toml, and updating the metadata files in eager codegen.
Sometimes things will break. This issue will track this.
```

See: https://github.com/torch-spyre/torch-spyre/issues/1315 (2.11 example)
