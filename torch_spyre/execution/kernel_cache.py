# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import os
import pathlib
import shutil
import uuid
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Optional

import torch
from torch._inductor.codecache import code_hash
from torch._inductor.runtime.runtime_utils import cache_dir

from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("kernel_cache")

# All artifacts that dxp_standalone must produce for a valid compiled kernel.
# A cache entry is only considered a hit if every one of these is present.
_REQUIRED_ARTIFACTS = [
    "bundle.mlir",
    os.path.join("spyreCodeDir", "init_binary.bin"),
    os.path.join("spyreCodeDir", "spyrecode.json"),
]

# Subdirectory of the cache root holding retained failed compile dirs.  It is a
# reserved name, not a cache key: cache keys are hex digests, so no key can ever
# collide with it, and anything enumerating cache entries must skip it.
_FAILED_DIR_NAME = "failed"

# Entry points of the artifact-generating code, relative to the torch_spyre
# package root.  These are *seeds*: the modules they import are discovered
# automatically (see ``_iter_source_files``), so a new helper module pulled in
# by any of these is covered without touching this list.  Only add a seed here
# if it emits artifacts without being reachable from an existing one.
_ARTIFACT_SOURCE_SEEDS = [
    os.path.join("_inductor", "codegen", "bundle.py"),
    os.path.join("_inductor", "codegen", "superdsc.py"),
    os.path.join("_inductor", "codegen", "compute_ops.py"),
    os.path.join("_inductor", "codegen", "ktir.py"),
    os.path.join("_inductor", "op_spec.py"),
]


@lru_cache(maxsize=1)
def _get_spyre_library_versions() -> dict[str, str]:
    """Return all Spyre library versions (deeptools, senlib, etc.) from LIB_VERSION_FILE.

    Reads from the file specified by the LIB_VERSION_FILE environment variable.
    This file should contain lines in the format "library-name:version".
    Returns a dict mapping library names to their versions (e.g., ibm-deeptools,
    ibm-senlib-core).

    Raises RuntimeError if LIB_VERSION_FILE is not set or the file cannot be read.
    To disable kernel caching, set SPYRE_KERNEL_CACHE=0.
    """
    lib_version_file = os.environ.get("LIB_VERSION_FILE")
    if not lib_version_file:
        raise RuntimeError(
            "LIB_VERSION_FILE environment variable is required for kernel caching. "
            "It should point to a .txt file containing Spyre library versions"
            "versions (deeptools, senlib, etc.). "
            "To disable caching, set SPYRE_KERNEL_CACHE=0."
        )

    try:
        libraries = {}
        with open(lib_version_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                name, version = line.split(":", 1)
                libraries[name.strip()] = version.strip()
        logger.info(
            "Loaded %d Spyre library versions from %s", len(libraries), lib_version_file
        )
        return libraries
    except FileNotFoundError as e:
        raise RuntimeError(
            f"LIB_VERSION_FILE={lib_version_file} not found. "
            "To disable caching, set SPYRE_KERNEL_CACHE=0."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error reading Spyre library versions from {lib_version_file}: {e}. "
            "To disable caching, set SPYRE_KERNEL_CACHE=0."
        ) from e


@lru_cache(maxsize=1)
def _get_system_info() -> dict[str, Any]:
    """Return system/device info to include when generating the kernel cache key."""
    return {
        "device": {
            "flex_device": os.environ.get("FLEX_DEVICE", ""),
            "flex_compute": os.environ.get("FLEX_COMPUTE", ""),
            "world_size": os.environ.get("WORLD_SIZE", "1"),
        }
    }


@lru_cache(maxsize=1)
def _get_compile_config() -> dict[str, Any]:
    """Return compiler options to include when generating the kernel cache key."""
    return {
        "lx_planning": os.environ.get("LX_PLANNING", "0"),
        "hbm_pool_planning": os.environ.get("HBM_POOL_PLANNING", "0"),
        "layout_solver": os.environ.get("LAYOUT_SOLVER", "greedy"),
    }


def _package_root() -> pathlib.Path:
    """Return the on-disk root of the installed ``torch_spyre`` package."""
    return pathlib.Path(__file__).resolve().parent.parent


def _module_to_relpath(module: str, root: pathlib.Path) -> Optional[pathlib.PurePath]:
    """Map a dotted ``torch_spyre.*`` module name to a path under *root*.

    Returns None for modules that are not plain files in the package (namespace
    packages, C extensions, or anything outside ``torch_spyre``).
    """
    if not module.startswith("torch_spyre"):
        return None
    tail = module[len("torch_spyre") :].lstrip(".")
    stem = pathlib.PurePath(*tail.split(".")) if tail else pathlib.PurePath()

    as_module = stem.with_suffix(".py") if tail else None
    if as_module is not None and (root / as_module).is_file():
        return as_module

    as_package = stem / "__init__.py"
    if (root / as_package).is_file():
        return as_package

    return None


def _imported_modules(tree: ast.Module, package: str) -> list[str]:
    """Return the absolute ``torch_spyre.*`` module names imported by *tree*.

    Relative imports are resolved against *package*, the dotted name of the
    package containing the module being parsed.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # ``from . import x`` / ``from ..y import z`` — walk up
                # ``level - 1`` packages from the containing package.
                parts = package.split(".")
                base = parts[: len(parts) - (node.level - 1)] or parts[:1]
                prefix = ".".join(base)
                found.append(f"{prefix}.{node.module}" if node.module else prefix)
            elif node.module:
                found.append(node.module)
    return found


@lru_cache(maxsize=1)
def _iter_source_files() -> tuple[pathlib.PurePath, ...]:
    """Return the transitive closure of torch-spyre sources that emit artifacts.

    Starts from ``_ARTIFACT_SOURCE_SEEDS`` and follows ``torch_spyre`` imports
    via the AST, so every module the artifact-generating code actually depends
    on is included.

    Import discovery is static, so a module reached only through a runtime
    ``importlib`` call would be missed.  The codegen path does not do that
    today; a deliberately dynamic import there must be added as a seed.

    Returned paths are relative to the package root and sorted, so the hash is
    independent of filesystem walk order.
    """
    root = _package_root()

    seen: set[pathlib.PurePath] = set()
    queue: list[pathlib.PurePath] = [
        pathlib.PurePath(seed) for seed in _ARTIFACT_SOURCE_SEEDS
    ]

    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        path = root / rel
        if not path.is_file():
            # A seed that no longer exists means this list has drifted from the
            # codebase; failing loudly beats hashing a smaller set than
            # intended, which would silently weaken the key.
            raise RuntimeError(
                f"Kernel-cache source seed {rel} not found under {root}. "
                "Update _ARTIFACT_SOURCE_SEEDS. "
                "To disable caching, set SPYRE_KERNEL_CACHE=0."
            )
        seen.add(rel)

        package = "torch_spyre"
        if rel.parent != pathlib.PurePath("."):
            package = f"torch_spyre.{'.'.join(rel.parent.parts)}"

        try:
            tree = ast.parse(path.read_bytes())
        except SyntaxError as e:
            raise RuntimeError(
                f"Could not parse {rel} while building the kernel-cache key: {e}. "
                "To disable caching, set SPYRE_KERNEL_CACHE=0."
            ) from e

        for module in _imported_modules(tree, package):
            dep = _module_to_relpath(module, root)
            if dep is not None and dep not in seen:
                queue.append(dep)

    return tuple(sorted(seen))


@lru_cache(maxsize=1)
def _get_torch_spyre_source_hash() -> str:
    """Hash the torch-spyre code that produces the compiled artifacts.

    ``compute_specs_hash`` replays ``compile_op_spec`` live, so most codegen
    changes already alter the key through their effect on ``sdsc_json`` and the
    symbol structure.  That coupling is a property of today's emitter, though,
    not an invariant the design enforces: code downstream of ``compile_op_spec``
    (notably ``generate_bundle``'s ``bundle.mlir`` emission) can in principle
    change its output while the replayed values stay byte-identical.  Hashing
    the sources closes that gap by construction, so the guarantee no longer
    rests on a property that has to be re-measured after every emitter change.

    The compiled ``_C`` extension is included as well: the scratchpad packer and
    layout solver live there, and a rebuild changes artifacts without touching
    any ``.py`` file.

    Trade-off, stated plainly: this invalidates the cache on source edits that
    could not have changed the artifacts.  That is deliberate.  A spurious miss
    costs one recompile; a false hit yields a silently wrong kernel.
    """
    root = _package_root()

    hasher_parts: list[bytes] = []
    for rel in _iter_source_files():
        hasher_parts.append(str(rel).encode())
        hasher_parts.append((root / rel).read_bytes())

    # The native extension is a build artifact, not a source file, so it is not
    # part of the import closure and must be added explicitly.
    ext_path = root / "_C.so"
    if ext_path.is_file():
        hasher_parts.append(b"_C.so")
        hasher_parts.append(ext_path.read_bytes())
    else:
        # An editable install always has it in-tree; a wheel install may name it
        # with an ABI suffix. Fall back to whatever the loaded module resolves
        # to so the extension still reaches the key.
        try:
            import torch_spyre._C as _C

            ext_file = getattr(_C, "__file__", None)
            if ext_file and os.path.isfile(ext_file):
                hasher_parts.append(os.path.basename(ext_file).encode())
                hasher_parts.append(pathlib.Path(ext_file).read_bytes())
            else:
                raise RuntimeError("torch_spyre._C has no file on disk")
        except Exception as e:
            raise RuntimeError(
                f"Could not locate the torch_spyre native extension to hash: {e}. "
                "A C++ rebuild would not invalidate the kernel cache. "
                "To disable caching, set SPYRE_KERNEL_CACHE=0."
            ) from e

    source_hash = code_hash(b"||".join(hasher_parts))
    logger.info(
        "torch-spyre source hash over %d module(s) + native extension: %s",
        len(_iter_source_files()),
        source_hash,
    )
    return source_hash


def get_cache_root_dir() -> str:
    """Return the root directory for the persistent kernel cache."""
    cache_root = os.path.join(cache_dir(), "inductor-spyre-cache")
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


# ---------------------------------------------------------------------------
# In-memory cache key — computed BEFORE any disk I/O
# ---------------------------------------------------------------------------


def _symbol_kind_key(kind) -> list:
    """Return the cache-relevant fields of a ``SymbolKind``.

    Only the parts of a symbol that are **baked into the compiled bundle** may
    enter the cache key.  Splitting ``SymbolKind`` this way is what makes the
    key both correct and reusable:

    The corresponding *value* in ``base_symbol_values`` (the raw HBM byte
    address) is deliberately **not** included: it is supplied at launch as an
    ``!sdscbundle.input_arg`` parameter, so two runs that place the same tensor
    at different addresses must still share a cache entry.  Hashing it would
    make every key allocation-specific and reduce the hit rate to ~zero.
    """
    return [
        kind.kind,
        kind.base_sym_idx,
        kind.offset,
        kind.arg_index,
        kind.granularity,
        kind.max_value,
        kind.pytorch_sym,
        kind.core_idx,
        kind.split_count,
    ]


def compute_specs_hash(specs: Sequence, pool_size: int = 0) -> str:
    """Compute a cache key directly from OpSpec objects — no disk I/O required.

    This is the preferred hashing entry point.  Because it operates entirely
    in-memory it allows the cache lookup to happen *before* ``generate_bundle``
    writes any files, so a cache hit skips ``generate_bundle`` and
    ``dxp_standalone`` entirely.

    The key is a SHA-256 hash (via Inductor's ``code_hash``) that covers:

    * The JSON serialisation of every ``sdsc_N.json`` dict produced by
      ``compile_op_spec`` — this captures the full op structure, iteration
      space, tiling, tensor shapes, and dtypes.
    * The ``SymbolKind`` *structure* of every registered symbol. This covers
      the compile-time address arithmetic that is baked into ``bundle.mlir``
      but is absent from the ``sdsc_N.json`` dicts.
    * ``torch.__version__`` — invalidates on PyTorch upgrades.
    * A content hash of the torch-spyre sources that generate the artifacts,
      plus the native ``_C`` extension — see
      ``_get_torch_spyre_source_hash``.  This covers emitter changes that do
      not surface in the replayed values above, and any C++ rebuild.
    * Spyre library versions (deeptools, senlib, etc.) from LIB_VERSION_FILE —
      invalidates when any Spyre tool version changes. Requires LIB_VERSION_FILE
      to be set; caching is disabled if it is not.
    * System info (device mode, topology) and compiler config (planning/layout
      env vars) — see ``_get_system_info`` and ``_get_compile_config``.
    * ``pool_size`` — emitted verbatim into ``bundle.mlir`` as the
      ``device_mem_allocate`` byte count, so it is not implied by the specs.
      Two graphs whose specs match but whose HBM pools differ must not share
      an entry.

    ``bundle.mlir`` is not hashed directly, but everything in it that is not
    already implied by the ``sdsc_N.json`` dicts *is* covered via the
    ``SymbolKind`` structure.
    """
    from torch_spyre._inductor.codegen.superdsc import compile_op_spec
    from torch_spyre._inductor.op_spec import LoopSpec, OpSpec

    specs_list = list(specs)

    content_parts: list[bytes] = []
    symbols: list[int] = []
    symbol_id_offset = 0
    sdsc_idx = 0

    def _collect(entries):
        nonlocal sdsc_idx, symbol_id_offset
        for entry in entries:
            if isinstance(entry, LoopSpec):
                _collect(entry.body)
            elif isinstance(entry, OpSpec):
                sdsc_json, local_sym_values, _, symbol_kinds = compile_op_spec(
                    sdsc_idx,
                    entry,
                    symbols,
                    symbol_id_offset,
                )
                symbol_id_offset += len(local_sym_values)
                sdsc_idx += 1
                content_parts.append(json.dumps(sdsc_json, sort_keys=True).encode())

                # The sdsc_json refers to addresses only as opaque negative symbol
                # ids, so we also hash the symbol structure to keep them distinct
                content_parts.append(
                    json.dumps(
                        [_symbol_kind_key(k) for k in symbol_kinds],
                        sort_keys=True,
                    ).encode()
                )

    _collect(specs_list)

    content = b"||".join(content_parts)

    # Build Spyre library versions string for cache key (sorted for determinism)
    library_versions = _get_spyre_library_versions()
    libraries_str = json.dumps(library_versions, sort_keys=True)

    system_info_str = json.dumps(_get_system_info(), sort_keys=True)
    compile_config_str = json.dumps(_get_compile_config(), sort_keys=True)

    extra = "||".join(
        [
            torch.__version__,
            _get_torch_spyre_source_hash(),
            libraries_str,
            system_info_str,
            compile_config_str,
            f"pool_size={pool_size}",
        ]
    )

    cache_key = code_hash(content, extra=extra)
    logger.info(
        "Computed specs hash from %d OpSpec(s): %s", len(content_parts), cache_key
    )
    return cache_key


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------


def get_cached_kernel_dir(cache_key: str) -> Optional[str]:
    """Return the cached kernel directory if all required artifacts are present.

    Validates that every artifact in ``_REQUIRED_ARTIFACTS`` exists and that
    at least one ``sdsc_N.json`` file is present.  A partial write from a
    killed process will fail this check and trigger recompilation.
    """
    cache_root = get_cache_root_dir()
    cached_dir = os.path.join(cache_root, cache_key)

    if not os.path.isdir(cached_dir):
        logger.info("Cache MISS: No cached kernel found for key %s", cache_key)
        return None

    missing = [
        p
        for p in _REQUIRED_ARTIFACTS
        if not os.path.isfile(os.path.join(cached_dir, p))
    ]
    if missing:
        logger.info(
            "Cache MISS: Cached dir exists but missing artifacts %s for key %s",
            missing,
            cache_key,
        )
        return None

    has_sdsc = any(
        f.startswith("sdsc_") and f.endswith(".json")
        for f in os.listdir(cached_dir)
        if os.path.isfile(os.path.join(cached_dir, f))
    )
    if not has_sdsc:
        logger.info(
            "Cache MISS: No sdsc_N.json files found in cached dir for key %s",
            cache_key,
        )
        return None

    logger.info("Cache HIT: Found cached kernel at %s", cached_dir)
    return cached_dir


# ---------------------------------------------------------------------------
# Cache write — single-directory approach (no copy step)
# ---------------------------------------------------------------------------


def allocate_compile_dir(cache_key: str) -> str:
    """Reserve a unique temp directory *inside* the cache root for compilation.

    ``dxp_standalone`` writes its output directly into this directory.  After
    compilation the caller promotes it to the final cache entry with
    ``commit_compile_dir``.  Because the temp dir lives inside the same
    filesystem as the final entry, the promotion is an atomic ``os.rename``.

    Using a directory inside the cache root (rather than the system
    ``/tmp``) ensures same-filesystem atomicity on POSIX.
    """
    cache_root = get_cache_root_dir()
    tmp_dir = os.path.join(cache_root, f"{cache_key}.tmp.{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def commit_compile_dir(tmp_dir: str, cache_key: str) -> str:
    """Atomically promote *tmp_dir* to the final ``<cache_root>/<cache_key>/`` entry.

    If another process already committed the same key, the loser discards its
    temp directory and reuses the winner's copy.  Returns the final cache dir.
    """
    cache_root = get_cache_root_dir()
    cached_dir = os.path.join(cache_root, cache_key)

    if os.path.isdir(cached_dir):
        # Another process/thread won the race — discard our copy.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cache race resolved: reusing existing entry at %s", cached_dir)
        return cached_dir

    try:
        os.rename(tmp_dir, cached_dir)  # Atomic on POSIX (same filesystem)
        logger.info("Saved compiled kernel to cache: %s", cached_dir)
    except OSError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cache race resolved: reusing existing entry at %s", cached_dir)

    return cached_dir


def retain_failed_compile_dir(tmp_dir: str, cache_key: str) -> Optional[str]:
    """Move a failed compile dir out of the commit namespace, keeping it on disk.

    A failed ``dxp_standalone`` run is usually only debuggable from the exact
    directory it was given: re-running ``dxp_standalone -d <dir>`` on the
    retained bundle reproduces the failure without having to reproduce the whole
    ``torch.compile`` invocation that emitted it.  So the directory is kept.

    It cannot be kept *where it is*, though.  ``allocate_compile_dir`` places it
    inside the cache root so that ``commit_compile_dir`` can promote it with an
    atomic rename; left there, every failure would litter the directory that
    holds live cache entries.  Moving it under ``failed/`` keeps the cache root
    free of half-written bundles while preserving the artifacts.

    Returns the retained path, or ``None`` if the directory could not be kept
    (in which case it is removed -- a failure to *retain* must not leave a
    partial bundle sitting in the commit namespace).
    """
    failed_root = os.path.join(get_cache_root_dir(), _FAILED_DIR_NAME)
    dest = os.path.join(failed_root, f"{cache_key}.{uuid.uuid4().hex}")
    try:
        os.makedirs(failed_root, exist_ok=True)
        os.rename(tmp_dir, dest)
    except OSError:
        # Retention is best-effort; cleanliness of the cache root is not.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.warning(
            "Could not retain failed compile dir for key %s; removed %s",
            cache_key,
            tmp_dir,
            exc_info=True,
        )
        return None

    return dest


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _is_cache_entry_dir(name: str) -> bool:
    """Return whether *name* (a cache-root child) is a committed cache entry.

    That is: neither an in-flight ``.tmp.`` compile dir nor the ``failed/``
    subdir.  Both statistics ``get_cache_stats`` reports share this definition
    so they cannot describe different sets.
    """
    return (
        not name.endswith(".tmp") and ".tmp." not in name and name != _FAILED_DIR_NAME
    )


def get_cache_stats() -> dict:
    """Return summary statistics about the persistent kernel cache.

    ``cache_size_mb`` counts only committed entries, so it reflects what
    ``clear_cache`` would actually reclaim as cache content.  Retained failures
    are reported separately as ``retained_failed_compiles``: they neither
    satisfy a lookup nor get evicted with one.
    """
    cache_root = get_cache_root_dir()
    failed_root = os.path.join(cache_root, _FAILED_DIR_NAME)

    if not os.path.exists(cache_root):
        # Keep the key set identical to the populated case, so a caller reading
        # a field does not have to care whether the cache exists yet.
        return {
            "total_cached_kernels": 0,
            "cache_size_mb": 0.0,
            "retained_failed_compiles": 0,
        }

    cached_dirs = [
        d
        for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d)) and _is_cache_entry_dir(d)
    ]

    total_size = 0
    for entry in cached_dirs:
        for dirpath, _, filenames in os.walk(os.path.join(cache_root, entry)):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(path)
                except OSError:
                    # A concurrent commit or clear can unlink a file mid-walk.
                    # Stats are advisory; a racing writer must not raise here.
                    continue

    failed_count = len(os.listdir(failed_root)) if os.path.isdir(failed_root) else 0

    return {
        "total_cached_kernels": len(cached_dirs),
        "cache_size_mb": total_size / (1024 * 1024),
        "retained_failed_compiles": failed_count,
    }


def clear_cache() -> None:
    """Remove all entries from the persistent kernel cache."""
    cache_root = get_cache_root_dir()
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root)
        os.makedirs(cache_root, exist_ok=True)
        logger.info("Kernel cache cleared")
