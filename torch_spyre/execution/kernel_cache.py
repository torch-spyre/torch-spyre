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

import json
import os
import shutil
import uuid
from collections.abc import Sequence
from functools import lru_cache
from typing import Optional

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


# ---------------------------------------------------------------------------
# Per-process kernel hash registry — maps cache_key to debug metadata.
# Populated by compute_specs_hash(). Hit/miss wiring (record_hit/record_miss)
# is deferred to a follow-up PR; saving the registry summary to disk will land
# alongside that wiring. Access via get_kernel_registry() for read, or
# directly for tests.
# ---------------------------------------------------------------------------


class _KernelHashRegistry:
    """Process-lifetime registry mapping cache_key to kernel debug metadata.

    Thread-safe for read-heavy workloads (GIL protects the dict). Each entry
    is a plain dict with keys:
        kernel_name  str        kernel name from async_compile.sdsc()
        ops          list[str]  op names of every OpSpec in the tree, depth-first
        loop_counts  list[str]  str(count) for every LoopSpec, depth-first
        pool_offsets list[str]  "arg{idx}:{offset}" for every hbm_pool arg
        hit_count    int        number of cache HITs for this key
        miss_count   int        number of cache MISSes (compiled fresh) for this key
    """

    def __init__(self):
        self._registry: dict[str, dict] = {}

    def register(
        self,
        cache_key: str,
        kernel_name: str,
        ops: list[str],
        loop_counts: list[str],
        pool_offsets: list[str],
    ) -> None:
        """Record the hash ingredients for cache_key.

        Called once per compute_specs_hash invocation. If the same key is seen
        again (e.g. after dynamo reset), metadata is updated and counters preserved.
        """
        if cache_key not in self._registry:
            self._registry[cache_key] = {
                "kernel_name": kernel_name,
                "kernel_names": [kernel_name] if kernel_name else [],
                "ops": ops,
                "loop_counts": loop_counts,
                "pool_offsets": pool_offsets,
                "hit_count": 0,
                "miss_count": 0,
            }
        else:
            # Same key, potentially a new kernel_name (e.g. after dynamo reset).
            # Update metadata but keep counters.
            entry = self._registry[cache_key]
            entry["kernel_name"] = kernel_name
            entry["ops"] = ops
            entry["loop_counts"] = loop_counts
            entry["pool_offsets"] = pool_offsets
            if kernel_name and kernel_name not in entry["kernel_names"]:
                entry["kernel_names"].append(kernel_name)

    def record_hit(self, cache_key: str) -> None:
        if cache_key in self._registry:
            self._registry[cache_key]["hit_count"] += 1

    def record_miss(self, cache_key: str) -> None:
        if cache_key in self._registry:
            self._registry[cache_key]["miss_count"] += 1

    def get(self, cache_key: str) -> Optional[dict]:
        return self._registry.get(cache_key)

    def all_entries(self) -> dict[str, dict]:
        return dict(self._registry)

    def summary(self) -> str:
        """Return a human-readable summary table of all registered kernels."""
        if not self._registry:
            return "KernelHashRegistry: empty"
        lines = [
            "KernelHashRegistry:",
            f"  {'HASH':>16}  {'KERNEL':40}  {'OPS':30}  LOOPS  POOL_OFFSETS  HIT  MISS  NAMES",
            f"  {'-' * 16}  {'-' * 40}  {'-' * 30}  {'-' * 5}  {'-' * 12}  {'-' * 3}  {'-' * 4}  {'-' * 20}",
        ]
        for key, meta in self._registry.items():
            ops_str = ",".join(meta["ops"])[:30]
            loops_str = ",".join(meta["loop_counts"]) or "-"
            pool_str = ",".join(meta["pool_offsets"]) or "-"
            names_str = ",".join(meta.get("kernel_names", []))[:20] or "-"
            lines.append(
                f"  {key[:16]:>16}  {meta['kernel_name']:40}  {ops_str:30}"
                f"  {loops_str:5}  {pool_str:12}"
                f"  {meta['hit_count']:3}  {meta['miss_count']:4}  {names_str:20}"
            )
        return "\n".join(lines)


_registry = _KernelHashRegistry()


def get_kernel_registry() -> _KernelHashRegistry:
    """Return the process-global kernel hash registry."""
    return _registry


@lru_cache(maxsize=1)
def _get_dxp_version() -> str:
    """Return a combined deeptools+flex version string from the Spyre components file.

    Reads the path given by the ``LIB_VERSION_FILE`` environment variable
    (set in the container build) and extracts the ``ibm-deeptools`` and
    ``flex`` entries.  Both components influence compiled output and must
    therefore both appear in the cache key.

    Raises ``RuntimeError`` if ``LIB_VERSION_FILE`` is unset, the file is
    missing, or either the ``ibm-deeptools`` or ``flex`` entry cannot be
    found  callers must disable caching rather than risk stale cache hits
    with an unknown compiler version.
    """
    components_file = os.environ.get("LIB_VERSION_FILE")
    if not components_file:
        raise RuntimeError(
            "LIB_VERSION_FILE is not set; cannot determine compiler version "
            "for cache key. Set SPYRE_KERNEL_CACHE=0 to run without caching."
        )
    versions: dict[str, str] = {}
    try:
        with open(components_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lib_name, lib_ver = line.split(":", 1)
                except ValueError:
                    continue
                name = lib_name.strip()
                if name in ("ibm-deeptools", "ibm-flex"):
                    versions[name] = lib_ver.strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"Spyre components file not found at {components_file!r}; "
            "cannot determine compiler version for cache key. "
            "Set SPYRE_KERNEL_CACHE=0 to run without caching."
        )
    for lib in ("ibm-deeptools", "ibm-flex"):
        if lib not in versions:
            raise RuntimeError(
                f"'{lib}' entry not found in {components_file!r}; "
                "cannot determine compiler version for cache key. "
                "Set SPYRE_KERNEL_CACHE=0 to run without caching."
            )
    return f"deeptools={versions['ibm-deeptools']};flex={versions['ibm-flex']}"


@lru_cache(maxsize=1)
def _get_torch_spyre_version() -> str:
    """Return the torch_spyre package version string."""
    from torch_spyre.version import __version__

    return __version__


def get_cache_root_dir() -> str:
    """Return the root directory for the persistent kernel cache."""
    cache_root = os.path.join(cache_dir(), "inductor-spyre-cache")
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


def _strip_debug_handles(obj):
    """Recursively remove "debug_handle_" keys from a JSON-serialisable object.

    debug_handle_ carries Inductor-assigned buffer names and source file paths
    that are process/run-specific. Including them in the cache key causes false
    misses (identical graphs with different buffer names hash differently) and
    does not affect compilation correctness — dxp_standalone ignores the field.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_debug_handles(v) for k, v in obj.items() if k != "debug_handle_"
        }
    if isinstance(obj, list):
        return [_strip_debug_handles(item) for item in obj]
    return obj


def compute_specs_hash(
    specs: Sequence, kernel_name: str = "", pool_size: int = 0
) -> str:
    """Compute a cache key from OpSpec objects without any disk I/O.

    The key is a SHA-256 hash covering: the JSON of every sdsc_N.json dict
    (op structure, iteration space, tiling, shapes, dtypes), the trip count of
    every LoopSpec, all baked symbol offsets (pool, kernel_slice, derived),
    the total pool allocation size, and the versions of torch, torch_spyre,
    dxp_standalone, and the active compile config.

    Args:
        specs:       The OpSpec/LoopSpec tree to hash.
        kernel_name: Optional name stored in the registry for debugging.
        pool_size:   Total HBM pool allocation in bytes, emitted as
                     ``sdscbundle.device_mem_allocate`` in bundle.mlir.
                     Must be included so that kernels that differ only in
                     their pool size get different cache keys.
    """
    from torch_spyre._inductor.codegen.superdsc import compile_op_spec
    from torch_spyre._inductor.op_spec import LoopSpec, OpSpec
    from torch_spyre._inductor import config as _spyre_config

    use_symbols = _spyre_config.bundle_symbolic_args

    specs_list = list(specs)

    content_parts: list[bytes] = []
    symbols: list[int] = []
    symbol_id_offset = 0
    sdsc_idx = 0

    # Debug metadata collected in parallel with content_parts.
    _debug_ops: list[str] = []
    _debug_loop_counts: list[str] = []
    _debug_pool_offsets: list[str] = []

    def _collect(entries):
        nonlocal sdsc_idx, symbol_id_offset
        for entry in entries:
            if isinstance(entry, LoopSpec):
                # Include the trip count so loops with different iteration
                # counts never collide, even when their body OpSpecs produce
                # identical SDSC JSON.
                loop_count_str = str(entry.count)
                content_parts.append(f"loop_count:{loop_count_str}".encode())
                _debug_loop_counts.append(loop_count_str)
                logger.debug("  [hash] LoopSpec  count=%s", loop_count_str)
                _collect(entry.body)
            elif isinstance(entry, OpSpec):
                sdsc_json, local_sym_values, affine_strides, local_symbol_kinds = (
                    compile_op_spec(
                        sdsc_idx,
                        entry,
                        symbols,
                        symbol_id_offset,
                    )
                )
                symbol_id_offset += len(local_sym_values)
                sdsc_idx += 1
                # Strip debug_handle_ before hashing: it contains run-specific
                # buffer names and source paths that must not affect the key.
                hashable_json = _strip_debug_handles(sdsc_json)
                content_parts.append(json.dumps(hashable_json, sort_keys=True).encode())
                # Include affine_strides in the hash: these are the per-tensor,
                # per-loop-level byte strides emitted as affine.apply ops in
                # bundle.mlir.  Two ops with identical SDSC JSON but different
                # tile strides produce different compiled kernels and must not
                # share a cache entry.
                if any(
                    any(level_strides for level_strides in per_level)
                    for per_level in affine_strides
                ):
                    content_parts.append(
                        json.dumps(
                            [
                                [
                                    {str(k): v for k, v in level.items()}
                                    for level in per_level
                                ]
                                for per_level in affine_strides
                            ],
                            sort_keys=True,
                        ).encode()
                    )
                _debug_ops.append(entry.op)
                logger.debug(
                    "  [hash] OpSpec[%d] op=%s  iter_space=%s  n_args=%d",
                    sdsc_idx - 1,
                    entry.op,
                    {str(k): str(v[0]) for k, v in entry.iteration_space.items()},
                    len(entry.args),
                )

                # Include baked symbol offsets (pool, kernel_slice, derived)
                # when use_symbols=True. These are emitted as concrete MLIR
                # constants in bundle.mlir but appear only as negative sentinel
                # IDs in sdsc_N.json, so they must be hashed separately to
                # distinguish graphs that differ only in their offsets.
                if use_symbols:
                    for sk in local_symbol_kinds:
                        if sk.kind == "pool":
                            tag = f"pool:{sk.offset}"
                            content_parts.append(tag.encode())
                            _debug_pool_offsets.append(tag)
                            logger.debug("  [hash]   pool offset=%d", sk.offset)
                        elif sk.kind == "kernel_slice":
                            tag = f"slice:arg{sk.arg_index}:{sk.offset}"
                            content_parts.append(tag.encode())
                            _debug_pool_offsets.append(tag)
                            logger.debug(
                                "  [hash]   slice arg%d offset=%d",
                                sk.arg_index,
                                sk.offset,
                            )
                        elif sk.kind == "kernel_derived" and sk.offset != 0:
                            tag = f"derived:arg{sk.arg_index}:{sk.offset}"
                            content_parts.append(tag.encode())
                            _debug_pool_offsets.append(tag)
                            logger.debug(
                                "  [hash]   derived arg%d offset=%d",
                                sk.arg_index,
                                sk.offset,
                            )
                        elif sk.kind == "kernel_derived_symbolic":
                            tag = f"derived_sym:{sk.pytorch_sym}:{sk.split_count}:{sk.core_idx}"
                            content_parts.append(tag.encode())
                            _debug_pool_offsets.append(tag)
                            logger.debug(
                                "  [hash]   derived_sym %s split=%d core=%d",
                                sk.pytorch_sym,
                                sk.split_count,
                                sk.core_idx,
                            )

    _collect(specs_list)

    # Include the total pool allocation: it is emitted as
    # sdscbundle.device_mem_allocate <pool_size> bytes in bundle.mlir.
    content_parts.append(f"pool_size:{pool_size}".encode())

    content = b"||".join(content_parts)
    extra = "||".join(
        [
            torch.__version__,
            _get_torch_spyre_version(),
            _get_dxp_version(),
        ]
    )

    cache_key = code_hash(content, extra=extra)
    logger.info(
        "Hash inputs summary  kernel=%s  sdsc_count=%d  content_parts=%d  "
        "content_bytes=%d  use_symbols=%s",
        kernel_name or "<unknown>",
        sdsc_idx,
        len(content_parts),
        len(content),
        use_symbols,
    )

    # Register ingredients in the process-global registry for hit/miss logging.
    _registry.register(
        cache_key,
        kernel_name,
        _debug_ops,
        _debug_loop_counts,
        _debug_pool_offsets,
    )

    logger.info(
        "Computed specs hash  kernel=%s  ops=[%s]  loops=[%s]  pool=[%s]  key=%s",
        kernel_name or "<unknown>",
        ",".join(_debug_ops),
        ",".join(_debug_loop_counts) or "-",
        ",".join(_debug_pool_offsets) or "-",
        cache_key,
    )
    return cache_key


def get_cached_kernel_dir(cache_key: str) -> Optional[str]:
    """Return the cached kernel directory if all required artifacts are present.

    Checks for every entry in _REQUIRED_ARTIFACTS and at least one sdsc_N.json.
    A partial write from a killed process will fail this check and trigger
    recompilation.
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


def allocate_compile_dir(cache_key: str) -> str:
    """Reserve a unique temp directory inside the cache root for compilation.

    Placing it inside the cache root (not /tmp) ensures the subsequent rename
    in commit_compile_dir is atomic on POSIX.
    """
    cache_root = get_cache_root_dir()
    tmp_dir = os.path.join(cache_root, f"{cache_key}.tmp.{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def commit_compile_dir(tmp_dir: str, cache_key: str) -> str:
    """Atomically promote tmp_dir to <cache_root>/<cache_key>/.

    If another process already committed the same key, discards the temp dir
    and reuses the existing entry. Returns the final cache dir.
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


def _move_to_failed_dir(compile_dir: str) -> None:
    """Move a failed compile dir into a ``failed/`` subdirectory of the cache root.

    Keeps the cache root clean while still retaining failed artifacts for
    manual debugging (``dxp_standalone -d <path>``).  If the rename itself
    fails (e.g. cross-device move), the original path is kept and logged.
    """
    cache_root = get_cache_root_dir()
    failed_root = os.path.join(cache_root, "failed")
    try:
        os.makedirs(failed_root, exist_ok=True)
        dest = os.path.join(failed_root, os.path.basename(compile_dir))
        os.rename(compile_dir, dest)
        logger.info("Compilation failed; moved compile dir to: %s", dest)
    except OSError:
        logger.info(
            "Compilation failed; could not move compile dir, keeping at: %s",
            compile_dir,
        )


def get_cache_stats() -> dict:
    """Return summary statistics about the persistent kernel cache.

    Only counts and sizes committed kernel directories (not in-progress
    ``.tmp.`` dirs and not the ``failed/`` directory).
    """
    cache_root = get_cache_root_dir()

    if not os.path.exists(cache_root):
        return {"total_cached_kernels": 0, "cache_size_mb": 0.0}

    cached_dirs = [
        d
        for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
        and not d.endswith(".tmp")
        and ".tmp." not in d
        and d != "failed"
    ]

    # Walk only the committed kernel dirs so that in-progress .tmp. dirs and
    # the failed/ directory do not inflate the reported size.
    total_size = 0
    for entry_name in cached_dirs:
        entry_path = os.path.join(cache_root, entry_name)
        for dirpath, _dirnames, filenames in os.walk(entry_path):
            for filename in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, filename))

    return {
        "total_cached_kernels": len(cached_dirs),
        "cache_size_mb": total_size / (1024 * 1024),
    }


def clear_cache() -> None:
    """Remove all entries from the persistent kernel cache."""
    cache_root = get_cache_root_dir()
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root)
        os.makedirs(cache_root, exist_ok=True)
        logger.info("Kernel cache cleared")
