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

"""Source-to-kernel provenance construction for the Spyre Inductor backend.

The provenance *types* (``SourceLoc``, ``DebugHandle``) live in ``op_spec.py``
alongside the other IR-op schema dataclasses. This module holds the *logic* that
builds them from Inductor IR: stable-id hashing and ``build_debug_handle``, which
reads a ``ComputedBuffer``'s ``origins`` to construct the handle.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import regex

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    DebugHandle,
    ProvenanceTransform,
    SourceLoc,
)


_FRAME_RE = regex.compile(r'File "([^"]+)", line (\d+)')

# Named "provenance" so drop warnings are on logger "spyre.inductor.provenance":
# unlike warnings.warn, a logger call has no per-process dedup registry, so the
# per-compile reset in reset_provenance_warnings() genuinely re-emits on a later
# compile that hits the same regression.
logger = get_inductor_logger("provenance")

# Buffer attribute owned exclusively by the explicit provenance helpers. The
# immutable records survive 1:1 reconstruction and retain multiple lower-IR
# transformations instead of collapsing them into one scalar context string.
_SPYRE_PROV_HISTORY_ATTR = "_spyre_prov_history"


def _stable_id(
    source: SourceLoc | None,
    aten_op: str | None,
    ir_chain: tuple[str, ...],
) -> int:
    """Deterministic content hash of an op's provenance.

    Stability contract: reproducible for the same op within a compile and across
    recompiles on the same toolchain, but NOT across torch/scheduler versions
    (``ir_chain`` includes scheduling-assigned buffer names). It is a within-compile
    linking key, not a cross-run fingerprint; cross-version consumers should key on
    ``source`` + ``aten_op`` instead.
    """
    canonical = "|".join(
        [
            source.to_str() if source is not None else "",
            aten_op or "",
            ",".join(ir_chain),
        ]
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    # Top 8 bytes = 64 bits; ``>> 1`` drops the sign bit so the id is a
    # non-negative value that always fits a *signed* 64-bit integer, the common
    # interchange width (JSON int64, MLIR ``i64``, protobuf ``int64``). A full
    # 64-bit value could be read as negative in those consumers.
    # Caveat: 63 bits exceeds JS ``Number.MAX_SAFE_INTEGER`` (2**53 - 1), so
    # ``DebugHandle.to_dict`` serializes the id as a string on the JSON path
    # (a JSON number would be rounded to float64 at ``JSON.parse`` time).
    return int.from_bytes(digest[:8], "big") >> 1


def _source_from_node(node: Any) -> SourceLoc | None:
    """Extract a structured SourceLoc from an FX node's ``stack_trace`` meta."""
    meta = getattr(node, "meta", None) or {}
    trace = meta.get("stack_trace")
    if not trace:
        return None
    matches = _FRAME_RE.findall(trace)
    if not matches:
        return None
    # Prefer the innermost non-torch frame: the model source line closest to the
    # op call (frames run outermost -> innermost, so [-1] is closest). Fall back to
    # the innermost frame overall when every frame is torch-internal.
    user = [(f, ln) for (f, ln) in matches if "/torch/" not in f]
    file, line = (user or matches)[-1]
    return SourceLoc(file=file, start_line=int(line))


def _aten_from_node(node: Any) -> str | None:
    """Extract the ``original_aten`` op string from an FX node's meta."""
    meta = getattr(node, "meta", None) or {}
    op = meta.get("original_aten")
    return str(op) if op is not None else None


def _headline_source(
    per_node: list[tuple[str, SourceLoc | None, str | None]],
) -> SourceLoc | None:
    """The single distinct source if the origins agree on one, else None.

    Symmetric with the ``aten_op`` rule in ``build_debug_handle``: we never
    present an arbitrary line as *the* source of an op fused across multiple
    distinct source locations. When the origins disagree, the headline is None
    and the full set lives in ``fused_from`` (consumers pick a representative).
    """
    distinct = {s.to_str(): s for (_, s, _) in per_node if s is not None}
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def build_debug_handle(buffer: Any) -> DebugHandle | None:
    """Build a DebugHandle from a ComputedBuffer's origins. Best-effort.

    Returns None when no handle can be derived. Provenance is debug-only, so the
    compile-path caller (``create_op_spec``) also wraps this in try/except — a
    failure here never breaks a build.

    ``origins`` (the set) is authoritative. ``origin_node`` is used only when
    Inductor set it (the clean 1:1 non-view case); for fused/view ops it is None
    and we do not invent a primary — the full set lives in ``fused_from``.

    This function only *iterates and sorts* ``origins``; it never relies on set
    identity or insertion order. The caller may pass any iterable (``OrderedSet``,
    plain ``set``, or the fake tuple used in tests).
    """
    origins = getattr(buffer, "origins", None) or set()
    if not origins:
        return None
    # Sort by (name, aten_op) for a fully deterministic ir_chain and _stable_id.
    # FX guarantees node names are unique within a graph (_Namespace deduplicates
    # with _N suffixes), so duplicate primary keys cannot arise in practice;
    # the secondary key is a defensive tie-breaker for any future synthetic nodes.
    nodes = sorted(
        origins,
        key=lambda n: (
            getattr(n, "name", ""),
            str((getattr(n, "meta", None) or {}).get("original_aten", "")),
        ),
    )
    per_node = [
        (getattr(n, "name", ""), _source_from_node(n), _aten_from_node(n))
        for n in nodes
    ]
    origin_node = getattr(buffer, "origin_node", None)

    if origin_node is not None:
        # Inductor's authoritative 1:1 op: take its aten/source directly. If it
        # has no stack_trace, fall back only to a single agreed sibling source.
        aten_op = _aten_from_node(origin_node)
        source = _source_from_node(origin_node) or _headline_source(per_node)
    else:
        # Fused/view: headline source and aten_op are set only when the origins
        # agree on a single distinct value; otherwise None (do not guess) — the
        # full per-origin set is preserved in fused_from. Source and aten are
        # handled symmetrically.
        source = _headline_source(per_node)
        atens = {a for (_, _, a) in per_node if a is not None}
        aten_op = next(iter(atens)) if len(atens) == 1 else None

    buf_name = None
    get_name = getattr(buffer, "get_name", None)
    if callable(get_name):
        buf_name = get_name()
    ir_chain = tuple([nm for (nm, _, _) in per_node] + ([buf_name] if buf_name else []))

    # fused_from: the authoritative set — every origin when the kernel fuses >1.
    fused_from: tuple[DebugHandle, ...] = ()
    if len(nodes) > 1:
        fused_from = tuple(
            DebugHandle(
                id=_stable_id(s, a, (nm,)),
                source=s,
                aten_op=a,
                ir_chain=(nm,),
            )
            for (nm, s, a) in per_node
        )

    return DebugHandle(
        id=_stable_id(source, aten_op, ir_chain),
        source=source,
        aten_op=aten_op,
        ir_chain=ir_chain,
        fused_from=fused_from,
        transform_history=_transform_history_of(buffer),
    )


def _transform_history_of(carrier: Any) -> tuple[ProvenanceTransform, ...]:
    """Return a carrier's immutable lower-IR transformation history."""
    return tuple(getattr(carrier, _SPYRE_PROV_HISTORY_ATTR, ()) or ())


def _combined_transform_history(
    *carriers: Any,
) -> tuple[ProvenanceTransform, ...]:
    """Combine histories in source order, coalescing shared prefixes."""
    # Prefix merging is O(n*m), but histories are normally 1-3 records, so the
    # cost is negligible in practice. Keep them unbounded because silent
    # truncation would make incomplete lineage look authoritative; revisit this
    # decision if history sizes ever make the merge cost measurable.
    combined: list[ProvenanceTransform] = []
    for carrier in carriers:
        history = _transform_history_of(carrier)
        common_prefix = 0
        limit = min(len(combined), len(history))
        while (
            common_prefix < limit and combined[common_prefix] == history[common_prefix]
        ):
            common_prefix += 1
        combined.extend(history[common_prefix:])
    return tuple(combined)


def _history_is_subsequence(
    before: tuple[ProvenanceTransform, ...],
    after: tuple[ProvenanceTransform, ...],
) -> bool:
    """Return whether every prior record survives in the same relative order."""
    remaining = iter(after)
    return all(
        any(candidate == transform for candidate in remaining) for transform in before
    )


def _set_transform_history(
    carrier: Any, history: tuple[ProvenanceTransform, ...]
) -> None:
    """Set an immutable transformation history on a provenance carrier."""
    object.__setattr__(carrier, _SPYRE_PROV_HISTORY_ATTR, history)


def _append_transform(
    carriers: Sequence[Any], new: Any, transform: ProvenanceTransform
) -> None:
    """Combine source/destination histories and append one new transform."""
    history = _combined_transform_history(*carriers, new)
    _set_transform_history(new, (*history, transform))


def _union_origins(src: Any, dst: Any) -> None:
    """Union src's origins into dst's origins in place.

    ``origins`` is a mutable ``OrderedSet`` on Buffer/ComputedBuffer; unioning
    in place (as graph_editor and Inductor lowering do) preserves the container
    type and any origins dst already accumulated, rather than rebinding the
    field to a plain ``set``.
    """
    src_origins = getattr(src, "origins", None)
    dst_origins = getattr(dst, "origins", None)
    if src_origins and dst_origins is not None:
        dst_origins.update(src_origins)


def preserve_provenance(
    old: Any,
    new: Any,
    pass_name: str,
    reason: str | None = None,
) -> None:
    """1:1 rewrite: carry provenance and append a rewrite record.

    Targets Buffer/ComputedBuffer, whose ``origins`` container is mutable.
    ``origins`` is unioned; ``origin_node`` keeps a value deliberately set by
    the pass. Transformation histories are combined in old-then-new order, so
    destination records remain the most recent without discarding the old
    buffer's lineage. ``pass_name`` identifies the pass performing the rewrite.
    """
    _union_origins(old, new)
    node = getattr(old, "origin_node", None)
    if node is not None and getattr(new, "origin_node", None) is None:
        object.__setattr__(new, "origin_node", node)
    transform = ProvenanceTransform(
        kind="rewrite",
        pass_name=pass_name,
        reason=reason,
    )
    _append_transform((old,), new, transform)


def merge_provenance(
    sources: Sequence[Any],
    new: Any,
    pass_name: str,
    reason: str | None = None,
) -> None:
    """n->1 fusion: union origins, clear a stale primary, append a record."""
    for source in sources:
        _union_origins(source, new)
    # A fused buffer has no intrinsically authoritative constituent. Clear any
    # primary inherited from the object used to initialize ``new``. The builder
    # then derives source and ATen headlines independently from the full set: a
    # field is populated only when there is one distinct non-None value;
    # conflicting values stay None, and fused_from remains authoritative.
    object.__setattr__(new, "origin_node", None)
    transform = ProvenanceTransform(
        kind="fusion",
        pass_name=pass_name,
        reason=reason,
    )
    _append_transform(sources, new, transform)


def decompose_provenance(
    old: Any,
    news: Sequence[Any],
    pass_name: str,
    reason: str | None = None,
    *,
    inherit_origins: bool = True,
) -> None:
    """1->n decomposition: each child inherits and extends old's provenance.

    By default each child inherits the parent's origins and primary node. Pass
    inherit_origins=False when the transformation creates distinct semantic FX
    origins for its children; their own origins remain authoritative while the
    parent's transformation history still flows into each child.
    """
    node = getattr(old, "origin_node", None)
    transform = ProvenanceTransform(
        kind="decomposition",
        pass_name=pass_name,
        reason=reason,
    )
    for child in news:
        if inherit_origins:
            _union_origins(old, child)
            if node is not None and getattr(child, "origin_node", None) is None:
                object.__setattr__(child, "origin_node", node)
        _append_transform((old,), child, transform)


# Passes whose newly-created buffers are legitimately source-less
# (compiler-generated, analogous to LLVM getCompilerGenerated). This exemption
# applies only to creation; it must not hide provenance loss on existing buffers
# reconstructed by the same pass.
SOURCELESS_CREATION_PASSES = frozenset(
    {
        "insert_restickify",
        "insert_post_mutation_restickify",
        "insert_bmm_padding",
    }
)

# Entries use the observer-visible name from passes._get_pass_name. For a pass
# hidden behind an @_runs wrapper, this is the wrapper name (for example,
# ``_maybe_coarse_tile``), not the wrapped function name.

# Passes that intentionally remap provenance on existing buffers. Keep this
# separate and conservative: add an entry only after verifying that dropping an
# old origin or origin_node is part of the pass's declared rewrite semantics.
INTENTIONAL_PROVENANCE_REMAP_PASSES: frozenset[str] = frozenset()

# Passes that intentionally delete pre-existing buffers without forwarding
# their provenance. Other removals must leave the removed unit's origins and
# history on at least one surviving carrier.
INTENTIONAL_PROVENANCE_REMOVAL_PASSES = frozenset({"deadcode_elimination"})

# Warn at most once per pass name per compile to avoid log spam. Reset by the
# Spyre pass pipelines at the start of each run via reset_provenance_warnings().
# Pass pipelines currently run serially, so this module-level mutable set does
# not require synchronization; revisit if concurrent pass execution is added.
_warned_passes: set[str] = set()


def reset_provenance_warnings() -> None:
    """Clear the per-pass warning dedup.

    Called by the Spyre pass pipelines at the start of each run so the observer's
    "warn once per pass" behavior is per-compile, not per-process: a later compile
    that hits the same regression must warn again.
    """
    _warned_passes.clear()


def _provenance_enabled() -> bool:
    """Return whether lower-IR provenance-drop detection is enabled."""
    try:
        # Import lazily so provenance construction remains independent of
        # Inductor's optional debug configuration at module import time.
        from torch._inductor import config as inductor_config

        trace_config = getattr(inductor_config, "trace", None)
        level = getattr(trace_config, "provenance_tracking_level", 0)
        return level >= 1
    except Exception:
        # Detection is debug-only and must never break compilation.
        return False


def _iter_prov_units(target: Any, kind: str) -> list:
    """Best-effort enumeration of provenance-bearing buffers in a pass target.

    kind == "graphlowering": target is a GraphLowering; its ``operations`` are
      ir.Operation/ComputedBuffer nodes that carry ``origins``.
    kind == "node": target is a list[BaseSchedulerNode]; scheduler wrappers are
      recursively resolved to their underlying buffers.
    Any other kind (e.g. FX "graph", already observed upstream) yields nothing.
    """
    units: list = []
    try:
        if kind == "graphlowering":
            for op in getattr(target, "operations", []) or []:
                units.append(op)
        elif kind == "node":
            visited: set[int] = set()

            def append_scheduler_units(node: Any) -> None:
                if id(node) in visited:
                    return
                visited.add(id(node))
                buf = getattr(node, "node", None)
                if buf is not None:
                    units.append(buf)
                    return
                get_nodes = getattr(node, "get_nodes", None)
                if callable(get_nodes):
                    nested = list(get_nodes())
                    if len(nested) != 1 or nested[0] is not node:
                        for child in nested:
                            append_scheduler_units(child)
                        return
                if hasattr(node, "origins"):
                    units.append(node)

            for n in target or []:
                append_scheduler_units(n)
    except Exception:
        return []
    return [u for u in units if u is not None]


def _origins_of(unit: Any) -> frozenset:
    return frozenset(getattr(unit, "origins", None) or ())


def _unit_key(unit: Any) -> Any:
    """Stable snapshot key for a provenance-bearing buffer.

    Prefer the buffer name: it is stable when a pass reconstructs a buffer as a
    fresh object, so a same-name replacement is matched to its predecessor and
    origin loss across the reconstruction is detected. Fall back to identity.
    """
    get_name = getattr(unit, "get_name", None)
    if callable(get_name):
        try:
            return get_name()
        except Exception:
            return id(unit)
    return id(unit)


class SpyreGraphTransformObserver:
    """Detects provenance (``origins``) dropped by a Spyre pass.

    Context manager wrapping one pass. On exit it compares each buffer's
    origins, primary node, and transformation history against a pre-pass
    snapshot. It warns once per pass when existing provenance was dropped or a
    new buffer was created without origins. Declared source-less creations and
    intentional remaps use separate exemptions. Best-effort: never raises into
    the compile path. Active when Inductor's
    ``trace.provenance_tracking_level >= 1``; handle construction and explicit
    provenance forwarding remain unconditional.

    It does NOT forward provenance or record transformations; that is the job
    of the explicit helpers and existing buffer-reconstruction path.
    """

    def __init__(self, target: Any, pass_name: str, kind: str) -> None:
        self.target = target
        self.pass_name = pass_name
        self.kind = kind
        self._before: dict[
            Any, tuple[frozenset, Any, tuple[ProvenanceTransform, ...]]
        ] = {}
        self._active = _provenance_enabled() and kind != "graph"

    def __enter__(self) -> "SpyreGraphTransformObserver":
        if not self._active:
            return self
        try:
            for u in _iter_prov_units(self.target, self.kind):
                self._before[_unit_key(u)] = (
                    _origins_of(u),
                    getattr(u, "origin_node", None),
                    _transform_history_of(u),
                )
        except Exception:
            self._active = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Reconcile only on clean pass exit; never suppress a real exception.
        if not self._active or exc_type is not None:
            return
        try:
            self._reconcile()
        except Exception:
            pass  # provenance is best-effort

    def _reconcile(self) -> None:
        creation_exempt = self.pass_name in SOURCELESS_CREATION_PASSES
        remap_exempt = self.pass_name in INTENTIONAL_PROVENANCE_REMAP_PASSES
        removal_exempt = self.pass_name in INTENTIONAL_PROVENANCE_REMOVAL_PASSES
        after_units = _iter_prov_units(self.target, self.kind)
        after_keys = {_unit_key(unit) for unit in after_units}
        for u in after_units:
            snap = self._before.get(_unit_key(u))
            now = _origins_of(u)
            if snap is not None:
                before, before_node, before_history = snap
                # Warn on ANY lost origin, not only a complete drop: a fused
                # buffer going {a, b} -> {a} silently loses one source. A pass
                # that legitimately remaps origins can declare that separately
                # without suppressing source-less creation checks.
                lost = before - now
                if lost and not remap_exempt:
                    self._warn(
                        f"dropped {len(lost)} of {len(before)} provenance "
                        f"origin(s) on an existing buffer"
                    )
                # origin_node is authoritative for handle source/aten; a pass
                # that had one and cleared it silently loses attribution.
                now_node = getattr(u, "origin_node", None)
                now_history = _transform_history_of(u)
                primary_was_merged = before_node in now and any(
                    transform.kind == "fusion" and transform.pass_name == self.pass_name
                    for transform in now_history[len(before_history) :]
                )
                if (
                    before_node is not None
                    and now_node is None
                    and not primary_was_merged
                    and not remap_exempt
                ):
                    self._warn(
                        "dropped origin_node (authoritative provenance) on an "
                        "existing buffer"
                    )
                if (
                    not _history_is_subsequence(before_history, now_history)
                    and not remap_exempt
                ):
                    self._warn(
                        "dropped lower-IR transformation history on an existing buffer"
                    )
            elif not now and not creation_exempt:
                self._warn(
                    "created a buffer without provenance; use "
                    "preserve_/merge_/decompose_provenance"
                )

        if removal_exempt or remap_exempt:
            return
        for key, snap in self._before.items():
            if key in after_keys:
                continue
            before, before_node, before_history = snap
            if not before and before_node is None and not before_history:
                continue
            if any(
                self._removed_provenance_survives(
                    before,
                    before_node,
                    before_history,
                    unit,
                )
                for unit in after_units
            ):
                continue
            self._warn("removed a buffer without forwarding its provenance")

    @staticmethod
    def _removed_provenance_survives(
        before: frozenset,
        before_node: Any,
        before_history: tuple[ProvenanceTransform, ...],
        unit: Any,
    ) -> bool:
        """Return whether one survivor carries a removed unit's provenance."""
        now = _origins_of(unit)
        if not before.issubset(now):
            return False
        if before_node is not None:
            now_node = getattr(unit, "origin_node", None)
            if now_node is not before_node and before_node not in now:
                return False
        return _history_is_subsequence(
            before_history,
            _transform_history_of(unit),
        )

    def _warn(self, msg: str) -> None:
        if self.pass_name in _warned_passes:
            return
        _warned_passes.add(self.pass_name)
        logger.warning(
            f"[spyre-provenance] pass '{self.pass_name}' {msg}. "
            f"Provenance may be lost for affected kernels."
        )
