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
from typing import Any

import regex

from torch_spyre._inductor.op_spec import DebugHandle, SourceLoc


_FRAME_RE = regex.compile(r'File "([^"]+)", line (\d+)')


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
    )
