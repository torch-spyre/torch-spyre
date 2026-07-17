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

from typing import Any, Callable, Optional, Sequence

from sympy import Expr
import torch
from torch._inductor.utils import ir_dataclass
from torch._inductor.ir import (
    FixedLayout,
    IRNode,
    Reduction,
    ReductionHint,
    TensorBox,
)
from torch_spyre._C import SpyreTensorLayout

from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
)
from torch._inductor.virtualized import V
import sympy
from torch.utils._ordered_set import OrderedSet
import torch._inductor.ir as ir


@ir_dataclass
class SpyreReduction(Reduction):
    """
    This class extends Reduction with an op_info to enable spyre-specific information
    to be passed from lowering to codegen for reduction operations.

    We believe this is needed because reduction operations do not go through the same
    virtualized ops API as pointwise operations do after lowering.
    TODO: validate this belief.
    """

    op_info: Any

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type,
        op_info=None,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: Optional[IRNode] = None,
    ) -> TensorBox:
        return TensorBox.create(
            SpyreReduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=inner_fn,
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
                op_info=op_info,
            )
        )


class FixedTiledLayout(FixedLayout):
    """
    A Tensor layout for a tensor that is on a Spyre device.
    It augments FixedLayout (the "host" tensor layout) with
    the device tensor layout and the information needed to map between them.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: list[Expr],
        stride: list[Expr],
        device_layout: SpyreTensorLayout,
    ) -> None:
        super().__init__(device, dtype, size, stride)
        self.device_layout: SpyreTensorLayout = device_layout
        self.allocation: dict[str, Any] = {}
        self.per_tile_fixed: bool = False

    def __str__(self) -> str:
        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}, device_layout={self.device_layout})"
        )

    __repr__ = __str__


def _resize_device_layout(
    orig_stl,
    old_host_size: list[int],
    new_host_size: list[int],
    stick_host_dim: int | None = None,
):
    """Derive a new SpyreTensorLayout for a resized host buffer.

    Used in two directions:

    * **shrink** (``_divide_ranges``): the buffer is the same physical
      allocation; coarse tiling narrows the per-tile iteration range.
      ``device_size`` entries for non-stick dims shrink to reflect the smaller
      per-tile extents.
    * **grow** (``_allocate_full_buffer``): a full-sized scatter-target buffer
      is allocated to match a per-tile source.  ``device_size`` entries grow
      back to the full extent.  The stick orientation (transposed vs row-major,
      ``element_arrangement``) is propagated verbatim so both buffers agree on
      physical layout and scatter-copy address arithmetic is correct.

    ``stride_map`` semantics: a value of ``-1`` means "this device dimension has
    extent 1 and is never stepped through; its stride is undefined."  When
    growing back from a singleton (``orig_sm[j] == -1``), the stride is
    recomputed from the new host stride (the rescue arm in Passes 2–4).  For
    non-contiguous (transposed / col-major) dims, the *physical* stride on
    device is invariant to resizing, so it is left unchanged.

    Device-dim classification (as produced by ``get_generic_stick_layout``):

    * **inner stick** (always ``j == ndev-1``) — ``device_size`` is always
      ``elems_per_stick``; left unchanged.  ``stride_map`` updated only if the
      stick host dim is contiguous or was a singleton.
    * **non-stick dim** — one device dim per non-stick host dim.  Matched to
      host dim ``p`` by size (``device_size[j] == old_host_size[p]``), with
      ``stride_map[j]`` used as a tiebreaker when two host dims share the same
      size.  ``device_size`` updated to ``new_host_size[p]``.  ``stride_map``
      updated iff ``orig_sm[j] == old_hs[p]`` (contiguous) or ``== -1``
      (was a singleton being grown).
    * **stick tile-count** — ``ceil(old_host_size[p*] / eps)`` device elements
      spanning the stick host dim ``p*``.  Updated to
      ``ceil(new_host_size[p*] / eps)``.  Same stride-update rule as non-stick.
    * **singleton** (``device_size == 1, stride_map == -1``) — either a sparse
      placeholder (no corresponding host dim) or a non-stick dim tiled to
      extent 1.  Left as-is when there is no host dim to match; matched by
      size-1 to a host dim of size 1 when one exists (grow path from singleton).

    ``p*`` (the stick host dim) is identified by elimination: the unique host
    dim *not* matched as a non-stick dim.  When a reduction collapses the stick
    axis, all host dims are matched as non-stick and ``pstar`` is ``None``; in
    that case Passes 3 and 4 are skipped (tile-count and inner-stick entries are
    frozen at their collapsed values).

    ``stick_host_dim`` (optional): the authoritative stick host-dim index,
    supplied by callers that know it from named-dim identity.  When given, it is
    used directly as ``p*`` and is excluded from the non-stick candidate pool,
    which resolves same-size-dim collisions (transposed layouts where two host
    dims share a size, e.g. flash-attention QK^T with ``Sq == Skv`` — issue
    #3116) without relying on the ambiguous size-elimination + contiguous-stride
    tiebreak.  When ``None`` (all current callers), behaviour is unchanged.

    Multi-pass algorithm:

    * **Pass 1**: match non-inner-stick device dims to host dims by size.
      Size-1 dims match only to host dims of size 1.  Size > 1 dims match by
      ``device_size == old_host_size[p]``, with stride as tiebreaker when sizes
      collide.  Unmatched dims are candidates for tile-count.
    * **Pass 1b**: fix tile-count / size collisions.  When
      ``ceil(old_host_size[p*] / eps) == old_host_size[q]`` for some non-stick
      dim ``q``, the tile-count dim and dim ``q`` have the same size and Pass 1
      may have provisionally claimed the tile-count dim as a non-stick dim.
      Pass 1b corrects this after ``p*`` is provisionally known: a provisional
      match is reclassified as tile-count when its stride also mismatches the
      expected contiguous non-stick stride.
    * **Pass 2**: update non-stick dims (``matched_host``).
    * **Pass 3**: validate and update tile-count dims (``unmatched_j``).
    * **Pass 4**: update inner stick (``j == ndev-1``).

    ``device_dtype`` and ``element_arrangement`` are copied verbatim from
    ``orig_stl``, preserving EXX2/QFP8/DL16 layouts.

    Raises ``RuntimeError`` if any non-stick device dim matches ambiguously, or
    if the stick host dim cannot be uniquely determined by elimination.
    """
    from torch._inductor.ir import FlexibleLayout

    orig_sm = list(orig_stl.stride_map)
    orig_ds = list(orig_stl.device_size)
    eps = orig_stl.elems_per_stick()
    ndev = len(orig_sm)
    ndim = len(old_host_size)

    # Trust a caller-provided stick_host_dim only when it is consistent with the
    # device tile-count structure: the stick host dim must have a corresponding
    # tile-count device dim of size ceil(size/eps).  Identity recovery can be
    # imperfect for permuted layouts (an ambiguous inner-stick coordinate match),
    # and a wrong label would otherwise *override* correct size-based inference
    # and break reconstruction.  When it does not validate, drop it and fall back.
    if stick_host_dim is not None:
        if not (0 <= stick_host_dim < ndim):
            stick_host_dim = None
        else:
            expected_tc = -(-int(old_host_size[stick_host_dim]) // eps)  # ceil
            if expected_tc not in orig_ds[:-1]:
                stick_host_dim = None

    old_hs = [int(s) for s in FlexibleLayout.contiguous_strides(old_host_size)]
    new_hs = [int(s) for s in FlexibleLayout.contiguous_strides(new_host_size)]

    new_ds = list(orig_ds)
    new_sm = list(orig_sm)

    # Pass 1: see docstring.
    matched_host = {}  # j → p (non-stick matches, provisional for size>1)
    unmatched_j = []  # device dims not matched → tile-count / placeholder

    for j in range(ndev - 1):  # j == ndev-1 is always inner stick
        dsz = orig_ds[j]
        if dsz == 1:
            size1_cands = [p for p in range(ndim) if old_host_size[p] == 1]
            # The authoritative stick host dim is never a non-stick match.
            if stick_host_dim is not None:
                size1_cands = [p for p in size1_cands if p != stick_host_dim]
            if len(size1_cands) == 1:
                matched_host[j] = size1_cands[0]
            # else: sparse placeholder with no host counterpart — skip silently.
        else:
            size_cands = [p for p in range(ndim) if old_host_size[p] == dsz]
            # When the stick host dim is known (named-dim identity), remove it
            # from the non-stick candidate pool. This resolves same-size
            # collisions (e.g. flash-attn QK^T with Sq == Skv, issue #3116)
            # without falling back to the contiguous-stride tiebreak.
            if stick_host_dim is not None:
                size_cands = [p for p in size_cands if p != stick_host_dim]
            if len(size_cands) == 1:
                # provisional; may be reclassified as tile-count in Pass 1b
                matched_host[j] = size_cands[0]
            elif len(size_cands) > 1:
                stride_cands = [p for p in size_cands if old_hs[p] == orig_sm[j]]
                if len(stride_cands) == 1:
                    matched_host[j] = stride_cands[0]
                else:
                    unmatched_j.append(j)
            else:
                unmatched_j.append(j)  # no size match → tile-count

    # Provisional pstar by elimination (before Pass 1b corrections).
    def _find_pstar(matched):
        matched_p = set(matched.values())
        unmatched_all = [p for p in range(ndim) if p not in matched_p]
        if not unmatched_all:
            return None, unmatched_all
        pstar_cands = [p for p in unmatched_all if old_host_size[p] > 1]
        if not pstar_cands:
            pstar_cands = unmatched_all
        if len(pstar_cands) != 1:
            return None, unmatched_all  # ambiguous
        return pstar_cands[0], unmatched_all

    pstar_provisional, _ = _find_pstar(matched_host)

    # Pass 1b: reclassify tile-count/size collisions; see docstring.
    if pstar_provisional is not None:
        expected_tc = -(-old_host_size[pstar_provisional] // eps)
        for j in list(matched_host):
            p = matched_host[j]
            if orig_ds[j] > 1 and orig_sm[j] != old_hs[p] and orig_ds[j] == expected_tc:
                del matched_host[j]
                unmatched_j.append(j)

    # Final pstar after Pass 1b corrections.
    matched_p = set(matched_host.values())
    unmatched_all = [p for p in range(ndim) if p not in matched_p]
    pstar: int | None
    if stick_host_dim is not None:
        # Authoritative stick host dim from named-dim identity — no inference by
        # elimination. It must not also have been matched as a non-stick dim.
        if stick_host_dim not in unmatched_all:
            raise RuntimeError(
                f"_resize_device_layout: caller-provided stick_host_dim="
                f"{stick_host_dim} was matched as a non-stick device dim "
                f"(matched_host={matched_host}) in {orig_stl!r} "
                f"(old_host_size={old_host_size}). Inconsistent dim identity."
            )
        pstar = stick_host_dim
    elif not unmatched_all:
        # Reduction output: stick dim eliminated, pstar=None.
        # unmatched_j must be empty — no device dims should be unclaimed.
        if unmatched_j:
            raise RuntimeError(
                f"_resize_device_layout: stick host dim is absent from "
                f"old_host_size={old_host_size} but device dims {unmatched_j} "
                f"could not be matched as non-stick dims in {orig_stl!r}. "
                f"This layout is not supported by the device-native reconstruction."
            )
        pstar = None
    else:
        pstar_cands = [p for p in unmatched_all if old_host_size[p] > 1]
        if not pstar_cands:
            pstar_cands = unmatched_all
        if len(pstar_cands) != 1:
            raise RuntimeError(
                f"_resize_device_layout: cannot uniquely identify the stick host dim "
                f"by elimination in {orig_stl!r} (old_host_size={old_host_size}); "
                f"unmatched host dims={unmatched_all} "
                f"(non-singleton candidates={pstar_cands}), "
                f"non-stick device dims={matched_host}. "
                f"This layout is not supported by the device-native reconstruction."
            )
        pstar = pstar_cands[0]

    # Pass 2: update non-stick dims.
    for j, p in matched_host.items():
        new_ds[j] = new_host_size[p]
        if new_host_size[p] == 1:
            new_sm[j] = -1
        elif orig_sm[j] == old_hs[p] or orig_sm[j] == -1:
            new_sm[j] = new_hs[p]
        # else: non-contiguous stride; physical layout is invariant — leave unchanged.

    if pstar is None:  # reduction output: tile-count / inner-stick entries frozen
        return SpyreTensorLayout(
            new_ds, new_sm, orig_stl.device_dtype, orig_stl.element_arrangement
        )

    # Pass 3: update tile-count dims (unmatched_j — all must equal expected tile-count).
    for j in unmatched_j:
        expected_tc = -(-old_host_size[pstar] // eps)  # ceil division
        if orig_ds[j] != expected_tc:
            raise RuntimeError(
                f"_resize_device_layout: device dim {j} "
                f"(stride_map={orig_sm[j]}, device_size={orig_ds[j]}) was not "
                f"matched as a non-stick dim and does not equal the expected "
                f"tile-count {expected_tc} for stick host dim {pstar} "
                f"(old_host_size={old_host_size}) in {orig_stl!r}. "
                f"This layout is not supported by the device-native reconstruction."
            )
        new_ds[j] = -(-new_host_size[pstar] // eps)  # ceil division
        if new_host_size[pstar] == 1:
            new_sm[j] = -1
        elif orig_sm[j] == eps * old_hs[pstar] or orig_sm[j] == -1:
            # tile-count stride = eps * contiguous stride of the stick host dim
            new_sm[j] = eps * new_hs[pstar]
        # else: non-contiguous stick; physical stride invariant.

    # Pass 4: inner stick (j == ndev-1) — device_size is always eps, update stride only.
    j = ndev - 1
    if new_host_size[pstar] == 1:
        new_sm[j] = -1
    elif orig_sm[j] == old_hs[pstar] or orig_sm[j] == -1:
        new_sm[j] = new_hs[pstar]
    # else: non-contiguous stick; physical stride invariant.

    return SpyreTensorLayout(
        new_ds, new_sm, orig_stl.device_dtype, orig_stl.element_arrangement
    )


class SpyreConstantFallback(ir.ExternKernel):
    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        wrapper.generate_const_tensor_fallback(self)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence[str]:
        return []

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self, op_overload: torch._ops.OpOverload, value, dtype, device
    ) -> None:
        cpp_kernel_name = "aoti_torch_constant"
        layout = FixedLayout(device, dtype, [], [])
        super().__init__(
            None,
            layout,
            [],
            (value,),
            python_kernel_name="torch.ops.spyre.constant",
            cpp_kernel_name=cpp_kernel_name,
            op_overload=op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class SpyreEmptyFallback(ir.ExternKernel):
    """IR node for spyre.empty — emits spyre_empty_with_layout via make_buffer_allocation.

    should_allocate() returns True so the wrapper calls make_buffer_allocation.
    SpyrePythonWrapperCodegen.make_buffer_allocation emits
    spyre_empty_with_layout(size, stride, dtype, device_layout) when the layout is
    a FixedTiledLayout; the placeholder FixedLayout set at construction time must be
    replaced with a FixedTiledLayout before codegen runs (lower_pad_sequence does
    this immediately after calling run_node).  If the layout is never upgraded the
    wrapper falls back to the generic CPU allocator, which is incorrect on Spyre.
    codegen() is a no-op because the allocation IS the result — there is no
    separate kernel call.
    """

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        pass

    def should_allocate(self) -> bool:
        layout = self.get_layout()
        if isinstance(layout, FixedTiledLayout) and "pool" in layout.allocation:
            return False
        return True

    def get_mutation_names(self) -> Sequence[str]:
        return []

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def __init__(
        self,
        op_overload: torch._ops.OpOverload,
        size: list[Expr],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        stride = ir.FlexibleLayout.contiguous_strides(size)
        layout = FixedLayout(device, dtype, size, stride)
        super().__init__(
            None,
            layout,
            [],
            (),
            op_overload=op_overload,
        )
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
