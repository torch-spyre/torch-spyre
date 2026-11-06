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

"""Rewrite reshapes that cut the stick dim off a stick boundary.

A contiguous Spyre tensor keeps its innermost dim on the 64-element (fp16)
stick. A reshape that splits that dim into ``(..., A, B)`` or merges
``(..., A, B)`` into it, with ``B % 64 != 0``, makes rows start mid-stick
(``W=160`` is 2.5 sticks per row). No stick-granular access can address that
in place: compute_coordinates raises ``UnrepresentableAccess`` (issue #1353).

The fix is a stick pivot. Move a dim the reshape leaves alone (``P``) onto the
stick, so the split/merge happens on non-stick dims, where it is an ordinary
view, then move ``P`` back:

    x (..., P, L)  -permute+clone->  (..., L, P)     P on the stick
                   -view->           (..., A, B, P)  split a non-stick dim
                   -rebase_copy->    (..., A, B, P)  dense, separate dims
                   -permute+clone->  (..., P, A, B)  original strides

The two permute+clones are on-device restickify copies; rebase_copy is a
same-stick copy that gives the second restickify separate dense dims to read
(it cannot read a flattened or padded coordinate). The final clone restores
the contiguous strides the original reshape produced, so downstream views and
graph outputs see exactly what they saw before. A merge runs the same steps
with the view merging instead of splitting. The reshape goes through
``spyre.reshape_via_cpu`` instead when no dim can serve as ``P``, when the
dtype is not in ``RESTICKIFY_SUPPORTED_DEVICE_DTYPES``, when the pivot
intermediate exceeds the per-core span limit, or when ``P`` spans more than
one stick and nothing non-unit sits in front of the body (see ``_rewrite``).

A sub-stick reshape (``64 -> 8x8``) is handled the same way: the destination
pads every row up to a whole stick, so ``(8, 8)`` spans 8 sticks where
``(64,)`` spans 1, and the layouts differ just as they do for ``W=160``.

The pivot is the default (``config.stick_reshape_on_device``; set
``SPYRE_STICK_RESHAPE_ON_DEVICE=0`` to force the host path). It measured about
twice as fast as the host round trip (see the flag's comment in config.py).
Either way the reshape is caught here instead of failing (split) or
miscompiling (merge) downstream.
"""

import operator
from math import prod
from typing import Optional

import torch
import torch.fx
from torch.utils import _pytree as pytree

from torch_spyre._C import get_device_dtype, get_elem_in_stick

from . import config
from .constants import RESTICKIFY_SUPPORTED_DEVICE_DTYPES
from .logging_utils import get_inductor_logger
from .padding import round_up_to_stick
from .pass_utils import copy_fx_custom_meta
from .work_division import MAX_SPAN_BYTES

aten = torch.ops.aten

logger = get_inductor_logger("stick_reshape_pivot")

_RESHAPE_OPS = (
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
)

_CONVERT_OPS = (
    torch.ops.prims.convert_element_type.default,
    aten._to_copy.default,
)


def _static_shape(val) -> Optional[list[int]]:
    shape = getattr(val, "shape", None)
    if shape is None or not all(isinstance(s, int) for s in shape):
        return None
    return list(shape)


def _trailing_group(shape: list[int], extent: int) -> Optional[int]:
    """Largest j with ``prod(shape[j:]) == extent`` and at least two pieces."""
    tail = shape[-1]
    for j in range(len(shape) - 2, -1, -1):
        tail *= shape[j]
        if tail == extent:
            return j
        if tail > extent:
            return None
    return None


def _strip_trailing_ones(shape: list[int]) -> list[int]:
    """``shape`` without its trailing size-1 dims, keeping at least one dim."""
    end = len(shape)
    while end > 1 and shape[end - 1] == 1:
        end -= 1
    return shape[:end]


def _stick_reshape_kind(
    in_shape: list[int], out_shape: list[int], eps: int
) -> Optional[tuple[str, list[int]]]:
    """Classify a reshape that cuts the stick dim off a stick boundary.

    Returns ``("split", P)`` when the innermost input dim ``L`` becomes the
    trailing output dims ``Q`` (``out = P + Q``), or ``("merge", P)`` when the
    trailing input dims ``R`` become the innermost output dim (``in = P + R``),
    and the innermost piece is not a multiple of ``eps``. The outer dims may be
    reshaped too (``(C, L) -> (1, C, H, W)``); that part never touches the
    stick and is done as a separate plain view. Returns None otherwise.
    """
    if prod(in_shape) == 0:
        return None  # empty tensor: no element to move, and 0 divides nothing
    if in_shape == out_shape or in_shape[-1:] == out_shape[-1:]:
        return None  # stick dim untouched
    # A sub-stick reshape (64 -> 8x8) is NOT free either: the destination pads
    # every row to a whole stick, so (8, 8) occupies 8 sticks where (64,)
    # occupies 1. Measured on upstream: every split of one stick, and every
    # merge back, fails with "no supported output layout". So these are
    # rewritten like any other unaligned reshape.
    j = _trailing_group(out_shape, in_shape[-1])
    if j is not None and out_shape[-1] % eps != 0:
        return "split", out_shape[:j]
    j = _trailing_group(in_shape, out_shape[-1])
    if j is not None and in_shape[-1] % eps != 0:
        return "merge", in_shape[:j]
    return None


def _pivot_dim(prefix: list[int]) -> Optional[int]:
    """Pick the unchanged dim to put on the stick: the largest non-unit one.

    A larger pivot wastes less of its padded stick; ties go to the innermost.
    """
    candidates = [i for i, s in enumerate(prefix) if s > 1]
    if not candidates:
        return None
    return max(candidates, key=lambda i: (prefix[i], i))


def _feeds_stick_dim_cat(node: torch.fx.Node) -> bool:
    """True when a direct user is a ``cat`` along the innermost dim.

    The pivot leaves its result with a padded stick dim (56 stored in 64).
    Concatenating along that dim asks the cat lowering to restickify such an
    operand and it asserts ("restickify restore: expected padding-prepended
    size-64 gap dim"). The host path hands the cat a dense operand instead,
    which works. Related: #5196, #5216.
    """
    out_rank = len(node.meta["val"].shape)
    if out_rank == 0:
        return False
    for user in node.users:
        if user.op != "call_function" or user.target is not aten.cat.default:
            continue
        dim = user.args[1] if len(user.args) > 1 else user.kwargs.get("dim", 0)
        if isinstance(dim, int) and dim % out_rank == out_rank - 1:
            return True
    return False


def _reaches_stick_size_change(node: torch.fx.Node, eps: int) -> bool:
    """True when anything downstream converts to a different stick size.

    float16 and bfloat16 pack 64 to a stick, float32 packs 32, and converting
    between the two reorders elements in the backend (#3999). Measured on
    device: float16 -> float32 is wrong on its own, with no reshape in the
    graph at all and whatever the row width ((2, 320), (2, 64) and (2, 40) all
    mismatch a CPU reference), so the conversion is what is broken, not the
    relayout. Upstream refuses an unaligned reshape in front of one ("no
    supported output layout"); repairing it would hand the broken conversion a
    layout it accepts and turn that refusal into wrong values, which is the
    failure mode this pass exists to remove.

    The walk is transitive: the conversion is just as wrong two ops later
    (measured: ``reshape -> bfloat16 -> float32`` and ``reshape -> mul ->
    float32`` both mismatch), so a direct-user check is not enough. Lift this
    once #3999 lands.
    """
    seen = {node}
    queue = [node]
    while queue:
        for user in queue.pop().users:
            if user in seen:
                continue
            seen.add(user)
            queue.append(user)
            if user.op != "call_function" or user.target not in _CONVERT_OPS:
                continue
            dtype = getattr(user.meta.get("val"), "dtype", None)
            if dtype is None:
                return True  # unknown conversion: assume it changes the stick
            try:
                if get_elem_in_stick(dtype) != eps:
                    return True
            except Exception:
                return True  # dtype with no stick layout of its own
    return False


def _escaping_nodes(graph: torch.fx.Graph) -> set:
    """Every node the graph returns, flattened out of the output structure."""
    try:
        out = graph.output_node()
    except (AssertionError, RuntimeError, StopIteration):
        return set()
    return {a for a in pytree.tree_leaves(out.args) if isinstance(a, torch.fx.Node)}


def _alias_source(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The tensor ``node`` is a view of, or None if it owns its storage.

    Read off the operator schema rather than a hand-kept list of view ops, so
    permute, transpose, squeeze, slice, detach and anything added later are all
    covered: a return that aliases without writing is exactly a view.
    """
    if node.op != "call_function":
        return None
    if node.target is operator.getitem:
        # split/unbind/chunk hand back their pieces through getitem, and those
        # pieces alias the tensor they came from. getitem has no schema of its
        # own, so defer to the producer: it aliases for those three, and not
        # for a multi-value op such as max.dim.
        parent = node.args[0] if node.args else None
        if isinstance(parent, torch.fx.Node) and _alias_source(parent) is not None:
            return parent
        return None
    schema = getattr(node.target, "_schema", None)
    if schema is None or not schema.returns:
        return None
    info = schema.returns[0].alias_info
    if info is None or info.is_write:
        return None
    src = node.args[0] if node.args else None
    return src if isinstance(src, torch.fx.Node) else None


def _alias_base(node: torch.fx.Node) -> torch.fx.Node:
    """Follow views up to the node that actually owns the storage."""
    seen = {node}
    while (src := _alias_source(node)) is not None and src not in seen:
        seen.add(src)
        node = src
    return node


def _alias_closure(node: torch.fx.Node) -> set:
    """``node`` plus every view derived from it, transitively."""
    out = {node}
    stack = [node]
    while stack:
        n = stack.pop()
        for u in n.users:
            if u not in out and _alias_source(u) is n:
                out.add(u)
                stack.append(u)
    return out


def _breaks_escaping_alias(node: torch.fx.Node, escaping: dict) -> bool:
    """True when rewriting would turn an alias the caller can see into a copy.

    A reshape left as a view shares storage with its base. Every repair here
    materialises it instead, so if the reshape reaches an output *and* some
    other alias of the same storage also reaches an output, a caller that
    mutates one no longer sees the change through the other::

        def fn(x):
            y = x * 2
            return y, y.view(4, 8, 8)             # base and view
            return y.t(), y.view(4, 8, 8)         # sibling view and view
            return y, y.view(4, 8, 8).transpose(1, 2)   # base and view-of-view

    Both ends are followed through views, so the reshape may reach the output
    indirectly and the other alias may be any view of the same base. Views
    derived from the reshape itself do not count: materialising it moves them
    all together, so nothing observable changes between them.

    Leaving such a reshape alone reproduces the base compiler exactly, which
    is the conservative choice: correctness of aliasing beats repairing the
    layout.
    """
    closure = _alias_closure(node)
    if closure.isdisjoint(escaping):
        return False
    base = _alias_base(node)
    return any(e not in closure and b is base for e, b in escaping.items())


def _call(graph, target, args, kwargs, src_node, fake_mode):
    node = graph.call_function(target, args=args, kwargs=kwargs)
    arg_vals = [a.meta["val"] if isinstance(a, torch.fx.Node) else a for a in args]
    with fake_mode:
        node.meta["val"] = target(*arg_vals, **kwargs)
    copy_fx_custom_meta(src_node, node)
    return node


def _rewrite(
    node: torch.fx.Node,
    kind: str,
    prefix: list[int],
    out_shape: list[int],
    final_shape: list[int],
) -> None:
    """Rewrite ``node``. ``out_shape`` has trailing size-1 dims stripped and
    drives the pivot; ``final_shape`` is what the node's users expect."""
    graph = node.graph
    src = node.args[0]
    src_val = src.meta["val"]
    fake_mode = src_val.fake_mode
    in_shape = _strip_trailing_ones(list(src_val.shape))
    # The pivot always runs on the input's own outer dims and produces
    # ``pivot_out``; any reshape of the outer dims happens afterwards, as a
    # plain view of the pivot's contiguous result. (A view in front of the
    # first restickify is read through to its base, which the backend cannot
    # map.)
    if kind == "split":
        body = out_shape[len(prefix) :]
        prefix = in_shape[:-1]
    else:
        body = [out_shape[-1]]
    pivot_out = prefix + body
    restickifiable = (
        get_device_dtype(src_val.dtype) in RESTICKIFY_SUPPORTED_DEVICE_DTYPES
    )
    on_device = (
        config.stick_reshape_on_device
        and restickifiable
        and not _feeds_stick_dim_cat(node)
    )
    p = _pivot_dim(prefix) if on_device else None
    if p is not None:
        eps = get_elem_in_stick(src_val.dtype)
        # The restickify back reads the intermediate wrongly when the body has
        # no non-unit dim in front of it and the pivot spans more than one
        # stick. Measured: pivot <= 64 is correct, pivot > 64 gives wrong
        # values, whether or not the pivot is stick-aligned (128 and 192 fail
        # too), and a size-1 outer dim does not help. Take the host path.
        # TODO(issue#TBD): lift once the restickify restore is understood.
        if prefix[p] > eps and not any(s > 1 for i, s in enumerate(prefix) if i != p):
            p = None
    if p is not None:
        # The pivot restickify splits work only along the new stick dim, so
        # each core spans about the whole intermediate. Past the per-core
        # span limit (e.g. a 1024-channel 379x519 conv output, ~400 MB) take
        # the host round trip instead of failing work division.
        #
        # Both intermediates are sized, not just the first: the restickify out
        # pads the pivot dim to a stick, and the rebase_copy and restickify
        # back pad the new innermost dim instead. A small innermost dim is the
        # bigger of the two by far -- 7 padded to 64 is 9x -- so checking only
        # the pivot would let work division fail where this means to fall back.
        esize = src_val.element_size()
        dt = src_val.dtype
        padded_pivot = round_up_to_stick(prefix[p], dt)
        pivot_bytes = prod(in_shape) // prefix[p] * padded_pivot * esize
        padded_body = round_up_to_stick(body[-1], dt)
        body_bytes = prod(in_shape) // body[-1] * padded_body * esize
        if max(pivot_bytes, body_bytes) > MAX_SPAN_BYTES:
            p = None
    if p is not None and len(prefix) > 2:
        # Collapse the dims around the pivot to at most one on each side. The
        # first restickify pads the old stick dim to whole sticks, and with
        # two or more dims in front of it the chosen layout can put that
        # padded dim inside a non-unit outer dim, so the padding falls between
        # the outer slices and the next read lands on the wrong rows.
        # Measured: (1,4,64,400) -> (1,4,64,20,20) and (2,4,64,400) gave
        # wrong values, (4,64,400) and these shapes collapsed did not.
        # Prefixes of one or two dims are left as they are: they were already
        # correct, and collapsing them would add a copy to restore the shape.
        before, after = prod(prefix[:p]), prod(prefix[p + 1 :])
        if before > 1 and after > 1:
            p = None  # not reducible to one outer dim; take the host path
        else:
            tail = in_shape[len(prefix) :]
            pivot = prefix[p]
            prefix = [before] * (before > 1) + [pivot] + [after] * (after > 1)
            p = int(before > 1)
            in_shape = prefix + tail
            pivot_out = prefix + body

    with graph.inserting_before(node):
        if list(src_val.shape) != in_shape:
            # Drop the input's trailing size-1 dims first: they carry no
            # elements but would otherwise land on the stick and be padded to
            # a whole one. Re-added by the view to final_shape at the end.
            src = _call(graph, aten.view.default, (src, in_shape), {}, node, fake_mode)
        if p is None:
            new = _call(
                graph,
                torch.ops.spyre.reshape_via_cpu.default,
                (src, final_shape),
                {},
                node,
                fake_mode,
            )
        else:
            # Move pivot dim p to the end so it becomes the stick dim.
            keep = [i for i in range(len(in_shape)) if i != p]
            moved = _call(
                graph, aten.permute.default, (src, keep + [p]), {}, node, fake_mode
            )
            moved = _call(
                graph,
                aten.clone.default,
                (moved,),
                {"memory_format": torch.contiguous_format},
                node,
                fake_mode,
            )
            # Split/merge the old stick dims, now non-stick, under the pivot.
            rest = [s for i, s in enumerate(prefix) if i != p]
            viewed = _call(
                graph,
                aten.view.default,
                (moved, rest + body + [prefix[p]]),
                {},
                node,
                fake_mode,
            )
            # Re-base: materialize the view so the restickify below reads
            # separate dense dims, not a flattened (or padded) coordinate.
            viewed = _call(
                graph,
                torch.ops.spyre.rebase_copy.default,
                (viewed,),
                {},
                node,
                fake_mode,
            )
            # Put the pivot back at position p: viewed dims are
            # rest (k-1) + body + [pivot].
            n_out = len(pivot_out)
            back = list(range(p)) + [n_out - 1] + list(range(p, n_out - 1))
            new = _call(
                graph, aten.permute.default, (viewed, back), {}, node, fake_mode
            )
            new = _call(
                graph,
                aten.clone.default,
                (new,),
                {"memory_format": torch.contiguous_format},
                node,
                fake_mode,
            )
            if pivot_out != final_shape:
                new = _call(
                    graph, aten.view.default, (new, final_shape), {}, node, fake_mode
                )
                if _strip_trailing_ones(final_shape) != pivot_out:
                    # Materialise the outer merge. Left as a plain view,
                    # Inductor folds it away and consumers see the pivot's
                    # higher-rank intermediate (a depthwise conv then reports
                    # "expected a 3D or 4D activation, got [1,4,64,20,20]").
                    #
                    # Skipped when only trailing size-1 dims were re-added:
                    # the view moves nothing, and materialising it would pad
                    # an innermost 1 up to a whole stick -- 64x the buffer for
                    # no gain.
                    new = _call(
                        graph,
                        torch.ops.spyre.rebase_copy.default,
                        (new,),
                        {},
                        node,
                        fake_mode,
                    )
    node.replace_all_uses_with(new)
    graph.erase_node(node)
    logger.info(
        "stick_reshape_pivot: %s %s %s -> %s via %s",
        node.name,
        kind,
        in_shape,
        out_shape,
        "reshape_via_cpu" if p is None else f"pivot on dim {p}",
    )


def pivot_unaligned_stick_reshapes(graph: torch.fx.Graph) -> None:
    """Rewrite every Spyre reshape that cuts the stick dim off a stick boundary."""
    escaping: Optional[dict] = None  # built once, only if a candidate survives
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _RESHAPE_OPS:
            continue
        src = node.args[0]
        if not isinstance(src, torch.fx.Node):
            continue
        src_val = src.meta.get("val")
        out_val = node.meta.get("val")
        if src_val is None or out_val is None:
            continue
        if getattr(src_val, "device", None) is None or src_val.device.type != "spyre":
            continue
        # The stick is the innermost host dim only for a contiguous input; a
        # non-contiguous reshape materializes its own copy.
        if not src_val.is_contiguous():
            continue
        in_shape = _static_shape(src_val)
        out_shape = _static_shape(out_val)
        if not in_shape or not out_shape:
            continue
        try:
            eps = get_elem_in_stick(src_val.dtype)
        except Exception:
            continue  # dtype without a stick layout
        # Classify on shapes without trailing size-1 dims. They hold no
        # elements, but an innermost 1 is never a multiple of the stick, so
        # leaving them on makes every keepdim view look unaligned.
        work_in = _strip_trailing_ones(in_shape)
        work_out = _strip_trailing_ones(out_shape)
        found = _stick_reshape_kind(work_in, work_out, eps)
        if found is None:
            continue
        if _reaches_stick_size_change(node, eps):
            continue
        # Last, because it is the only check that walks the graph: the cheap
        # shape and device filters reject almost everything before this.
        if escaping is None:
            escaping = {e: _alias_base(e) for e in _escaping_nodes(graph)}
        if _breaks_escaping_alias(node, escaping):
            continue
        kind, prefix = found
        _rewrite(node, kind, prefix, work_out, out_shape)
    graph.lint()
