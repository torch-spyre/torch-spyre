# Copyright 2026 The Torch-Spyre Authors.
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

"""What torch-spyre supports, read from its live registries.

Nothing here keeps its own list of supported ops. Every fact comes from the
tables the compiler and the eager path use, so the report follows the code:

=========================  ==================================================
Fact                       Source
=========================  ==================================================
decomposition table        ``_inductor.decompositions.get_spyre_decomp_table``
Spyre decompositions       ``_inductor.decompositions.spyre_decompositions``
lowerings (Spyre + in-tree) ``torch._inductor.lowering.lowerings`` inside
                           ``_inductor.lowering.enable_spyre_lowerings()``
Spyre lowerings            ``_inductor.lowering.spyre_lowerings``
CPU fallbacks              ``ops.fallbacks.fallback_ops``
compiled eager kernels     ``ops.eager.COMPILED_OPS``
pointwise codegen          ``_inductor.spyre_kernel.SpyreOpFuncs``
reduction codegen          ``_inductor.constants.SINGLE_INPUT_REDUCTION_OPS``
eager kernels              the dispatcher's PrivateUse1 table
=========================  ==================================================
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Iterable, Optional

import torch
from torch._ops import HigherOrderOperator, OpOverload

from .model import Support, Verdict

_CPU_MARKERS = ("warn_fallback(", '.to("cpu")', ".to('cpu')", ".cpu()")


def op_name(op: Any) -> str:
    """``aten.mm.default`` style name for an op, overload or HOP."""
    if isinstance(op, str):
        # "aten::tril" or "spyre::triu_cpu"
        namespace, _, rest = op.partition("::")
        name, _, overload = rest.partition(".")
        return f"{namespace}.{name}.{overload or 'default'}"
    if isinstance(op, OpOverload):
        return str(op)
    if isinstance(op, HigherOrderOperator):
        return f"higher_order.{op.name()}"
    return str(op)


def _closure_values(fn: Any, depth: int = 0, seen: Optional[set] = None):
    """Values captured by ``fn``'s closure, recursively (bounded)."""
    seen = set() if seen is None else seen
    if id(fn) in seen or depth > 4:
        return
    seen.add(id(fn))
    if isinstance(fn, functools.partial):
        yield fn.func
        yield from _closure_values(fn.func, depth + 1, seen)
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        yield value
        if callable(value):
            yield from _closure_values(value, depth + 1, seen)


def lowering_kind(fn: Callable[..., Any]) -> tuple[str, Optional[str]]:
    """Recognize Inductor's generic lowering factories.

    Returns ``("pointwise", "<ops name>")`` for ``make_pointwise(ops_wrapper(n))``,
    ``("reduction", "<reduction type>")`` for ``make_reduction(t)``, else
    ``("other", None)``.
    """
    values = list(_closure_values(fn))
    names = [getattr(v, "__qualname__", "") for v in values]
    if "make_pointwise.<locals>.inner" in names and "ops_wrapper.<locals>.fn" in names:
        wrapper = values[names.index("ops_wrapper.<locals>.fn")]
        for value in _closure_values(wrapper):
            if isinstance(value, str):
                return "pointwise", value
    if "make_reduction.<locals>.inner" in names:
        inner = values[names.index("make_reduction.<locals>.inner")]
        for value in _closure_values(inner):
            if isinstance(value, str):
                return "reduction", value
    return "other", None


def _source(fn: Any) -> str:
    try:
        return inspect.getsource(inspect.unwrap(fn))
    except (OSError, TypeError):
        return ""


class SpyreRegistry:
    """A snapshot of torch-spyre's op support tables."""

    def __init__(self) -> None:
        import torch_spyre

        # A plain ``import torch`` autoloads the backend; an explicit call
        # covers checkouts that are importable but not installed.
        torch_spyre._autoload()  # type: ignore[attr-defined]  # not in the .pyi

        import torch._inductor.lowering as inductor_lowering
        from torch_spyre._inductor import constants
        from torch_spyre._inductor import decompositions as spyre_decomp
        from torch_spyre._inductor import lowering as spyre_lowering
        from torch_spyre._inductor.spyre_kernel import SpyreOpFuncs
        from torch_spyre.ops import eager, fallbacks

        with spyre_lowering.enable_spyre_lowerings():
            self.lowerings: dict[Any, Callable[..., Any]] = dict(
                inductor_lowering.lowerings
            )
            self.fallback_lowerings = set(inductor_lowering.fallbacks)
        self.spyre_lowerings = set(spyre_lowering.spyre_lowerings)
        self.decomp_table = spyre_decomp.get_spyre_decomp_table()
        self.spyre_decomps = set(spyre_decomp.spyre_decompositions)
        self.cpu_fallbacks = {op_name(op) for op in fallbacks.fallback_ops}
        self.compiled_eager = {
            str(op) for op in eager._compile_kernel_overloads(eager.COMPILED_OPS)
        }
        self.pointwise_handlers = {
            name for name in vars(SpyreOpFuncs) if not name.startswith("_")
        }
        self.reduction_types = set(constants.SINGLE_INPUT_REDUCTION_OPS)

    # -- decompositions ----------------------------------------------------

    def decomposition_source(self, op: Any) -> str:
        return "spyre" if op in self.spyre_decomps else "inductor"

    # -- eager path --------------------------------------------------------

    def eager_verdict(self, op: Any) -> Verdict:
        """How the op runs when called on a Spyre tensor without compiling."""
        if isinstance(op, HigherOrderOperator):
            return Verdict(Support.UNVERIFIED, "higher-order op")
        if not isinstance(op, OpOverload):
            return Verdict(Support.UNKNOWN, f"not an ATen op: {op}")
        name = op_name(op)
        if name in self.cpu_fallbacks:
            return Verdict(
                Support.CPU_FALLBACK, "CPU fallback (torch_spyre/ops/fallbacks.py)"
            )
        has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
        qualified = op.name()
        if op.namespace not in ("aten", "prims") and has_kernel(
            qualified, "PrivateUse1"
        ):
            if self._custom_op_runs_on_cpu(op):
                return Verdict(
                    Support.CPU_FALLBACK, f"custom op `{qualified}` runs on the CPU"
                )
            return Verdict(Support.SPYRE, f"custom op `{qualified}` Spyre kernel")
        if str(op) in self.compiled_eager:
            return Verdict(
                Support.SPYRE, "standalone-compiled kernel (ops/eager.py COMPILED_OPS)"
            )
        if op in self.spyre_decomps:
            return Verdict(
                Support.SPYRE, "Spyre decomposition, compiled on first eager call"
            )
        if has_kernel(qualified, "PrivateUse1"):
            return Verdict(Support.SPYRE, "PrivateUse1 kernel")
        if op.is_view:
            return Verdict(Support.VIEW, "view op (C++ device view)")
        if has_kernel(qualified, "CompositeExplicitAutograd"):
            return Verdict(
                Support.UNVERIFIED,
                "CompositeExplicitAutograd kernel (runs its sub-ops)",
            )
        if has_kernel(qualified, "CompositeImplicitAutograd"):
            return Verdict(
                Support.UNVERIFIED, "CompositeImplicitAutograd (decomposes at dispatch)"
            )
        return Verdict(Support.UNSUPPORTED, "no PrivateUse1 kernel")

    def _custom_op_runs_on_cpu(self, op: OpOverload) -> bool:
        from torch._library.custom_ops import OPDEFS

        opdef = OPDEFS.get(op.name())
        if opdef is None:
            return False
        source = _source(getattr(opdef, "_init_fn", None))
        return any(marker in source for marker in _CPU_MARKERS)

    # -- compiled path -----------------------------------------------------

    def compile_verdict(
        self, op: Any, input_dtypes: Iterable[torch.dtype] = ()
    ) -> Verdict:
        """How Spyre's Inductor handles an op that survived decomposition.

        Follows ``GraphLowering.call_function``: an explicit lowering is used
        if there is one; otherwise Inductor falls back to the op's eager
        kernel on the Spyre device.
        """
        name = op_name(op)
        if name in self.cpu_fallbacks:
            return Verdict(
                Support.CPU_FALLBACK,
                "registered CPU fallback (torch_spyre/ops/fallbacks.py)",
            )
        lowering = self.lowerings.get(op)
        if isinstance(op, OpOverload) and lowering is None:
            # A packet-level registration covers every overload.
            lowering = self.lowerings.get(op.overloadpacket)
        if op in self.spyre_lowerings or (
            isinstance(op, OpOverload) and op.overloadpacket in self.spyre_lowerings
        ):
            assert lowering is not None
            return Verdict(
                Support.SPYRE,
                f"Spyre lowering `{getattr(lowering, '__name__', 'lowering')}`",
                self._dtype_notes(lowering, input_dtypes),
            )
        if lowering is None or op in self.fallback_lowerings:
            how = "Inductor fallback" if lowering is not None else "no lowering"
            eager = self.eager_verdict(op)
            if eager.status is Support.SPYRE:
                return Verdict(
                    Support.EAGER_FALLBACK, f"{how}; runs the eager {eager.mechanism}"
                )
            if eager.status is Support.CPU_FALLBACK:
                return Verdict(Support.CPU_FALLBACK, f"{how}; {eager.mechanism}")
            if eager.status is Support.VIEW:
                return Verdict(Support.VIEW, f"{how}; {eager.mechanism}")
            return Verdict(
                Support.UNSUPPORTED,
                f"{how} and no Spyre eager kernel ({eager.mechanism})",
            )
        if isinstance(op, HigherOrderOperator):
            return Verdict(Support.UNVERIFIED, "in-tree higher-order op lowering")
        if isinstance(op, OpOverload) and op.is_view:
            return Verdict(Support.VIEW, "Inductor view lowering")
        kind, detail = lowering_kind(lowering)
        if kind == "pointwise":
            if detail in self.pointwise_handlers:
                return Verdict(
                    Support.SPYRE,
                    f"in-tree pointwise lowering; Spyre implements `ops.{detail}`",
                )
            return Verdict(
                Support.UNSUPPORTED,
                f"in-tree pointwise lowering emits `ops.{detail}`, which "
                "SpyreOpFuncs does not implement",
            )
        if kind == "reduction":
            if detail in self.reduction_types:
                return Verdict(
                    Support.SPYRE, f"in-tree reduction lowering (`{detail}`)"
                )
            return Verdict(
                Support.UNSUPPORTED,
                f"in-tree reduction lowering of type `{detail}`, which Spyre "
                "code generation does not support",
            )
        return Verdict(
            Support.UNVERIFIED,
            f"in-tree lowering `{getattr(lowering, '__name__', 'lowering')}`, "
            "not verified on Spyre",
        )

    def _dtype_notes(
        self, lowering: Any, input_dtypes: Iterable[torch.dtype]
    ) -> list[str]:
        if torch.int64 not in set(input_dtypes):
            return []
        if "with_int64_fallback" in _source(lowering):
            return ["int64 inputs are computed in fp32 (exact only below 2**24)"]
        return []


def is_structural(op: Any) -> bool:
    """Ops that move no data and need no kernel (tuple indexing)."""
    import operator

    return op is operator.getitem
