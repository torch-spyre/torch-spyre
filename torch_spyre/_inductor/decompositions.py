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


from contextlib import contextmanager

from typing import Optional, Sequence, Union
import torch
import torch._decomp as decomp
# from torch._inductor.decomposition import decompositions

import threading

# A module-level lock + nesting counter to make the CM reentrant/thread-safe
_decompositions_lock = threading.RLock()
_decompositions_nesting = 0

# Dictionary for Spyre-specific decompositions
spyre_decompositions: dict = {}

# Exclude specific Inductor default decompositions on Spyre.
# Some Inductor decompositions do not work reliably on the Spyre backend yet.
# We disable them here and rely on implicit fallbacks to eager ops instead. Once
# the blocking issues are resolved, these exclusions can be removed.
spyre_decompositions_to_exclude = [
    # The default decomposition for torch.new_ones (defined in pytorch/torch/refs/__init__.py)
    # uses torch.full, which is not yet supported in Spyre eager mode.
    # See: https://github.com/torch-spyre/torch-spyre/issues/128#issuecomment-3576168221
    torch.ops.aten.new_ones,
]


def register_spyre_decomposition(
    ops: Union[torch._ops.OperatorBase, list],
):
    """
    Register decompositions specifically for Spyre device.
    These will only be active when compiling for the Spyre device.
    """
    return decomp.register_decomposition(ops, spyre_decompositions)


# Context manager that enables spyre specific decompositions in addition to PyTorch in-tree decompositions
@contextmanager
def enable_spyre_decompositions():
    """
    CM that enables Spyre decompositions:
      - Temporarily adds relevant Spyre decompositions to global decompositions dictionary
      - Restore original decompositions on exit

    This CM is reentrant and safe under nested usage.
    """
    global _decompositions_nesting
    with _decompositions_lock:
        first_enter = (_decompositions_nesting == 0)  # fmt: skip
        _decompositions_nesting += 1
        
        if first_enter:
            from torch_spyre.fallbacks import fallback_ops
            from torch._inductor.decomposition import decompositions
            
            saved_intree_decompositions = {}
            for (
                spyre_decompositions_op,
                spyre_decompositions_impl,
            ) in spyre_decompositions.items():
                if spyre_decompositions_op in decompositions:
                    saved_intree_decompositions[spyre_decompositions_op] = decompositions[
                        spyre_decompositions_op
                    ]

            # Attach to the function so we can restore on last exit
            enable_spyre_decompositions._saved_decompositions = saved_intree_decompositions
            
            # Remove the selected decompositions from Inductor's registry for Spyre.
            torch._decomp.remove_decompositions(
                decompositions, spyre_decompositions_to_exclude
            )

            # Remove decompositions for fallback ops defined in fallbacks.py
            torch._decomp.remove_decompositions(decompositions, fallback_ops)

        try:
            yield
        finally:
            _decompositions_nesting -= 1
            last_exit = (_decompositions_nesting == 0)  # fmt: skip
            if last_exit:
                # Reset the saved in-tree lowerings if needed
                saved_intree_decompositions = getattr(
                    enable_spyre_decompositions, "_saved_decompositions", {}
                )
                for (
                    spyre_decompositions_op,
                    spyre_decompositions_impl,
                ) in spyre_decompositions.items():
                    if spyre_decompositions_op in saved_intree_decompositions:
                        decompositions[spyre_decompositions_op] = saved_intree_decompositions[
                            spyre_decompositions_op
                        ]
                    else:
                        decompositions.pop(spyre_decompositions_op, None)
                # Clean up
                enable_spyre_decompositions._saved_lowerings = {}

@register_spyre_decomposition([torch.ops.spyre.compact])
def compact_decomp(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.slice(torch.ops.spyre.swap(x))


@register_spyre_decomposition([torch.ops.spyre.layer_norm])
def layernorm_decomp(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


# TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# a one-element Spyre tensor. This currently fails because Spyre does not handle
# single-element tensors well.
# Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
#
# To avoid constant folding, we introduce a custom op `spyre::full` that runs
# torch.full on CPU and copies the result to Spyre. Remove this workaround once
# Spyre supports one-element tensors.
@register_spyre_decomposition([torch.ops.aten.full])
def full_decomp(
    size: list[Union[int, torch.SymInt]],
    fill_value: torch.types.Number,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.spyre.full(size, fill_value, device, dtype=dtype)


"""
Hook torch.nn.functional.layer_norm to select spyre optimized version where applicable
"""
orig_layer_norm = torch.nn.functional.layer_norm


def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    from .patches import _should_run_on_spyre

    if _should_run_on_spyre([input]) and len(normalized_shape) == 1:
        return torch.ops.spyre.layer_norm(input, normalized_shape, weight, bias, eps)
    else:
        return orig_layer_norm(input, normalized_shape, weight, bias, eps)


torch.nn.functional.layer_norm = spyre_layer_norm

orig_gelu = torch.nn.functional.gelu


def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    from .patches import _should_run_on_spyre

    if _should_run_on_spyre([input]):
        return torch.ops.spyre.gelu(input, approximate)
    else:
        return orig_gelu(input, approximate=approximate)


torch.nn.functional.gelu = spyre_gelu


orig_softplus = torch.nn.functional.softplus


def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    from .patches import _should_run_on_spyre

    if _should_run_on_spyre([input]):
        return torch.ops.spyre.softplus(input, beta, threshold)
    else:
        return orig_softplus(input, beta, threshold)


torch.nn.functional.softplus = spyre_softplus


@register_spyre_decomposition([torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Tensor_out])
def gt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement greaterthan in the backend compiler
    out_ge = torch.ge(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_ge, out_ne, out=out).to(dtype=torch.bool)


@register_spyre_decomposition([torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Tensor_out])
def lt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement lessthan in the backend compiler
    out_le = torch.le(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_le, out_ne, out=out).to(dtype=torch.bool)
