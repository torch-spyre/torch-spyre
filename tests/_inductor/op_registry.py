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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class OpAdapter:
    name: str
    fn: Callable[..., Any]
    is_inplace: bool = False
    # pre hook may normalize (args, attrs) (e.g., set dropout(training=False))
    pre: Optional[Callable[[list, dict], Tuple[list, dict]]] = None


# -----------------------------
# Helpers: resolve paths lazily
# -----------------------------
def _resolve_attr_path(root: Any, path: str) -> Any:
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def lazy_torch(path: str) -> Callable[..., Any]:
    """
    Returns a callable that resolves torch.<path> at runtime and calls it.

    Example: lazy_torch("amp.autocast_mode._enter_autocast")
    """

    def _fn(*args, **kwargs):
        target = _resolve_attr_path(torch, path)
        return target(*args, **kwargs)

    _fn.__name__ = f"lazy_torch__{path.replace('.', '_')}"
    return _fn


def _dropout_pre(args: list, attrs: dict) -> tuple[list, dict]:
    # default deterministic behavior unless case overrides
    attrs = dict(attrs)
    attrs.setdefault("training", False)
    return args, attrs


# -----------------------------
# Wrappers for Tensor methods
# -----------------------------
def _tensor_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous()


def _tensor_expand(x: torch.Tensor, shape, *args) -> torch.Tensor:
    if isinstance(shape, (list, tuple)):
        final_shape = list(shape)
    elif args:
        final_shape = [shape] + list(args)
    else:
        final_shape = [shape]
    return x.expand(final_shape)


def _tensor_expand_as(x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return x.expand_as(other)


def _tensor_float(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def _tensor_int(x: torch.Tensor) -> torch.Tensor:
    return x.int()


def _tensor_long(x: torch.Tensor) -> torch.Tensor:
    return x.long()


def _tensor_to(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # allow YAML to pass (dtype/device/etc.) via args/kwargs
    return x.to(*args, **kwargs)


def _tensor_repeat(x: torch.Tensor, rep, *args) -> torch.Tensor:
    if isinstance(rep, (list, tuple)):
        final_reps = list(rep)
    elif args:
        final_reps = [rep] + list(args)
    else:
        final_reps = [rep]
    return x.repeat(final_reps)


def _tensor_repeat_interleave(x: torch.Tensor, rep, **kwargs) -> torch.Tensor:
    return x.repeat_interleave(x, rep, **kwargs)


def _tensor_view(x: torch.Tensor, shape, *args) -> torch.Tensor:
    if isinstance(shape, (list, tuple)):
        final_shape = list(shape)
    elif args:
        final_shape = [shape] + list(args)
    else:
        final_shape = [shape]
    return x.view(final_shape)


def _tensor_permute(x: torch.Tensor, dims, *args) -> torch.Tensor:
    if isinstance(dims, (list, tuple)):
        final_dims = list(dims)
    elif args:
        final_dims = [dims] + list(args)
    else:
        final_dims = [dims]
    return x.permute(final_dims)


def _tensor_transpose(x: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    return x.transpose(dim0, dim1)


def _tensor_squeeze(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.squeeze(dim)


def _tensor_unsqueeze(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.unsqueeze(dim)


def _tensor_item(x: torch.Tensor):
    return x.item()


def _tensor_numel(x: torch.Tensor) -> int:
    return x.numel()


def _tensor_new_ones(x: torch.Tensor, size, *args, **kwargs) -> torch.Tensor:
    # size can be list/tuple/int; forward to Tensor.new_ones
    if isinstance(size, (list, tuple)):
        return x.new_ones(tuple(size), *args, **kwargs)
    return x.new_ones(size, *args, **kwargs)


def _tensor_getitem(x: torch.Tensor, idx):
    return x.__getitem__(idx)


def _tensor_setitem_(x: torch.Tensor, idx, value):
    x.__setitem__(idx, value)
    return x  # treat as in-place op returning mutated tensor


# -----------------------------
# Wrappers for Python operator-like torch names
# -----------------------------
def _torch___and__(a, b):
    return a.__and__(b)


def _torch___eq__(a, b):
    return a.__eq__(b)


def _torch_eq(a, b):
    return a.__eq__(b)


def _torch_le(a, b):
    return a.__le__(b)


def _torch_ne(a, b):
    return a.__ne__(b)


def _torch_gt(a, b):
    return a.__gt__(b)


def _torch_neg(a):
    return -a


def _torch_truediv(a, b):
    return a.__truediv__(b)


# -----------------------------
# Special / internal wrappers
# -----------------------------
def _aten_index(x, indices):
    """
    torch.ops.aten.index has overloads; prefer .Tensor then .default if present.
    Typical signature: aten::index(Tensor self, Tensor?[] indices) -> Tensor
    """
    pkt = torch.ops.aten.index
    if hasattr(pkt, "Tensor"):
        return pkt.Tensor(x, indices)
    if hasattr(pkt, "default"):
        return pkt.default(x, indices)
    # last resort: try calling packet itself
    return pkt(x, indices)


def _scalar_tensor(val, *, dtype=None, device=None):
    # torch.scalar_tensor exists in recent versions
    return torch.scalar_tensor(val, dtype=dtype, device=device)


def _sym_sum(x, dim=None):
    # torch.sym_sum exists for SymInt/SymFloat paths; fall back to torch.sum if absent
    if hasattr(torch, "sym_sum"):
        return torch.sym_sum(x, dim=dim) if dim is not None else torch.sym_sum(x)
    return torch.sum(x, dim=dim) if dim is not None else torch.sum(x)


def _vmap_lazy_load_decompositions():
    # torch._functorch.vmap.lazy_load_decompositions
    return _resolve_attr_path(torch, "_functorch.vmap.lazy_load_decompositions")()


# SDPA / attention internals (these often return context managers or control flags)
def _attn_backend_from_string(s: str):
    return _resolve_attr_path(torch, "nn.attention._backend_from_string")(s)


def _attn_sdpa_kernel(*args, **kwargs):
    return _resolve_attr_path(torch, "nn.attention._sdpa_kernel")(*args, **kwargs)


# -----------------------------
# In-place wrappers (Tensor methods)
# -----------------------------
def _tensor_add_(x: torch.Tensor, other, alpha=1):
    return x.add_(other, alpha=alpha)


def _tensor_copy_(x: torch.Tensor, source: torch.Tensor):
    return x.copy_(source)


def _tensor_scatter_(x: torch.Tensor, dim: int, index: torch.Tensor, src):
    # src can be tensor or scalar
    return x.scatter_(dim, index, src)


def _tensor_masked_fill_(x: torch.Tensor, mask: torch.Tensor, value):
    return x.masked_fill_(mask, value)


def _tensor_index_copy_(
    x: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor
):
    return x.index_copy_(dim, index, source)


# -----------------------------
# OP_REGISTRY: unique ops only
# -----------------------------
OP_REGISTRY: Dict[str, OpAdapter] = {
    "torch.cat": OpAdapter("torch.cat", torch.cat),
    "torch.chunk": OpAdapter("torch.chunk", torch.chunk),
    # Basic math / reductions
    "torch.add": OpAdapter("torch.add", torch.add),
    "torch.Tensor.add": OpAdapter("torch.add", torch.add),
    "operator.__add__": OpAdapter("torch.add", torch.add),
    "torch.sub": OpAdapter("torch.sub", torch.sub),
    "operator.__sub__": OpAdapter("torch.sub", torch.sub),
    "torch.mul": OpAdapter("torch.mul", torch.mul),
    "torch.Tensor.mul": OpAdapter("torch.mul", torch.mul),
    "operator.__mul__": OpAdapter("torch.mul", torch.mul),
    "torch.invert": OpAdapter("torch.bitwise_not", torch.bitwise_not),
    "torch.pow": OpAdapter("torch.pow", torch.pow),
    "torch.div": OpAdapter("torch.div", torch.div),
    "torch.truediv": OpAdapter("torch.truediv", _torch_truediv),
    "torch.neg": OpAdapter("torch.neg", _torch_neg),
    "torch.sum": OpAdapter("torch.sum", torch.sum),
    "torch.mean": OpAdapter("torch.mean", torch.mean),
    "torch.max": OpAdapter("torch.max", torch.max),
    "torch.softmax": OpAdapter("torch.softmax", torch.max),
    "torch.cumsum": OpAdapter("torch.cumsum", torch.cumsum),
    "torch.all": OpAdapter("torch.all", torch.all),
    "torch.numel": OpAdapter("torch.numel", _tensor_numel),
    "torch.exp": OpAdapter("torch.exp", torch.exp),
    "torch.rsqrt": OpAdapter("torch.rsqrt", torch.rsqrt),
    "torch.sigmoid": OpAdapter("torch.sigmoid", torch.sigmoid),
    "torch.sin": OpAdapter("torch.sin", torch.sin),
    "torch.cos": OpAdapter("torch.cos", torch.cos),
    "torch.clamp": OpAdapter("torch.clamp", torch.clamp),
    "torch.where": OpAdapter("torch.where", torch.where),
    "torch.tril": OpAdapter("torch.tril", torch.tril),
    "torch.triu": OpAdapter("torch.triu", torch.triu),
    # Linear algebra
    "torch.matmul": OpAdapter("torch.matmul", torch.matmul),
    "operator.__matmul__": OpAdapter("torch.matmul", torch.matmul),
    "torch.bmm": OpAdapter("torch.bmm", torch.bmm),
    "torch.functional.einsum": OpAdapter(
        "torch.functional.einsum", torch.functional.einsum
    ),
    # Shape / view / layout
    # stride-sensitive ops
    "torch.contiguous": OpAdapter("torch.contiguous", _tensor_contiguous),
    "torch.Tensor.contiguous": OpAdapter("torch.contiguous", _tensor_contiguous),
    "torch.reshape": OpAdapter("torch.reshape", torch.reshape),
    "torch.view": OpAdapter("torch.view", _tensor_view),
    "torch.Tensor.view": OpAdapter("torch.view", _tensor_view),
    "torch.aten.view": OpAdapter("torch.view", _tensor_view),
    "torch.transpose": OpAdapter("torch.transpose", _tensor_transpose),
    "torch.permute": OpAdapter("torch.permute", _tensor_permute),
    "torch.squeeze": OpAdapter("torch.squeeze", _tensor_squeeze),
    "torch.unsqueeze": OpAdapter("torch.unsqueeze", _tensor_unsqueeze),
    "torch.flatten": OpAdapter("torch.flatten", torch.flatten),
    "torch.expand": OpAdapter("torch.expand", _tensor_expand),
    "torch.Tensor.expand": OpAdapter("torch.expand", _tensor_expand),
    "torch.expand_as": OpAdapter("torch.expand_as", _tensor_expand_as),
    "torch.repeat": OpAdapter("torch.repeat", _tensor_repeat),
    "torch.repeat_interleave": OpAdapter(
        "torch.repeat_interleave", _tensor_repeat_interleave
    ),
    "torch.clone": OpAdapter("torch.clone", torch.clone),
    "torch.split": OpAdapter("torch.split", torch.split),
    "torch.functional.split": OpAdapter(
        "torch.functional.split", torch.functional.split
    ),
    "torch.Tensor.copy_": OpAdapter("torch.copy_", _tensor_copy_, is_inplace=True),
    # Indexing / getitem / setitem
    "torch.getitem": OpAdapter("torch.getitem", _tensor_getitem),
    "operator.__getitem__": OpAdapter("torch.getitem", _tensor_getitem),
    "torch.setitem": OpAdapter("torch.setitem", _tensor_setitem_, is_inplace=True),
    "_operator.setitem": OpAdapter(
        "_operator.setitem", _tensor_setitem_, is_inplace=True
    ),
    "torch.ops.aten.index": OpAdapter("torch.ops.aten.index", _aten_index),
    # Scatter / copy / masking
    "torch.scatter": OpAdapter("torch.scatter", torch.scatter),
    "torch.scatter_": OpAdapter("torch.scatter_", _tensor_scatter_, is_inplace=True),
    # "torch.scatter_": OpAdapter("torch.scatter_", torch.Tensor.scatter_, is_inplace=True),
    "torch.index_copy_": OpAdapter(
        "torch.index_copy_", _tensor_index_copy_, is_inplace=True
    ),
    "torch.masked_fill": OpAdapter("torch.masked_fill", torch.masked_fill),
    "torch.masked_fill_": OpAdapter(
        "torch.masked_fill_", _tensor_masked_fill_, is_inplace=True
    ),
    "torch.masked_scatter": OpAdapter("torch.masked_scatter", torch.masked_scatter),
    # Comparisons / bitwise
    "torch.__and__": OpAdapter("torch.__and__", _torch___and__),
    "torch.__eq__": OpAdapter("torch.__eq__", _torch___eq__),
    "torch.eq": OpAdapter("torch.le", _torch_eq),
    "torch.le": OpAdapter("torch.le", _torch_le),
    "torch.ne": OpAdapter("torch.ne", _torch_ne),
    "torch.gt": OpAdapter("torch.gt", _torch_gt),
    "torch.logical_and": OpAdapter("torch.logical_and", torch.logical_and),
    # Type/device conversions
    "torch.float": OpAdapter("torch.float", _tensor_float),
    "float": OpAdapter("torch.float", _tensor_float),
    "torch.int": OpAdapter("torch.int", _tensor_int),
    "torch.long": OpAdapter("torch.long", _tensor_long),
    "torch.to": OpAdapter("torch.to", _tensor_to),
    "torch.type_as": OpAdapter("torch.Tensor.type_as", torch.Tensor.type_as),
    # Creation
    "torch.zeros": OpAdapter("torch.zeros", torch.zeros),
    "torch.zeros_like": OpAdapter("torch.zeros_like", torch.zeros_like),
    "torch.ones": OpAdapter("torch.ones", torch.ones),
    "torch.arange": OpAdapter("torch.arange", torch.arange),
    "torch.tensor": OpAdapter("torch.tensor", torch.tensor),
    "torch.scalar_tensor": OpAdapter("torch.scalar_tensor", _scalar_tensor),
    "torch.new_ones": OpAdapter("torch.new_ones", _tensor_new_ones),
    "torch.full": OpAdapter("torch.full", torch.full),
    # Sort / Topk
    "torch.sort": OpAdapter("torch.sort", torch.sort),
    "torch.topk": OpAdapter("torch.topk", torch.topk),
    # NNs / functionals (use F.* where appropriate)
    "torch.nn.functional.dropout": OpAdapter(
        "torch.nn.functional.dropout", torch.nn.functional.dropout, pre=_dropout_pre
    ),
    "torch.nn.functional.embedding": OpAdapter(
        "torch.nn.functional.embedding", torch.nn.functional.embedding
    ),
    "torch.nn.functional.softmax": OpAdapter(
        "torch.nn.functional.softmax", torch.nn.functional.softmax
    ),
    "torch.nn.functional.layer_norm": OpAdapter(
        "torch.nn.functional.layer_norm", torch.nn.functional.layer_norm
    ),
    "torch.nn.functional.batch_norm": OpAdapter(
        "torch.nn.functional.batch_norm", torch.nn.functional.batch_norm
    ),
    "torch.nn.functional.gelu": OpAdapter(
        "torch.nn.functional.gelu", torch.nn.functional.gelu
    ),
    "torch.nn.functional.glu": OpAdapter(
        "torch.nn.functional.glu", torch.nn.functional.glu
    ),
    "torch.nn.functional.silu": OpAdapter(
        "torch.nn.functional.silu", torch.nn.functional.silu
    ),
    "torch.nn.functional.softplus": OpAdapter(
        "torch.nn.functional.softplus", torch.nn.functional.softplus
    ),
    # gelu has both torch.nn.functional.gelu and torch.nn.gelu depending on version
    "torch.nn.gelu": OpAdapter(
        "torch.nn.gelu",
        getattr(torch.nn, "gelu", torch.nn.functional.gelu),
    ),
    # linear: prefer torch.nn.functional.linear; torch.nn.linear may exist in some builds
    "torch.nn.linear": OpAdapter(
        "torch.nn.linear",
        getattr(torch.nn, "linear", torch.nn.functional.linear),
    ),
    "torch.nn.functional.linear": OpAdapter(
        "torch.nn.functional.linear", torch.nn.functional.linear
    ),
    "torch.nn.pad": OpAdapter(
        "torch.nn.pad",
        getattr(torch.nn, "pad", torch.nn.functional.pad),
    ),
    "torch.nn.functional.pad": OpAdapter(
        "torch.nn.functional.pad", torch.nn.functional.pad
    ),
    # Attention / SDPA
    "torch.nn.scaled_dot_product_attention": OpAdapter(
        "torch.nn.scaled_dot_product_attention",
        getattr(
            torch.nn,
            "scaled_dot_product_attention",
            torch.nn.functional.scaled_dot_product_attention,
        ),
    ),
    "torch.nn.functional.scaled_dot_product_attention": OpAdapter(
        "torch.nn.functional.scaled_dot_product_attention",
        torch.nn.functional.scaled_dot_product_attention,
    ),
    "torch.nn.functional.multi_head_attention_forward": OpAdapter(
        "torch.nn.functional.multi_head_attention_forward",
        torch.nn.functional.multi_head_attention_forward,
    ),
    "torch.nn.attention._backend_from_string": OpAdapter(
        "torch.nn.attention._backend_from_string",
        _attn_backend_from_string,
    ),
    "torch.nn.attention._sdpa_kernel": OpAdapter(
        "torch.nn.attention._sdpa_kernel",
        _attn_sdpa_kernel,
    ),
    # Conv
    "torch.conv1d": OpAdapter("torch.conv1d", torch.conv1d),
    "torch.conv2d": OpAdapter("torch.conv2d", torch.conv2d),
    # Autocast internal enter/exit (kept for completeness; not usually unit-tested directly)
    "torch.amp.autocast_mode._enter_autocast": OpAdapter(
        "torch.amp.autocast_mode._enter_autocast",
        lazy_torch("amp.autocast_mode._enter_autocast"),
    ),
    "torch.amp.autocast_mode._exit_autocast": OpAdapter(
        "torch.amp.autocast_mode._exit_autocast",
        lazy_torch("amp.autocast_mode._exit_autocast"),
    ),
    # Functorch internal vmap plumbing (kept for completeness)
    "torch._functorch.vmap.lazy_load_decompositions": OpAdapter(
        "torch._functorch.vmap.lazy_load_decompositions",
        _vmap_lazy_load_decompositions,
    ),
    "torch._C._functorch._add_batch_dim": OpAdapter(
        "torch._C._functorch._add_batch_dim",
        lazy_torch("_C._functorch._add_batch_dim"),
    ),
    "torch._C._functorch._remove_batch_dim": OpAdapter(
        "torch._C._functorch._remove_batch_dim",
        lazy_torch("_C._functorch._remove_batch_dim"),
    ),
    "torch._C._functorch._vmap_increment_nesting": OpAdapter(
        "torch._C._functorch._vmap_increment_nesting",
        lazy_torch("_C._functorch._vmap_increment_nesting"),
    ),
    "torch._C._functorch._vmap_decrement_nesting": OpAdapter(
        "torch._C._functorch._vmap_decrement_nesting",
        lazy_torch("_C._functorch._vmap_decrement_nesting"),
    ),
    # Symbolic sum (present in some builds; fallback provided)
    "torch.sym_sum": OpAdapter("torch.sym_sum", _sym_sum),
    # Misc
    "torch.item": OpAdapter("torch.item", _tensor_item),
    # In-place add_ listed separately
    "torch.add_": OpAdapter("torch.add_", _tensor_add_, is_inplace=True),
    # "torch.add_": OpAdapter("torch.add_", torch.Tensor.add_, is_inplace=True),
}

# Handy set if you want it:
INPLACE_OPS = {name for (name, a) in OP_REGISTRY.items() if a.is_inplace}
