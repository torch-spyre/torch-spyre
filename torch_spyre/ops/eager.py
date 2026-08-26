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

import torch
from torch_spyre._C import fill_tensor, copy_tensor, SpyreTensorLayout
import torch_spyre.ops.fallbacks  # noqa: F401
from .fallbacks import _get_op_overloads
import warnings
import functools
import inspect
import operator


aten = torch.ops.aten


# Decorator to keep track of compiled variant
def compile_once(op, **compile_kwargs):
    def decorator(fn):
        compiled = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal compiled
            nonlocal op
            if compiled is None:
                if isinstance(op, str):
                    op = operator.attrgetter(op)(torch.ops)
                compiled = torch.compile(op, **compile_kwargs)
            return fn(*args, compiled=compiled, **kwargs)

        # We remove the `compiled` arg from the signature to have
        # a clean signature.
        old_signature = inspect.signature(fn)
        params = dict(old_signature.parameters)
        params.pop("compiled", None)
        new_signature = old_signature.replace(parameters=params.values())
        wrapper.__signature__ = new_signature

        return wrapper

    return decorator


def maybe_wrap_dim(dim: int, ndims: int) -> int:
    if dim < 0:
        return dim + ndims
    return dim


def _materialize_offset_view(x):
    """Return an offset-0 copy of a Spyre tensor that carries a nonzero
    ``storage_offset``; pass everything else through unchanged.

    A standalone-compiled kernel drops its input's ``storage_offset`` (upstream
    Inductor's placeholder path reads sizes/strides only, and SpyreTensorLayout
    has no offset field), so it binds the storage BASE pointer and reads from
    element 0 regardless of the view's true offset — see ``_reoffset`` in
    ``torch_spyre/_inductor/lowering.py``. Only ``spyre::copy_from_d2d``
    re-injects the offset in-graph; every other compiled eager op would
    silently read the wrong data.

    ``clone()`` dispatches ``aten::clone`` -> ``aten::copy_`` ->
    ``spyre::copy_from_d2d``, i.e. that same offset-honoring path, producing a
    correct offset-0 buffer. This is a no-op for the overwhelmingly common
    offset-0 case (fresh buffers, and slices that already forced a copy).

    ``List[Tensor]`` args (e.g. ``aten.cat``/``aten.stack``) are recursed into
    element-wise so an offset view nested in a list is materialized too;
    otherwise those ops would silently read element 0 of each nested view.
    """
    if isinstance(x, torch.Tensor):
        if x.device.type == "spyre" and x.storage_offset() != 0:
            return x.clone()
        return x
    if isinstance(x, (list, tuple)):
        return type(x)(_materialize_offset_view(e) for e in x)
    return x


class RetileWarning(UserWarning):
    """Warning issued when an eager result had to be re-tiled to the layout a
    compiled graph assumes for it."""


warnings.simplefilter("once", RetileWarning)


def _normalize_result_layout(x):
    """Return a copy of a Spyre tensor whose device layout is the *canonical* one
    for its logical shape.

    ``propagate_layouts`` stamps a fallback's output with ``generic_layout(op)``,
    i.e. the size-only ``SpyreTensorLayout(size, dtype)``, and inserts no
    restickify to make that assumption true — so an eager kernel returning a
    differently-tiled buffer is read by the wrong tiling, silently. Rebuilding
    the result here makes the assumption hold.

    Only whole buffers are considered. ``device_tensor_layout()`` describes the
    tensor's BASE allocation, not the view, so for any view it reports a layout
    for a different logical shape and would compare unequal no matter how the
    bytes are tiled — rebuilding on that basis corrupts a buffer that was
    already self-consistent (see ``TestPermutedEagerResultNotNormalized``).
    ``_base is None`` restricts us to freshly allocated kernel results, which is
    exactly the case the assumed layout is stamped on. Note also that
    ``dim_order`` is not reachable from Python (the binding exposes only
    ``device_size``/``stride_map``/``device_dtype``/``element_arrangement``), so
    comparing whole layouts is the only way to detect the mismatch.
    """

    if not isinstance(x, torch.Tensor) or x.device.type != "spyre":
        return x
    if x._base is not None or not x.is_contiguous():
        return x
    real = x.device_tensor_layout()
    if real is None:
        # No layout to compare (e.g. a FakeTensor under tracing): leave it alone
        # rather than force a copy on a tensor we cannot reason about.
        return x
    if real == SpyreTensorLayout([int(s) for s in x.shape], x.dtype):
        return x

    warnings.warn(
        f"re-tiling a {tuple(x.shape)} {x.dtype} eager result whose device "
        f"layout is not the one a compiled graph assumes for its shape",
        category=RetileWarning,
        stacklevel=2,
    )
    out = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    # The host round-trip reads the source by its own real layout and writes the
    # destination by the canonical one, which is the re-tiling we want.
    #
    # A device-to-device copy_ would re-tile without leaving the device, but it
    # routes through spyre::copy_from_d2d, i.e. a nested torch.compile from
    # inside the eager kernel we are already compiling. That nesting raises
    # InductorError from optimize_restickify.beam_global_min_cost and breaks
    # test_reduction_reads_correct_slice[2|32] (measured). Revisit once a
    # cross-layout D2D copy is available without re-entering the compiler — see
    # the same TODO at csrc/spyre_mem.cpp:759.
    out.copy_(x.to("cpu"))
    return out


def _write_arg_slots(op):
    """Positions and names of an op's mutated (write-aliased) arguments.

    Such arguments (e.g. ``out=`` of an out-variant, ``self`` of an in-place op)
    need read-modify-write handling rather than a plain read-side clone: the
    standalone-compiled kernel would write to the storage base (element 0)
    instead of the view's ``storage_offset``. ``alias_info.is_write`` identifies
    them from the schema.
    """
    positions: set[int] = set()
    names: set[str] = set()
    for i, arg in enumerate(op._schema.arguments):
        if arg.alias_info is not None and arg.alias_info.is_write:
            positions.add(i)
            names.add(arg.name)
    return positions, names


def _map_result(result, fn):
    """Apply ``fn`` to every tensor in an op's return value, rebuilding the
    containers around them.

    The two tuple subclasses a multi-output aten schema can return are rebuilt as
    themselves, and they need different calls: a namedtuple's ``__new__`` takes
    the fields positionally so it must go through ``_make``, while a structseq
    (``torch.return_types.*``) has no ``_make`` and its ``__new__`` takes the
    iterable directly. Anything else becomes a plain ``tuple``, since an
    arbitrary subclass's ``__init__`` need not accept either form.
    """
    if isinstance(result, torch.Tensor):
        return fn(result)
    if isinstance(result, tuple):
        mapped = [_map_result(r, fn) for r in result]
        cls = type(result)
        if hasattr(cls, "_make"):  # namedtuple
            return cls._make(mapped)
        if hasattr(cls, "n_fields"):  # structseq, e.g. torch.return_types.max
            return cls(mapped)
        return tuple(mapped)
    if isinstance(result, list):
        return [_map_result(r, fn) for r in result]
    return result


def _remap_result(result, lookup):
    """Swap substituted clones back to the caller's originals in a return value.

    ``lookup`` maps ``id(clone) -> original``. The compiled kernel echoes the
    write-arg object it was handed (verified: ``relu_``/``add.out`` return the
    same object), so an in-place/out op returns the clone we substituted;
    restore the caller's tensor identity so aliasing is preserved.
    """
    return _map_result(result, lambda t: lookup.get(id(t), t))


def _check_same_device(operands):
    """Raise if tensor ``(value, is_write)`` operands span more than one device,
    exempting a single non-write 0-dim CPU tensor (mirrors TensorIterator's ``allow_cpu_scalars_``
    unlike ``fallbacks._ensure_device``, which moves tensors instead of rejecting).

    Returns ``(mandatory_device, exempted_tensor)``: the device every
    non-exempt operand agreed on (``None`` if there were none), and the
    exempted CPU scalar tensor if one was found (``None`` otherwise) -- the
    caller must still materialize it onto ``mandatory_device`` before handing
    it to a standalone-compiled kernel, which cannot accept a raw CPU input.
    """
    mandatory_device = None
    exempted = None

    def visit(x, is_write):
        nonlocal mandatory_device, exempted
        if isinstance(x, torch.Tensor):
            if (
                not is_write
                and exempted is None
                and x.device.type == "cpu"
                and x.dim() == 0
            ):
                exempted = x
                return
            if mandatory_device is None:
                mandatory_device = x.device
            elif x.device != mandatory_device:
                raise RuntimeError(
                    "Expected all tensors to be on the same device, but "
                    f"found at least two devices, {mandatory_device} and "
                    f"{x.device}!"
                )
        elif isinstance(x, (list, tuple)):
            for e in x:
                visit(e, is_write)

    for value, is_write in operands:
        visit(value, is_write)

    return mandatory_device, exempted


def _replace_tensor(x, old, new):
    """Substitute ``old`` for ``new`` by identity, recursing into list/tuple."""
    if x is old:
        return new
    if isinstance(x, (list, tuple)):
        return type(x)(_replace_tensor(e, old, new) for e in x)
    return x


def _make_offset_safe_dispatch(op):
    """Build a dispatch wrapper that keeps nonzero-offset Spyre tensors correct
    across the standalone-compiled kernel.

    - Read (non-write) args: materialize a nonzero-offset view to a fresh
      offset-0 buffer (``_materialize_offset_view``) before the call.
    - Write (mutated) args: read-modify-write. Clone the offset view to an
      offset-0 buffer, run the kernel against the clone, then ``copy_`` the
      result back into the caller's view.
    - Results: rebuild any output whose device tiling is not the canonical one
      for its shape (``_normalize_result_layout``), since a compiled graph
      consuming this op as a fallback will assume the canonical tiling.

    Everything is a no-op for the common offset-0, canonical-layout case.
    """
    write_positions, write_names = _write_arg_slots(op)
    # In-place/out variants return the caller's own buffer; rebuilding it would
    # break the aliasing the schema promises, so leave their results alone.
    normalize_results = not (write_positions or write_names)

    def dispatch(*args, compiled=None, **kwargs):
        device, exempted = _check_same_device(
            [(a, i in write_positions) for i, a in enumerate(args)]
            + [(v, k in write_names) for k, v in kwargs.items()]
        )
        if exempted is not None:
            # The compiled kernel only accepts spyre-device inputs -- move the
            # exempted CPU scalar over rather than leaving it for Inductor,
            # which has no notion of a live CPU graph input.
            moved = exempted.to(device)
            args = tuple(_replace_tensor(a, exempted, moved) for a in args)
            kwargs = {k: _replace_tensor(v, exempted, moved) for k, v in kwargs.items()}
        write_back = []  # (clone, original) for each substituted write view

        def prep_write(x):
            if isinstance(x, torch.Tensor):
                if x.device.type == "spyre" and x.storage_offset() != 0:
                    local = x.clone()
                    write_back.append((local, x))
                    return local
                return x
            if isinstance(x, (list, tuple)):
                return type(x)(prep_write(e) for e in x)
            return x

        args = tuple(
            prep_write(a) if i in write_positions else _materialize_offset_view(a)
            for i, a in enumerate(args)
        )
        kwargs = {
            k: (prep_write(v) if k in write_names else _materialize_offset_view(v))
            for k, v in kwargs.items()
        }

        result = compiled(*args, **kwargs)

        if normalize_results:
            result = _map_result(result, _normalize_result_layout)

        if write_back:
            for local, original in write_back:
                original.copy_(local)
            result = _remap_result(
                result, {id(local): original for local, original in write_back}
            )
        return result

    return dispatch


def _compile_kernel_overloads(ops):
    """Overloads of ``ops`` that get a standalone-compiled Spyre kernel."""
    for op in _get_op_overloads(ops):
        if "Tensor" not in str(op._schema):
            # there are some ops that do not take in Tensors
            # like aten.sum.int
            continue
        if "dtype" in op.name():
            # ops that change dtype are not supported yet
            continue
        yield op


def register_torch_compile_kernel(ops):
    for op in _compile_kernel_overloads(ops):
        dispatch = _make_offset_safe_dispatch(op)
        compiled_kernel = compile_once(op, dynamic=False)(dispatch)
        torch.library.register_kernel(op.name(), ["spyre"])(compiled_kernel)


# Single source of truth: every op that gets a standalone-compiled Spyre kernel.
# ``register_inplace_kernels`` derives the in-place registrations from this same
# list, so the two cannot drift apart (see its docstring).
COMPILED_OPS = [
    aten.mm,
    aten.silu.out,
    aten.mish.out,
    aten.abs,
    aten.add,
    aten.bitwise_not,
    aten.logical_not,
    aten.bmm,
    aten.cat,
    aten.div,
    aten.exp,
    aten.floor,
    aten.index_select,
    aten.log,
    aten.mean,
    aten.mul,
    aten.reciprocal,
    aten.neg,
    aten.relu,
    aten.rsqrt,
    aten.sigmoid,
    aten._softmax,
    aten.stack,
    aten.sum,
    aten.sqrt,
    aten.tanh,
    aten.sub,
    aten.addmm,
    aten.eq,
    aten.le,
    aten.ne.Tensor,
    aten.ne.Tensor_out,
    aten.ge,
    aten.gt,
    aten.lt,
    aten.amax,
    aten.maximum,
    aten.minimum,
    aten.pow,
    aten.linalg_vector_norm,
    aten.where.self,
    aten.where.self_out,
    aten.clamp,
    aten.constant_pad_nd,
    aten.embedding.default,
]

register_torch_compile_kernel(COMPILED_OPS)


def _arg_signature(schema):
    """``(name, type, kwarg_only)`` per argument, alias annotations stripped.

    A safe functional/in-place pair differs *only* in the ``(a!)`` write-alias
    on ``self`` and the return alias, so these tuples must be equal.
    """
    return [(a.name, str(a.type), a.kwarg_only) for a in schema.arguments]


def _functional_sibling(inplace_op):
    """The functional overload matching ``inplace_op``, or ``None``.

    Requires the same overload name *and* a matching argument signature. The
    name alone is not enough: ``pow_.Scalar(Tensor self, Scalar exponent)`` and
    ``pow.Scalar(Scalar self, Tensor exponent)`` share an overload name with
    *swapped* operands, so a name-only pairing would build a kernel computing
    ``other ** self``. The signature check rejects that pair.

    Matching on signature alone would instead reach ``pow.Tensor_Scalar``, which
    is operand-correct, but the device's functional ``pow`` is itself wrong
    today, so the conservative name requirement stays.

    ``None`` means the pair is not a safe functional/in-place match and the
    caller must skip it.
    """
    if len(inplace_op._schema.returns) != 1:
        return None
    packet_name, _, overload = inplace_op.name().partition(".")
    functional_name = packet_name.split("::")[1][:-1]  # 'aten::mul_' -> 'mul'
    functional_packet = getattr(aten, functional_name, None)
    if functional_packet is None:
        return None
    overload = overload or "default"
    if overload not in functional_packet.overloads():
        return None
    functional_op = getattr(functional_packet, overload)
    if _arg_signature(functional_op._schema) != _arg_signature(inplace_op._schema):
        return None
    return functional_op


def _make_inplace_kernel(functional_op):
    """Build an in-place kernel as functional-compute + ``copy_`` back.

    The mutation must go through ``self.copy_`` (a runtime-addressed
    ``spyre::copy_from_d2d``) rather than a compiled in-place kernel, which
    bakes its write-destination address at trace time and can therefore write
    to a stale address, clobbering an unrelated live buffer.
    """

    def kernel(self, *args, **kwargs):
        # functional_op's schema has no write arg marking self as output, so check device write-status here first.
        _check_same_device(
            [(self, True)]
            + [(a, False) for a in args]
            + [(v, False) for v in kwargs.values()]
        )
        result = functional_op(self, *args, **kwargs)
        # PyTorch's in-place contract: the promoted result dtype must be
        # castable back to ``self``. ``copy_`` would happily downcast (int32
        # ``self`` silently truncating a float32 result), so check first and
        # raise the same error eager CPU/CUDA does. Shape is left to ``copy_``,
        # whose broadcast check already rejects a result wider than ``self``.
        if not torch.can_cast(result.dtype, self.dtype):
            raise RuntimeError(
                f"result type {result.dtype} can't be cast to the desired "
                f"output type {self.dtype}"
            )
        self.copy_(result)
        # Return ``self``, not the functional result: an in-place schema
        # declares ``Tensor(a!)``, so callers rely on getting back the very
        # tensor they passed in. Discarding the functional result here is
        # deliberate -- its values already landed in ``self`` via ``copy_``.
        return self

    return kernel


def register_inplace_kernels(ops):
    """Register the in-place sibling of every compiled op in ``ops``.

    Derived from ``COMPILED_OPS`` rather than a second hand-maintained list:
    any op whose functional form gets a compiled kernel also needs its
    ``foo_`` variant routed through ``copy_``, and deriving both from one list
    keeps them from drifting (``relu_`` was previously registered as a
    compiled in-place kernel, exactly the pattern this avoids).

    Pairs are accepted only when the in-place and functional signatures match
    modulo the write-alias, and only when the resolved functional overload
    actually got a compiled kernel above -- see :func:`_functional_sibling`.
    """
    compiled = {op.name() for op in _compile_kernel_overloads(ops)}
    base_names = {name.partition(".")[0].split("::")[1] for name in compiled}

    for base_name in sorted(base_names):
        inplace_packet = getattr(aten, base_name + "_", None)
        if inplace_packet is None:
            # e.g. no ``mm_`` for ``mm``
            continue
        for overload in inplace_packet.overloads():
            try:
                inplace_op = getattr(inplace_packet, overload)
            except RuntimeError:
                # schema-only entries such as ``add_.t`` have no dispatcher op
                continue
            sibling = _functional_sibling(inplace_op)
            if sibling is None or sibling.name() not in compiled:
                continue
            torch.library.register_kernel(inplace_op.name(), ["spyre"])(
                _make_inplace_kernel(sibling)
            )


register_inplace_kernels(COMPILED_OPS)


@torch.library.register_kernel("aten::fill_.Scalar", ["spyre"])  # type:ignore
def spyre__fill_scalar(
    self: torch.Tensor, other: int | float | bool | complex
) -> torch.Tensor:
    if isinstance(other, complex):
        raise TypeError("spyre fill_ does not support complex fill values")
    fill_tensor(self, float(other))
    return self


@torch.library.register_kernel("aten::full", ["spyre"])  # type:ignore
def spyre_full(
    size: list | tuple,
    fill_value: int | float | bool | complex,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    if isinstance(fill_value, complex):
        raise TypeError("spyre full does not support complex fill values")
    t = torch.empty(size, dtype=dtype, device=device)
    fill_tensor(t, float(fill_value))
    return t


@torch.library.register_kernel("aten::ones", ["spyre"])  # type:ignore
def spyre_ones(
    size: list | tuple,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    t = torch.empty(size, dtype=dtype, device=device)
    fill_tensor(t, 1.0)
    return t


@torch.library.register_kernel("aten::normal_", ["spyre"])  # type:ignore
def spyre__normal_(self, mean=0.0, std=1.0, *, generator=None):
    # "normal_" generates a random tensor, thus copying
    # "self" back from SPYRE to CPU is not needed.
    # cpu_tmp = self.to("cpu")

    # Create a new tensor on cpu itself to avoid unnecessary data copy.
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)
    cpu_tmp.normal_(mean, std, generator=generator)
    self.copy_(cpu_tmp)
    return self


@torch.library.register_kernel("aten::zero_", ["spyre"])  # type:ignore
def spyre__zero_(self: torch.Tensor) -> torch.Tensor:
    """Zero out the tensor in-place using device-side FillDMA."""
    fill_tensor(self, 0.0)
    return self


@torch.library.register_kernel("aten::uniform_", "spyre")  # type:ignore
def spyre__uniform_(self, from_=0.0, to=1.0, generator=None):
    # Create a new tensor on cpu
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)

    # Fill the CPU tensor with uniform random values
    cpu_tmp.uniform_(from_, to, generator=generator)

    # Copy the CPU tensor back to the spyre device
    self.copy_(cpu_tmp)

    return self


@torch.library.register_kernel("aten::random_.from", ["spyre"])  # type:ignore
def spyre__random_from(self, from_=0, to=1, generator=None) -> torch.Tensor:
    # Create a new tensor on CPU.
    cpu_tmp = torch.empty_like(self, device="cpu", memory_format=torch.preserve_format)

    # Fill the CPU tensor with random values and copy to device.
    cpu_tmp.random_(from_, to, generator=generator)
    self.copy_(cpu_tmp)

    return self


@torch.library.register_kernel("aten::_local_scalar_dense", "spyre")
def spyre__local_scalar_dense(self):
    return self.cpu().item()


@torch.library.register_kernel("aten::_copy_from", ["spyre"])
def spyre__copy_from(self, dst, non_blocking=False):
    if self.numel() == 0:
        return dst

    # Check if views of same data
    if (
        self.data_ptr() == dst.data_ptr()
        and self.storage_offset() == dst.storage_offset()
        and self.stride() == dst.stride()
        and self.size() == dst.size()
        and self.dtype == dst.dtype
        and self.is_conj() == dst.is_conj()
        and self.is_neg() == dst.is_neg()
    ):
        return dst

    if (self.device.type == "cpu" and dst.device.type == "spyre") or (
        self.device.type == "spyre" and dst.device.type == "cpu"
    ):
        copy_tensor(self, dst, non_blocking)
        return dst
    elif self.device.type == "spyre" and self.device == dst.device:
        # copy_from_d2d requires torch.compile, which cannot run inside
        # no_dispatch() (e.g. during FakeTensorMode constant propagation).
        # Fall back to a CPU roundtrip copy in that case.
        #
        # Detecting "am I inside no_dispatch()" uses the private
        # ``torch._C._dispatch_tls_is_dispatch_key_excluded("Python")`` (no
        # public predicate exists — no_dispatch() excludes the Python dispatch
        # key). Revisit if upstream exposes a stable API; the alternative
        # (attempt copy_from_d2d and catch the re-entrancy failure) is worse.
        if torch._C._dispatch_tls_is_dispatch_key_excluded("Python"):
            cpu_tmp = self.to("cpu")
            copy_tensor(cpu_tmp, dst, non_blocking)
        else:
            # Pass storage_offsets explicitly: a graph input's storage_offset
            # is dropped by Inductor, so the lowering must re-introduce it
            # in-graph (see copy_from_d2d in customops.py and
            # lower_spyre_from_d2d).
            torch.ops.spyre.copy_from_d2d(
                self, dst, self.storage_offset(), dst.storage_offset()
            )
        return dst
    else:
        if non_blocking:
            warnings.warn(
                f"non_blocking is set to {non_blocking}", UserWarning, stacklevel=2
            )

        torch.ops.aten._copy_from.default(self, dst, non_blocking)
        return dst


# INSERT_CODEGEN_HERE
