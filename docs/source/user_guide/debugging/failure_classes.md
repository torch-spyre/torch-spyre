# Common Failure Classes

Most wrong results on Spyre fall into a small number of classes. Each has a
symptom that identifies it and a tool that confirms it. This page is a
reference you can scan once a [differential test](differential_testing.md)
shows that a run diverges from CPU.

The classes exist because the device differs from CPU in three ways that are
not visible at the PyTorch API. Tensors are tiled into fixed-size sticks with
padding. The device float format does not match IEEE `float16` in the
extremes. And the compiler decides every byte of movement and every core
assignment ahead of time.

## Symptom to class

| Symptom | Class | Confirm with |
|---|---|---|
| Zeros or stale values past the first stick | [Layout and stick alignment](#layout-and-stick-alignment) | `copy_range` and layout in the `sdsc` artifact |
| `NaN` where CPU gives a finite number | [Special values, inf and NaN](#special-values-inf-and-nan) | print the mask or identity constant |
| A warning, a slower run, and a changed dtype | [Silent CPU fallback](#silent-cpu-fallback) | promote `FallbackWarning` to an error |
| An "unsupported dtype" error, or an int64 downcast warning | [dtype and downcast](#dtype-and-downcast) | dtypes in `fx_graph_readable.py` |
| Correct with planning off, wrong with it on | [Scratchpad residency and planning](#scratchpad-residency-and-planning) | rerun with `LX_PLANNING=0` |
| Result changes between identical runs | [Nondeterminism](#nondeterminism) | disable caches, pin `SENCORES` |

The rest of this page takes each class in turn.

## Layout and stick alignment

A tensor's device layout pads the stick dimension to 128 bytes, which is 64
elements at `float16`. A `float16` tensor of 65 elements is laid out as two
sticks:

:::{figure} ../../_static/images/stick-layout.svg
:alt: A float16 tensor of 65 elements laid out as two 64-element device sticks. The first stick holds elements 0 to 63. The second holds element 64 and 63 padding lanes.
:width: 100%

A `float16` tensor of 65 elements occupies two sticks. The second stick holds
element 64 and 63 padding lanes. `is_contiguous()` returns true, so a
stride-based check does not see the padding tail.
:::

Operations on shapes that are not stick-aligned, slices that start partway
through a stick, and transposed weights all depend on the layout being
materialized correctly. The failure is a value that is wrong, or stale, past
the first stick, while the first stick is correct.

A `float16` tensor also reports `is_contiguous() == True` here even though it
spans two sticks with a nonzero offset into the second, so a stride-based
check does not detect this and `.contiguous()` does not fix it. A copy that
must be correct needs a real materialization, such as `.clone()`.

To confirm, read the `copy_range` and `device_size` in the kernel's `sdsc`
artifact. See [Inductor Artifacts](inductor_artifacts.md) and the worked clone
example in the [debugging overview](index.md). For the layout model itself,
see [Tensor Layouts](../tensors_and_layouts.md).

## Special values, inf and NaN

The device float format does not represent IEEE infinities the way CPU
`float16` does. A mask built from `-inf` encodes to a `NaN` bit pattern,
because the device format has no infinity. That `NaN` then propagates through
the entire reduction that reads it, not only the masked positions. The symptom
is a `NaN` where CPU produces a finite number, often appearing only as the
masked region grows.

:::{figure} ../../_static/images/mask-nan.svg
:alt: With a -inf mask, the masked position encodes to NaN and the reduction output is NaN at every position. With a -1e4 mask, the output stays finite.
:width: 100%

A single masked position carries the whole row to `NaN`. A finite mask such as
`-1e4` keeps the reduction correct.
:::

Use finite constants that the device encodes correctly:

```python
mask_value      = -1e4                              # for an exp or softmax mask
reduce_identity = torch.finfo(torch.float16).min    # for a max-reduction identity
```

To confirm, print the mask for a small case and check that masked positions
hold the finite value you intended and unmasked positions hold zero.

## Silent CPU fallback

An operation the backend cannot lower runs on the host instead and emits a
`FallbackWarning`. A fallback changes the numerical path, because CPU and
device accumulate `float16` differently, and it inserts a device-to-host and
host-to-device copy that can change dtype or layout. The operation that looks
wrong may not be the operation that fell back.

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart LR
    A[op on device] --> B[device to host copy]
    B --> C["CPU kernel<br/>(different accumulation)"]
    C --> D[host to device copy]
    D --> E[result back on device]
```

The two copies are where dtype or layout can change, and in a steady-state
loop they repeat on every call.

In a long run the warning scrolls past, so promote it to an error to stop at
the exact call:

```bash
python -W "error::torch_spyre.ops.fallbacks.FallbackWarning" my_reproducer.py
```

Filter on the class, not on `UserWarning`, which would also catch unrelated
warnings. The on-device operations are registered in `torch_spyre/ops/eager.py`
and the host fallbacks in `torch_spyre/ops/fallbacks.py`, per overload, so
`aten.mm` and `aten.mm.out` can differ. A fallback inside a steady-state loop
is also a performance problem, since it performs a host round trip on every
call.

## dtype and downcast

The validated device path is `float16`. A `float32` intermediate that reaches
the device raises an "unsupported dtype" error, which the FX graph shows as
the dtype on the offending node. An `int64` index is downcast to `int32`,
which emits the warning "Backend Spyre does not support int64; downcasting to
int32 may change values outside the 32-bit range." Silence it with the
environment variable, or programmatically with
`spyre.set_downcast_warning(False)`, once you have confirmed the index range
fits in `int32`:

```bash
TORCH_SPYRE_DOWNCAST_WARN=0 python my_reproducer.py   # suppress once understood
```

Read the dtypes in `fx_graph_readable.py` to find where a `float32` value
entered the graph.

## Scratchpad residency and planning

The compiler plans which tensors stay in the on-chip LX scratchpad and which
intermediates share a pooled region of device memory. A residency or lifetime
error produces a wrong result that depends on those passes being enabled. The
symptom is a result that is correct with planning disabled and wrong with it
enabled.

Isolate it by turning the passes off:

```bash
LX_PLANNING=0 HBM_POOL_PLANNING=0 python my_reproducer.py
```

If the result becomes correct, the fault is in residency or allocation rather
than in the operation. For how the planner works, see
[Scratchpad Planning](../../compiler/scratchpad_planning.md).

## Nondeterminism

A result that changes between runs that should be identical, or between a test
run in isolation and the same test run inside a larger suite, must be brought
under control before any magnitude is trusted. The usual sources are a warm
compile cache, the number of cores, and hash-seed-dependent iteration order.

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
SENCORES=1 \
PYTHONHASHSEED=0 \
python my_reproducer.py
```

A bug that disappears at `SENCORES=1` is in the multi-core work division. A
result that changes with `PYTHONHASHSEED` means a pass depends on set
iteration order. Never compare magnitudes across different run scopes without
fixing these first.

## Where this is in the source

Each class maps to code or a worked case you can read.

- Layout and stick alignment: the layout model in
  [Tensor Layouts](../tensors_and_layouts.md); the clone case in
  [issue #524](https://github.com/torch-spyre/torch-spyre/issues/524); the
  mid-stick `torch.cat` case (PR #3875) and transposed-weight K padding
  (PR #4225).
- Special values: the finite mask and reduction constants, `-1e4` for an exp
  mask and `-65504`/`65504` for a reduction identity (PR #4234).
- Silent CPU fallback: the `FallbackWarning` class in
  `torch_spyre/ops/fallbacks.py`, and the on-device registry in
  `torch_spyre/ops/eager.py`.
- Scratchpad residency: the `LX_PLANNING` and `HBM_POOL_PLANNING` flags in
  `torch_spyre/_inductor/config.py`; the slice-derived buffer case (PR #2683).
- Nondeterminism: the fix that makes LX planning independent of
  `PYTHONHASHSEED` (PR #4196).

## Next steps

- [Differential Testing Against CPU](differential_testing.md) to produce the
  divergence that points at one of these classes.
- [Inductor Artifacts](inductor_artifacts.md) to read the layout, work
  division, and allocations a class points to.
- [The OpSpec Lab](op_spec_lab.md) to replay a single kernel and split the
  front-end from the backend.
