# Inductor Front-End: Deep Dive

This page provides a detailed reference for the Torch-Spyre Inductor
front-end compiler. For a high-level overview of the full compilation
pipeline, see [Compiler Architecture](architecture.md).

:::{figure} ../_static/images/torch-spyre-compilation-spectrum.png
:alt: Torch-Spyre compilation pipeline showing upstream versus custom components
:width: 95%
:align: center

The Torch-Spyre compilation pipeline. The left end (green) is entirely upstream PyTorch — Dynamo/Autograd and Inductor. The right end (pink) is Torch-Spyre's custom Inductor backend, which generates OpSpecs, SuperDSCs, and host code. Torch-Spyre also adds configurations and extensions to the upstream stages to tailor them for the Spyre device.
:::

## Inductor Backend Registration

At import time the Spyre backend registers three components with Inductor. Together they take the place of the Triton/CUDA codegen path on a GPU:

| Component | Module | Role |
|---|---|---|
| `SuperDSCScheduling` | [`scheduler.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/scheduler.py) | Inductor backend scheduling class. Decides how to group and order operations on the LoopLevelIR. Replaces Triton scheduling. |
| `SpyrePythonWrapperCodegen` | [`wrapper.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/wrapper.py) | Inductor wrapper-codegen class. Generates the Python wrapper that allocates tiled buffers via `spyre_empty_with_layout()` and dispatches kernels via `async_compile.sdsc()`. |
| `SpyreDeviceOpOverrides` | [`device/op_overrides.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/device/op_overrides.py) | Device-specific op overrides surfaced to Inductor. |

The Spyre-specific Inductor configuration (decompositions, lowerings, the `mm_to_bmm_pass` that rewrites 2D matmul into 3D bmm for better core utilization, fusion heuristics, and dataflow-friendly Inductor config overrides) is activated through a single context manager:

```python
from torch_spyre._inductor.patches import enable_spyre_context

with enable_spyre_context(...):
    compiled = torch.compile(model, backend="spyre")
```

`enable_spyre_context` is the central entry point that wires everything together. The three registrations above happen earlier, at package import time, in [`_inductor/__init__.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/__init__.py).

## Extending Compilation

The front-end adds compilation passes into upstream Inductor via six extension
points, all registered in
[passes.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/passes.py):

| Extension Point | Stage | Purpose |
|----------------|-------|---------|
| `CustomPreGradPasses` | Pre-grad FX graph | Graph rewrites before autograd partitioning |
| `CustomPrePasses` | Post-grad FX graph (early) | Spyre-specific rewrites early in `post_grad.post_grad_passes` |
| `CustomPostPasses` | Post-grad FX graph (late) | Late post-grad rewrites: `insert_padding`, `replace_scalar_with_tensor`, `mm_to_bmm_pass`, `bmm_unflatten_pass` |
| `CustomPreFusionPasses` | LoopLevelIR (pre-fusion) | Pre-fusion scheduler passes (currently `propagate_mutation_layouts`) |
| `CustomPostFusionPasses` | LoopLevelIR (post-fusion) | Post-fusion scheduler passes (currently `spyre_fuse_nodes`) |
| `CustomPreSchedulingPasses` | LoopLevelIR (pre-scheduler) | Operation-list passes that run immediately before the Scheduler is constructed (wired in via a `GraphLowering._update_scheduler` monkey-patch in [`patches.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/patches.py)). Dispatches `deadcode_elimination`, `propagate_spyre_tensor_layouts`, `insert_restickify`, `core_division_planning`, and `scratchpad_planning` (gated on `config.lx_planning`). |

### FX Graph Passes

Transformations on the FX Graph tend to be simpler to implement, but happen before the
layout of intermediate Tensors in device memory has been computed.  Therefore they need to be layout-agnostic.
Some examples of passes that are appropriate to perform at this level are:
+ replacing constants with size 1 tensors
+ normalizing matrix multiplies to add padding (also applied to `mm` and `bmm`)

### LoopLevelIR Passes

Passes on the LoopLevelIR run relatively late in compilation. `CustomPreSchedulingPasses` dispatches the LoopLevelIR pipeline in a fixed order: dead-code elimination ([deadcode_elimination.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/deadcode_elimination.py)), device-layout propagation through stickification ([stickify.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/stickify.py)), restickify insertion ([insert_restickify.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/insert_restickify.py)), core division ([core_division.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/core_division.py)), and, when `config.lx_planning` is enabled, scratchpad allocation ([scratchpad.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/scratchpad.py)).

Once stickification has run, every `ComputedBuffer` carries a `FixedTiledLayout`, so the later passes can take device layout into account when making decisions.

### Views and Index Translation

Real models lean on views heavily. Here is the RoPE block from Granite:

```python
def rope(cached_freqs, q):
    q_ = q.view(2, 256, 32, 128).view(2, 256, 32, 2, 64)         # B L H 2 D/2
    mul_out = cached_freqs[:, :, None, :, :, :] * q_.unsqueeze(-3)  # B L H 2 2 D/2
    sum_out = mul_out.sum(4, keepdim=True)                        # B L H 2 1 D/2
    return sum_out.flatten(3)                                     # B L H D
```

Two `view` calls, an `unsqueeze`, a reduction, and a `flatten`, all on
the hot path of inference. Materializing a tensor copy at every one of
those view boundaries would erase any benefit from tiling. The rest of
this section walks through how the compiler keeps that from happening.

PyTorch models are full of view operations: `reshape`, `view`,
`transpose`, `permute`, `flatten`, `unsqueeze`, slicing, and so on. A
single transformer block in Granite goes through dozens of them.
On Spyre we want most of these views to cost nothing at runtime;
materializing a copy every time a tensor is reshaped would defeat the
point of tiling.

:::{figure} ../_static/images/spyre-device-views.png
:alt: Worked example of how a shared Inductor index becomes per-tensor device coordinates
:width: 95%
:align: center

A worked end-to-end example. Two tensors with different PyTorch shapes
(`x` is rank-3, `y` is rank-2) both flow through one shared Inductor
index expression. The compiler then lifts that single expression into a
distinct device-coordinate vector for each tensor, and finally
co-simplifies the iteration space so the integer divisions and modulos
collapse into ordinary loop variables. The two tensors end up with
different per-argument dim orders, which SuperDSC handles natively.
:::

Inductor gives us a useful starting point. When it lowers a graph that
involves views, it normalizes everything onto a shared iteration space
and emits a single per-output index expression. For example, this code:

```python
x = torch.rand(50, 10, 200, dtype=torch.float16)
y = torch.rand(500, 200, dtype=torch.float16)

def f(x, y):
    return x.flatten(0, 1) + y

result = torch.compile(f)(x.to("spyre"), y.to("spyre")).cpu()
```

produces an Inductor body that looks roughly like:

```python
var_ranges = {p0: 500, p1: 200}
index0 = 200*p0 + p1

def body(self, ops):
    get_index = self.get_index('index0')
    load   = ops.load('arg0_1', get_index)
    load_1 = ops.load('arg1_1', get_index)
    add    = ops.add(load, load_1)
    store  = ops.store('buf0', get_index, add, None)
    return store
```

Both tensors share `index0` even though `x` was rank-3 and `y` was
rank-2 in the original program: the flatten was absorbed into a single
linear expression `200*p0 + p1`. From here, the Spyre compiler has to
turn that one expression into per-tensor *device* coordinates. There
are three steps.

**1. Lift index expressions to device coordinates.** The host iteration
variables (`p0`, `p1`) describe positions in the PyTorch shape. They
are mapped into per-tensor device coordinate expressions that walk the
tiled, padded device shape. Continuing the example:

| Host vars | x: device shape `[10, 4, 50, 64]` | y: device shape `[4, 500, 64]` |
|---|---|---|
| `{p: 500, s: 200}`, `index0 = 200*p + s` | `[p%10, s//64, p//10, s%64]` | `[s//64, p, s%64]` |

The stick dimension always comes out as a pair of expressions, one for
the tile index (`floor(s/64)`) and one for the intra-stick offset
(`Mod(s, 64)`).

**2. Co-simplify the iteration space and the per-tensor coordinates.**
A naive translation leaves expensive integer divisions in place. The
front-end factors the iteration space so the divisions and modulos
disappear. Splitting `p` into `(q, p)` with `q = p // 10`, `p = p % 10`
gives:

| Iteration space | x | y |
|---|---|---|
| `{p: 10, q: 50, s: 200}` | `[p, s//64, q, s%64]` | `[s//64, q, p, s%64]` |

Each tensor is now indexed with the same iteration variables, but the
expressions inside its coordinate vector are simple and the dim orders
are different per tensor. That is exactly what the SuperDSC IR allows
(`layoutDimOrder_` is per-argument). All of this is held in
[`torch_spyre/_inductor/views.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/views.py)
(`compute_coordinates`, `align_tensors`, `normalize_coordinates`).

**3. Emit the OpSpec.** The simplified iteration space and per-tensor
device coordinates are dropped onto the `OpSpec` and `TensorArg`s. The
"Example: an `add` OpSpec" section in the
[Back-End Compiler](backend.md) doc walks through what one of these
artifacts looks like.

The net result for the user is what you would expect: ops on tensor
views run without cloning whenever the compiler can express the new
layout as a different read pattern over the same storage. When that is
not feasible (for example when a downstream op forces a different stick
dimension), the `insert_restickify` pass adds an explicit re-stick
operation so the rest of the pipeline still sees a clean layout.

### Code Generation

We do code generation in three stages.
1. LoopLevelIR nodes are fused together to form Kernels.
2. Each Kernel is processed by [spyre_kernel.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py)
to convert it to a list of `OpSpec` ([op_spec.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/op_spec.py)).
3. Finally, the [codegen/](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/codegen/)
package translates `OpSpec` into SuperDSC JSON — the input format
for the DeepTools back-end compiler.

Our intent is that the `OpSpec` will capture all important semantic information about the operation in a
more human readable form than the SuperDSC JSON.  Therefore, the `OpSpec` should be the primary artifact
used to understand the output of the front-end compiler.  Inspecting the SuperDSC JSON should only be necessary
when debugging problems in the `codegen` package of the front-end compiler.

## Extending Operations

We extend Inductor to compile Spyre-specific operations by adding Custom Operations.
We modify how existing operations are compiled by adding Spyre-specific decompositions
and lowerings. See [Adding Operations](adding_operations.md) for a step-by-step guide.

### Custom Operations

Spyre-specific operations with no ATen equivalent are defined in
[customops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/customops.py)
using `@torch.library.custom_op`. Each custom op requires:

1. A signature definition (`@custom_op`)
2. A fake/meta function (`@opname.register_fake`)
3. Either a lowering + `SpyreOpFuncs` entry, or a decomposition that
   removes it from the graph before lowering

### Decompositions

Spyre-specific decompositions are registered with `@register_spyre_decomposition`
in
[decompositions.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/decompositions.py).
Decompositions transform complex ATen operations into simpler primitives
before the graph is lowered to loop-level IR.

### Lowerings

Spyre-specific lowerings to Inductor's LoopLevelIR are defined in
[lowering.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/lowering.py)
using the `@register_spyre_lowering` decorator.  This mechanism supports both the replacement
of upstream lowerings and the addition of new lowerings for Spyre-specific custom operations.

## Module Reference

The headline modules above are the ones a contributor reaches for first. The front-end is also made up of a number of smaller modules; the table below names each and points to the source.

| Module | Purpose |
|---|---|
| [`passes.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/passes.py) | The six extension-point classes, plus `_format_operations`, which renders the LoopLevelIR before and after the pre-scheduling pipeline. |
| [`temp_passes.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/temp_passes.py) | Transitional FX-graph rewrites registered in `CustomPostPasses`: `bmm_unflatten_pass`, `mm_to_bmm_pass`, and `replace_scalar_with_tensor`. The "temp" name reflects the plan to retire them as upstream Inductor evolves. |
| [`stickify.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/stickify.py) | `propagate_spyre_tensor_layouts` and `propagate_mutation_layouts`. Assigns `FixedTiledLayout` to every `ComputedBuffer`. |
| [`insert_restickify.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/insert_restickify.py) | `insert_restickify`. Re-sticks tensors whose layout would otherwise be inconsistent. |
| [`deadcode_elimination.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/deadcode_elimination.py) | `deadcode_elimination` for the pre-scheduling LoopLevelIR. |
| [`core_division.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/core_division.py) | `plan_splits`, `apply_splits`, `divide_pointwise_op`, `core_division_planning`. Splits each operation across cores. |
| [`scratchpad.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/scratchpad.py) | `scratchpad_planning`. LX scratchpad allocation, gated on `LX_PLANNING`. |
| [`pass_utils.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py) | Shared helpers for the pre-scheduling pipeline, including `splits_by_index_coeff` and `apply_splits_from_index_coeff`. These translate between iteration-variable splits and the index-coefficient-keyed splits stored on `ComputedBuffer.op_it_space_splits`. |
| [`views.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/views.py) | `compute_coordinates`, `align_tensors`, `normalize_coordinates`, `Term`, `matching_dim`. Coordinate computation for memory-dep expressions and tensor alignment for fused kernels. Used by `spyre_kernel.py` and `core_division.py`. |
| [`multi_dim_reduction_pass.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/multi_dim_reduction_pass.py) | `decompose_multi_dim_reductions`. Splits a multi-dim reduction into a sequence of single-dim reductions. |
| [`op_spec.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/op_spec.py) | `OpSpec`, `TensorArg`, `UnimplementedOp`. The high-level per-operation artifact emitted by `spyre_kernel.py`. |
| [`spyre_kernel.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py) | Converts a fused kernel into a list of `OpSpec`s. |
| [`padding.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/padding.py) | `insert_padding`. Pads `mm`, `bmm`, and tensor methods to satisfy hardware alignment. |
| [`fusion.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/fusion.py) | `spyre_fuse_nodes`. Post-fusion scheduler pass. |
| [`wrapper.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/wrapper.py) | `SpyrePythonWrapperCodegen`. The host-code generator that produces the Python wrapper around device kernels. |
| [`codegen/`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/codegen/) | `superdsc.py` (`SDSCArgs`, `SDSCSpec`, `parse_op_spec`, `compile_op_spec`, `_get_core_to_slice_mapping`, `_get_padded_iteration_space`, `_get_op_dim_labels`), `compute_ops.py` (`generate_sdsc`), `bundle.py` (`generate_bundle`). Translates `OpSpec` into SuperDSC JSON for the back-end compiler. |
| [`config.py`](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/config.py) | Spyre-specific Inductor configuration: `SENCORES`, `LX_PLANNING`, `DXP_LX_FRAC_AVAIL`. |

`torch.compile(..., dynamic=True)` is supported through the static-binary path. Shapes are specialized at compile time and the resulting binary is reused across calls with the same input geometry.
