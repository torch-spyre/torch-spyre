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

## Extending Compilation

The front-end adds compilation passes into upstream Inductor via four extension points,
all registered in
[passes.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/passes.py):

| Extension Point | Stage | Purpose |
|----------------|-------|---------|
| `CustomPrePasses` | FX Graph (pre-lowering) | Graph rewrites before decomposition |
| `CustomPostPasses` | FX Graph (late post-grad) | Graph rewrites late in the post-grad FX graph passes |
| `_pre_fusion_custom_pass` | Scheduler | Passes on LoopLevelIR immediately before nodes are fused into kernels |
| `_post_fusion_custom_pass` | Scheduler | Passes on LoopLevelIR immediately after nodes are fused into kernels |

### FX Graph Passes

Transformations on the FX Graph tend to be simpler to implement, but happen before the
layout of intermediate Tensors in device memory has been computed.  Therefore they need to be layout-agnostic.
Some examples of passes that are appropriate to perform at this level are:
+ replacing constants with size 1 tensors
+ normalizing matrix multiplies to add padding

### LoopLevelIR Passes

Passes on the LoopLevelIR happen relatively late in compilation. One of the first
things we do on the LoopLevelIR is to determine the device layout (`FixedTiledLayout`)
of all `ComputedBuffers` via the stickification pass ([stickify.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/stickify.py)).
Subsequent passes like core division ([core_division.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/core_division.py))
and scratchpad allocation ([scratchpad.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/scratchpad.py))
are therefore able to consider device layout information when making decisions.

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
