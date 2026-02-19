## Spyre Inductor Operation Cookbook

This document describe the common patterns used to define operations
in the front-end compiler.

### Direct mapping from ATen to OpFunc

If a pointwise ATen operation can be implemented with a single Spyre OpFunc,
then enabling it in our backend only requires
adding a method to `SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py).
Canonical examples are `add` and `softplus` (see `softplus`for an example of using `op_info` for non-tensor arguments).

Note that some pointwise ATen operations that can be be implemented with a single Spyre OpFunc
have default decompositions defined by Inductor. Adding a method to
`SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py)
overrides the default decomposition and thus enables the desired direct mapping.
Canonical examples are `reciprocal` and `sigmoid`.

### Spyre-specific decompositions

We define Spyre-specific decompositions in [decompositions.py](../torch_spyre/_inductor/decompositions.py)
using the `@register_decomposition` decorator.  Decompositions are graph transformations
that are performed before the graph is lowered to loop level IR.

### Spyre-specific lowerings

We define Spyre-specific lowerings from ATen operations to Inductor's
loop level IR in [lowering.py](../torch_spyre/_inductor/lowering.py) using the `@register_spyre_lowering` decorator.

### Spyre-specific OpFuncs

For Spyre OpFuncs that do not have corresponding ATen operations, we use
the `@torch.library.custom_op` decorator to define a new operation in
[customops.py](../torch_spyre/_inductor/customops.py). This has two pieces:
+ defining the signature of the operation (using `@custom_op`)
+ defining its fake function (using the `@opname.register_fake` that is defined as part of the `@custom_op`)

In addition, when defining a custom op, you will also need to do one of:
+ register a lowering for the custom op in [lowering.py](../torch_spyre/_inductor/lowering.py) and
  add a method to `SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py).
  A canonical example is `spyre.clamp`.
+ register a decomposition for the custom op in [decompositions.py](../torch_spyre/_inductor/decompositions.py)
  that removes the custom op from the graph before lowering. A canonical example is `spyre.compact`.
+ define a `CustomPrePass` or `CustomPostPass` that implements a more general graph
  rewrite that removes the custom op from the graph before lowering. We currently do not have any custom ops that use this option.
