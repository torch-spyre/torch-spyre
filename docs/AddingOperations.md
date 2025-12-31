## Spyre Inductor Operation Cookbook

This document describe the common patterns used to define operations
in the Inductor Spyre backend.

### Direct mapping from ATen to OpFunc

If a pointwise ATen operation can be implemented with a single Spyre OpFunc,
then adding it to our backend just requires
adding a method to `SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py).
Cannonical examples are `add` for a pointwise operation.

Some pointwise ATen operations that could be directly mapped to a Spyre OpFunc
have default decompositions defined by Inductor. We disable the default
decomposition by adding a method to `SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py).
Cannonical examples are `reciprocal` and `sigmoid`.

### Spyre-specific lowerings

We define Spyre-specific lowerings from ATen operations to Inductor's
loop level IR in [lowering.py](../torch_spyre/_inductor/lowering.py) using the `@lowering.register_lowering`
decorator.

### Spyre-specific decompositions

We define Spyre-specific decompositions in [decompositions.py](../torch_spyre/_inductor/decompositions.py)
using the `@register_decomposition` decorator.  Decompositions are graph transformations
that are performed before the graph is lowered to loop level IR.

### Spyre-specific OpFuncs

For Spyre OpFuncs that do not have corresponding ATen operations, we use
the `@torch.library.custom_op` decorator to define a new operation in
[customops.py](../torch_spyre/_inductor/customops.py). This has two pieces:
+ defining the signature of the operation (using `@custom_op`)
+ defining its fake function (using the `@opname.register_fake` that is defined as part of the `@custom_op`)

In addition when defining a custom op, you will also need to do one of:
+ register a lowering for the custom op in [lowering.py](../torch_spyre/_inductor/lowering.py) and
  adding a method to `SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py).
  A cannonical example is `spyre.clamp`.
+ register a decomposition for the custom op in [decompositions.py](../torch_spyre/_inductor/decompositions.py).
  A cannonical example is `spyre.compact`.
+ define a CustomPrePass or CustomPostPass that defines a more general graph
  rewrite that removes the custom op. We currently have no custom ops that use this option.
