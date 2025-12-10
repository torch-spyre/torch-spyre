## Spyre Inductor Operation Cookbook

This document describe the common patterns used to define operations
in the Inductor Spyre backend.

### Direct mapping from ATen to OpFunc

If a core ATen operation can be implemented with a single Spyre OpFunc,
then adding it to our backend requires:
+ adding an entry to `opfunc_mapping` in [opfuncs.py](../torch_spyre/_inductor/opfuncs.py)
Cannonical examples are `aten.add` for a pointwise operation and
`aten.sum` for a reduction.

Some ATen operations that can be directly mapped to a Spyre OpFunc
have default decompositions defined by Inductor. To disable the default
decompostion in addition to the two steps above, we also add a
method to `OpOverrides` in [opoverrides.py](../torch_spyre/_inductor/opoverrides.py).
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
+ register a lowering for the custom op (eg `aten.mm.default`) and add method to `OpOverrides` in [opoverrides.py](../torch_spyre/_inductor/opoverrides.py).
+ register a decomposition for the custom op (eg `spyre.compact`)
+ define a CustomPrePass or CustomPostPass that defines a more general graph
  rewrite that removes the custom op
