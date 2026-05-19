# Common Compilation Errors

Error message → likely cause → fix.

---

## "Spyre backend does not support: ..."

**Source:** `torch_spyre/_inductor/errors.py` — `Unsupported()` class

**Cause:** An operation reached a code path that explicitly raises
`Unsupported`. This means the op is recognized but not yet implemented.

**Fix:**

- Check which file raised it (traceback will show)
- Implement the missing support (see `add-spyre-operation` skill)
- Or register a CPU fallback in `torch_spyre/ops/fallbacks.py`

---

## "does not have FixedTiledLayout"

**Source:** `torch_spyre/_inductor/propagate_layouts.py`

**Cause:** The layout propagation pass could not assign a `FixedTiledLayout`
to a node. Usually means an unsupported op or reshape produced a layout that
cannot be tiled.

**Fix:**

- Check if the op needs a custom layout propagation handler
- Verify input tensor shapes are compatible with stick alignment
- Check for unsupported view/reshape operations in the graph

---

## AttributeError on SpyreOpFuncs

**Error:** `AttributeError: type object 'SpyreOpFuncs' has no attribute '<op>'`

**Source:** `torch_spyre/_inductor/spyre_kernel.py`

**Cause:** The lowering produced an op that has no corresponding method in
`SpyreOpFuncs`.

**Fix:**

- Add a `@staticmethod` method to `SpyreOpFuncs` in `spyre_kernel.py`
- Method name must match the op name from the lowering

---

## Dtype Errors

**Error:** Variations of dtype mismatch, unexpected float32, or
`"expected float16 but got float32"`

**Cause:** Spyre default dtype is `float16`. Operations may receive
`float32` inputs from PyTorch defaults.

**Fix:**

- Ensure inputs are cast to `float16` before compilation
- Check `TORCH_SPYRE_DOWNCAST_WARN=0` to see if downcasting warnings
  appear
- Add explicit `to_dtype` handling in the lowering if needed
- Check `SPYRE_FP32_OPS` in `torch_spyre/_inductor/constants.py` for
  ops that run in fp32

---

## Shape Mismatch in Work Division

**Error:** Assertions in `torch_spyre/_inductor/work_division.py` or
dimension label mismatches

**Cause:** The work division planning cannot split the operation's
iteration space across cores.

**Fix:**

- Check `docs/source/compiler/work_division_planning.md` for dimension
  label rules
- Verify the op's `dimensions` and `scales` in its `KernelSpec`
- Try `SENCORES=1` to bypass multi-core splitting for debugging

---

## Graph Break

**Error:** `torch._dynamo` reports a graph break

**Cause:** Dynamo cannot trace through certain Python constructs. Not
Spyre-specific but affects what reaches the Spyre backend.

**Fix:**

- Check `TORCH_LOGS="+dynamo"` for the break reason
- Restructure the model code to avoid the unsupported construct
- Use `torch._dynamo.allow_in_graph` if appropriate

---

## dxp\_standalone Compilation Failure

**Error:** Non-zero exit code from `dxp_standalone`, timeout, or CBOR
errors

**Cause:** The generated SuperDSC JSON is invalid or uses unsupported
features.

**Fix:**

- Use `TORCH_COMPILE_DEBUG=1` to find the JSON in `torch_compile_debug/`
- Run `dxp_standalone` manually with the JSON to get detailed output
- Check `torch_spyre/_inductor/codegen/superdsc.py` for the generated
  descriptor
- Verify op parameters in `torch_spyre/_inductor/codegen/compute_ops.py`

---

## launch\_kernel Runtime Failure

**Error:** Errors from `_C.launch_kernel()` or segfaults during execution

**Cause:** The compiled binary is valid but runtime state is incorrect
(wrong tensor sizes, misaligned memory, missing constants).

**Fix:**

- Use `TORCH_SPYRE_DEBUG=1` for C++ debug output
- Check tensor argument layouts in `KernelSpec.args`
- Verify constant encoding in `torch_spyre/execution/kernel_runner.py`
- Check `SENCORES` matches the core division in the compiled binary

---

## Numeric Mismatches

**Error:** `torch.testing.assert_close` failures in tests

**Cause:** Floating-point precision differences between CPU and Spyre,
often due to fp16 precision limits.

**Fix:**

- Increase tolerances: `atol=0.1, rtol=0.1` is standard for fp16
- Check if the op accumulates in fp16 vs fp32
- Use `compare_with_pytorch()` for reference comparison against a
  different PyTorch implementation
- Check for numerically unstable operations (division by near-zero, etc.)
