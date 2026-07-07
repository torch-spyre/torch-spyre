---
name: debug-compilation
description: "Systematic debugging of torch.compile failures on the Spyre backend. Covers each pipeline stage from Dynamo tracing through runtime execution, with diagnostic env vars and common error patterns."
---

# Debugging Spyre Compilation Failures

When `torch.compile()` fails on Spyre, the error originates from one of eight
pipeline stages. This skill helps identify which stage failed and how to fix it.

See `common-errors.md` for an error-message → cause → fix lookup table.

---

## Diagnostic Environment Variables

Set these before reproducing the failure:

```bash
export TORCH_SPYRE_DEBUG=1          # C++ debug logging, -O0 build
export SENCORES=1                   # Single core (isolates issues; multi-core bugs won't reproduce)
export TORCH_LOGS="+inductor"       # Verbose Inductor logging
export TORCH_COMPILE_DEBUG=1        # Dump debug artifacts to torch_compile_debug/
```

Additional logging:

```bash
export DT_DEEPRT_VERBOSE=-1        # Reduce runtime noise
export TORCH_SPYRE_DOWNCAST_WARN=0  # Suppress dtype warnings
```

---

## Pipeline Stages (in order)

### Stage 1: Dynamo Tracing

**File:** PyTorch core (`torch._dynamo`)

**What happens:** Dynamo captures an FX graph from the Python model.

**Failure symptoms:**

- `torch._dynamo.exc.Unsupported`
- Graph breaks
- `TorchDynamoException`

**Debugging:**

- Check `TORCH_LOGS="+dynamo"` output
- Look for unsupported Python constructs or dynamic control flow
- This is a PyTorch-core issue, not Spyre-specific

---

### Stage 2: Decompositions

**File:** `torch_spyre/_inductor/decompositions.py`

**What happens:** FX graph ops are rewritten into simpler ops that Spyre
supports.

**Failure symptoms:**

- Op appears in the graph that has no lowering
- Unexpected op composition after decomposition

**Debugging:**

- Check if the op has a `@register_spyre_decomposition`
- Check `spyre_decompositions_to_exclude` for intentionally excluded ops
- Use `TORCH_COMPILE_DEBUG=1` to inspect the pre/post-decomposition graph

---

### Stage 3: Lowering

**File:** `torch_spyre/_inductor/lowering.py`

**What happens:** FX graph ops are lowered to Inductor's LoopLevelIR.

**Failure symptoms:**

- `LoweringException`
- Missing lowering for an op
- `Unsupported("...")` from lowering code

**Debugging:**

- Check if the op has a `@register_spyre_lowering`
- For custom ops, verify the lowering is registered for `torch.ops.spyre.*`
- Check the fallback registry in `torch_spyre/ops/fallbacks.py`

---

### Stage 4: Layout Propagation

**File:** `torch_spyre/_inductor/propagate_layouts.py`

**What happens:** `SpyreTensorLayout` is assigned to each IR node, converting
from standard PyTorch layouts to tiled stick layouts.

**Failure symptoms:**

- `"does not have FixedTiledLayout"` errors
- Layout mismatch assertions
- `dim_map` errors

**Debugging:**

- Check that all input tensors have compatible shapes
- Verify stick alignment (innermost dim should be padded to multiples of 64
  for fp16)
- Look for unsupported reshape/view operations

---

### Stage 5: SpyreKernel Codegen

**File:** `torch_spyre/_inductor/spyre_kernel.py`

**What happens:** LoopLevelIR is translated to `KernelSpec` objects via
`SpyreOpFuncs`.

**Failure symptoms:**

- `AttributeError: 'SpyreOpFuncs' has no attribute '<op_name>'`
- Missing `@staticmethod` for an op
- Wrong `op_info` dict structure

**Debugging:**

- Check `SpyreOpFuncs` class for the missing method
- Compare with existing methods (e.g., `add`, `softplus`) for the pattern
- Verify the method name matches what the lowering produces

---

### Stage 6: SuperDSC Generation

**File:** `torch_spyre/_inductor/codegen/superdsc.py`

**What happens:** `KernelSpec` → SuperDSC JSON descriptor for the backend
compiler.

**Failure symptoms:**

- KeyError in `generate_sdsc()`
- Invalid SuperDSC JSON structure
- Op dispatch failure

**Debugging:**

- Check `generate_sdsc()` dispatch logic
- Verify op handlers in `torch_spyre/_inductor/codegen/compute_ops.py`
- Use `TORCH_COMPILE_DEBUG=1` to inspect the generated JSON

---

### Stage 7: Backend Compiler

**File:** External `dxp_standalone` binary

**What happens:** SuperDSC JSON → `g2.graph.cbor` binary.

**Failure symptoms:**

- `dxp_standalone` exit code != 0
- Compilation timeout
- Invalid CBOR output

**Debugging:**

- Check `TORCH_COMPILE_DEBUG=1` output for the input JSON
- Run `dxp_standalone` manually on the JSON to get detailed errors
- Check `torch_spyre/execution/async_compile.py` for the invocation

---

### Stage 8: Runtime Execution

**File:** `torch_spyre/execution/kernel_runner.py`

**What happens:** `SpyreSDSCKernelRunner` calls `_C.launch_kernel()` to
execute the compiled binary.

**Failure symptoms:**

- `launch_kernel` failure
- Runtime segfault
- Wrong results (numerics)

**Debugging:**

- Use `TORCH_SPYRE_DEBUG=1` for C++ runtime logging
- Check tensor shapes and layouts at the kernel boundary
- For numeric issues, compare with `compare_with_cpu()` using tighter
  tolerances

---

## Common Debugging Workflow

1. **Reproduce** with minimal model and `SENCORES=1`
2. **Set env vars**: `TORCH_SPYRE_DEBUG=1 TORCH_COMPILE_DEBUG=1
   TORCH_LOGS="+inductor"`
3. **Identify the stage** from the error traceback
4. **Check the relevant file** listed above
5. **Inspect debug artifacts** in `torch_compile_debug/` directory
6. **Fix or report** — see `common-errors.md` for known patterns
