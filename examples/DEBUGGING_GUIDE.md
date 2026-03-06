# Educational Debugging Guide for torch-spyre

This guide provides a comprehensive walkthrough for debugging `mul.py` execution through the torch-spyre stack, from Python user code down to C++ runtime and hardware execution.

## Quick Start

### Option 1: Interactive Python Walkthrough (Recommended for Beginners)

```bash
python examples/mul_debug.py
```

This script will pause at strategic points and explain what's happening at each layer.

### Option 2: Python Debugger (pdb)

```bash
python -m pdb examples/mul_debug.py
```

### Option 3: VSCode Debugging

1. Copy `examples/vscode_debug_config.json` to `.vscode/launch.json`
2. Open `examples/mul_debug.py` in VSCode
3. Set breakpoints by clicking in the left margin
4. Press F5 to start debugging

### Option 4: C++ Debugging with GDB

```bash
# 1. Build with debug symbols
CFLAGS="-g -O0" CXXFLAGS="-g -O0" pip install -e .

# 2. Start GDB
gdb --args python examples/mul_debug.py

# 3. Load breakpoints
(gdb) source examples/gdb_breakpoints.txt

# 4. Run
(gdb) run
```

## Execution Flow Overview

```
User Code (mul.py)
    ↓
PyTorch Frontend
    ↓
torch.compile / Dynamo (Graph Capture)
    ↓
Inductor Backend Router
    ↓
torch-spyre Frontend Compiler
    ├─ Decompositions
    ├─ Lowering to LoopLevelIR
    ├─ Kernel Generation
    ├─ Work Division Planning
    └─ SuperDSC JSON Generation
    ↓
DeepTools Backend Compiler
    └─ Hardware Binary (g2.graph.cbor)
    ↓
Runtime Execution (C++)
    ├─ Memory Allocation
    ├─ Data Transfer (Host → Device)
    ├─ Kernel Launch
    ├─ Device Execution
    └─ Result Transfer (Device → Host)
```

## Strategic Breakpoint Locations

### Python Layer Breakpoints

#### 1. Compilation Entry
**File:** `torch_spyre/_inductor/__init__.py`  
**Line:** 96  
**Function:** `_uses_spyre()`  
**Purpose:** Detects if graph uses Spyre device and routes to torch-spyre

```python
# Set breakpoint here to see when compilation is triggered
if _uses_spyre(gm, example_inputs):
    # torch-spyre compilation path
```

#### 2. Operation Lowering
**File:** `torch_spyre/_inductor/lowering.py`  
**Line:** 46  
**Function:** `register_spyre_lowering()`  
**Purpose:** Maps ATen operations to Spyre implementations

```python
# Set breakpoint to see how torch.mul is lowered
def register_spyre_lowering(op, name=None, ...):
```

#### 3. Kernel Generation
**File:** `torch_spyre/_inductor/spyre_kernel.py`  
**Line:** 100  
**Class:** `SpyreKernel`  
**Purpose:** Generates kernel specifications from LoopLevelIR

```python
# Set breakpoint in SpyreKernel methods to see kernel generation
class SpyreKernel(Kernel):
    def codegen_pointwise(self, ...):
```

#### 4. Work Division Planning
**File:** `torch_spyre/_inductor/core_division.py`  
**Purpose:** Determines how work is split across 32 Spyre cores

```python
# Set breakpoint to see parallelization decisions
def plan_work_division(op, tensors, num_cores):
```

#### 5. SuperDSC Generation
**File:** `torch_spyre/_inductor/codegen/superdsc.py`  
**Purpose:** Generates JSON specification for backend compiler

```python
# Set breakpoint to see SuperDSC JSON generation
def generate_superdsc(kernel_spec):
```

#### 6. Kernel Execution
**File:** `torch_spyre/_inductor/runtime/kernel_runner.py`  
**Line:** 39  
**Method:** `SpyreSDSCKernelRunner.run()`  
**Purpose:** Launches compiled kernel on device

```python
# Set breakpoint to see kernel launch
def run(self, *args, **kw_args):
    g2 = os.path.join(self.code_dir, "g2.graph.cbor")
    return launch_kernel(g2, actuals)
```

### C++ Layer Breakpoints

#### 1. Runtime Initialization
**File:** `torch_spyre/csrc/module.cpp`  
**Line:** 70  
**Function:** `spyre::_startRuntime()`  
**Purpose:** One-time initialization of Spyre runtime

```cpp
// GDB: break spyre::_startRuntime
void _startRuntime() {
  std::shared_ptr<sendnn::RuntimeInterface> base_runtime;
  auto s = flex::CreateRuntimeInterface(&base_runtime);
  // ...
}
```

#### 2. Memory Allocation
**File:** `torch_spyre/csrc/spyre_mem.cpp`  
**Line:** 453  
**Method:** `SpyreAllocator::allocate()`  
**Purpose:** Allocates device memory from pool

```cpp
// GDB: break spyre::SpyreAllocator::allocate
at::DataPtr allocate(size_t nbytes) override {
  auto allocator = getAllocator(device_id);
  flex::DeviceMemoryAllocationPtr data;
  allocator->TryAllocate(&data, nbytes, 0);
  // ...
}
```

#### 3. Tensor Layout Computation
**File:** `torch_spyre/csrc/spyre_tensor_impl.cpp`  
**Line:** 91  
**Method:** `SpyreTensorLayout::init()`  
**Purpose:** Computes device layout with tiling and padding

```cpp
// GDB: break spyre::SpyreTensorLayout::init
void SpyreTensorLayout::init(std::vector<int64_t> host_size,
                              c10::ScalarType dtype) {
  // Compute device_size and dim_map
  // ...
}
```

#### 4. Host → Device Transfer
**File:** `torch_spyre/csrc/spyre_mem.cpp`  
**Line:** 404  
**Function:** `copy_host_to_device()`  
**Purpose:** DMA transfer with layout conversion

```cpp
// GDB: break spyre::copy_host_to_device
auto copy_host_to_device(const at::Tensor& self, const at::Tensor& dst) {
  std::shared_ptr<sendnn::GraphLoader> gl = create_dma_graph(self, dst, true);
  // Create DMA graph, execute transfer
  // ...
}
```

#### 5. Kernel Launch
**File:** `torch_spyre/csrc/module.cpp`  
**Line:** 95  
**Function:** `launchKernel()`  
**Purpose:** Loads binary, patches graph, executes on device

```cpp
// GDB: break spyre::launchKernel
void launchKernel(std::string g2_path, std::vector<at::Tensor> args) {
  auto gl = sendnn::GraphLoader(GlobalRuntime::get());
  
  // Load compiled kernel
  auto g2 = sendnn::Graph();
  sendnn::Deserialize(&g2, g2_path);
  
  // Patch with tensor pointers
  // ...
  
  // Execute
  gl.Compute(sen_outputs, sen_inputs, 2);
}
```

#### 6. Device → Host Transfer
**File:** `torch_spyre/csrc/spyre_mem.cpp`  
**Line:** 423  
**Function:** `copy_device_to_host()`  
**Purpose:** Transfer results back to CPU

```cpp
// GDB: break spyre::copy_device_to_host
auto copy_device_to_host(const at::Tensor& self, const at::Tensor& dst) {
  std::shared_ptr<sendnn::GraphLoader> gl = create_dma_graph(self, dst, false);
  // ...
}
```

## Debugging Workflows

### Workflow 1: Trace Full Execution

```bash
# Terminal 1: Run with all logging
export SPYRE_DEBUG=1
export TORCH_LOGS="+dynamo,+inductor,+aot,+graph_breaks"
python examples/mul_debug.py
```

### Workflow 2: Debug Compilation Only

```python
import torch
torch._inductor.config.debug = True
torch._inductor.config.trace.enabled = True

# Your code here
compiled = torch.compile(lambda a, b: torch.mul(a, b))
# Check /tmp/torchinductor_<user>/ for generated files
```

### Workflow 3: Debug C++ Runtime

```bash
# Build with debug symbols
CFLAGS="-g -O0" CXXFLAGS="-g -O0" pip install -e .

# Run with GDB
gdb --args python examples/mul.py
(gdb) break spyre::launchKernel
(gdb) run
(gdb) bt  # Show backtrace when breakpoint hits
```

### Workflow 4: Profile Performance

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    result = compiled(x_device, y_device)

print(prof.key_averages().table(sort_by="cpu_time_total"))
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
```

## Understanding the Output

### Python Logging Output

```
[spyre_kernel] Creating kernel for pointwise operation
[lowering] Lowering torch.mul to Spyre OpFunc
[kernel_runner] RUN: kernel_0 /tmp/.../g2.graph.cbor
```

### C++ Debug Output (SPYRE_DEBUG=1)

```
[_startRuntime] starting runtime
[allocate] allocating 16384 (bytes) on Spyre0
[copy_host_to_device] Tensor info on CPU (Size:[128, 64], ...)
[launchKernel] Launching compiled kernel
```

### Generated Artifacts

After compilation, check `/tmp/torchinductor_<user>/` for:

```
/tmp/torchinductor_<user>/<hash>/
├── fx_graph_*.py          # FX Graph representation
├── output_code.py         # Generated Python wrapper
├── superdsc.json          # SuperDSC specification
└── g2.graph.cbor          # Compiled hardware binary
```

## Common Debugging Scenarios

### Scenario 1: Compilation Fails

**Symptoms:** Error during `torch.compile()` or first execution  
**Debug Steps:**
1. Enable verbose logging: `TORCH_LOGS="+dynamo,+inductor"`
2. Check FX graph: Look for unsupported operations
3. Set breakpoint in `torch_spyre/_inductor/lowering.py`
4. Verify operation is registered

### Scenario 2: Wrong Results

**Symptoms:** Output doesn't match CPU baseline  
**Debug Steps:**
1. Check tensor layouts: `tensor.device_tensor_layout()`
2. Verify data transfer: Set breakpoint in `copy_host_to_device`
3. Inspect SuperDSC JSON for correct operation parameters
4. Compare intermediate values with CPU execution

### Scenario 3: Performance Issues

**Symptoms:** Execution slower than expected  
**Debug Steps:**
1. Profile with `torch.profiler`
2. Check work division: Look at `op_dim_splits` in logs
3. Verify kernel is cached (second execution should be faster)
4. Check memory transfer overhead

### Scenario 4: Memory Errors

**Symptoms:** Allocation failures or segfaults  
**Debug Steps:**
1. Enable C++ debugging: `SPYRE_DEBUG=1`
2. Run with GDB: `gdb --args python examples/mul.py`
3. Set breakpoint in `SpyreAllocator::allocate`
4. Check tensor sizes and device memory capacity

## Tips and Tricks

### Tip 1: Conditional Breakpoints in pdb

```python
import pdb

def conditional_break(condition, message=""):
    if condition:
        print(f"Breaking: {message}")
        pdb.set_trace()

# Example: Break only for specific tensor shapes
conditional_break(x.shape == (128, 64), "Found target shape")
```

### Tip 2: Inspect Generated Code

```python
import torch
torch._inductor.config.debug = True

# After compilation, generated code is in /tmp/torchinductor_<user>/
# Read output_code.py to see the generated Python wrapper
```

### Tip 3: Compare with CPU Execution

```python
# Run same operation on CPU for comparison
cpu_result = torch.mul(x_cpu, y_cpu)
spyre_result = compiled(x_device, y_device).cpu()

# Check differences
diff = torch.abs(cpu_result - spyre_result)
print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")
```

### Tip 4: Use Logging Levels

```python
import logging
from torch_spyre._inductor.logging_utils import get_inductor_logger

# Set different levels for different components
get_inductor_logger("spyre_kernel").setLevel(logging.DEBUG)
get_inductor_logger("lowering").setLevel(logging.INFO)
get_inductor_logger("kernel_runner").setLevel(logging.WARNING)
```

## Additional Resources

- **Documentation:** See `docs/` directory for architecture details
- **RFCs:** See `RFCs/` directory for design documents
- **Tests:** See `tests/` directory for example usage patterns
- **PyTorch Docs:** https://pytorch.org/docs/stable/torch.compiler.html

## Troubleshooting

### Issue: "Import torch could not be resolved"

This is a type checker warning, not a runtime error. The code will run fine.

### Issue: GDB breakpoints not hitting

Make sure you compiled with debug symbols:
```bash
CFLAGS="-g -O0" CXXFLAGS="-g -O0" pip install -e .
```

### Issue: No output from SPYRE_DEBUG

The C++ extension may not be built with debug support. Rebuild:
```bash
pip install -e . --force-reinstall --no-cache-dir
```

### Issue: Can't find generated files

Check the temp directory:
```bash
ls -la /tmp/torchinductor_$USER/
```

## Summary

This guide provides multiple approaches to debug torch-spyre execution:

1. **Interactive walkthrough** (`mul_debug.py`) - Best for learning
2. **Python debugger** (pdb/ipdb) - For Python-level debugging
3. **VSCode debugger** - For IDE-integrated debugging
4. **GDB** - For C++ runtime debugging
5. **Logging** - For understanding execution flow
6. **Profiling** - For performance analysis

Start with the interactive walkthrough to understand the execution flow, then use specific debugging tools as needed for your investigation.