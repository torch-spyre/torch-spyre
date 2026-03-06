#!/usr/bin/env python3
"""
Educational debugging walkthrough for torch-spyre mul.py execution.

This script adds strategic breakpoints at key execution points to help trace
the flow from user code through PyTorch, torch-spyre compilation, and runtime.

Usage:
    python examples/mul_debug.py

For C++ debugging:
    gdb --args python examples/mul_debug.py
    (gdb) source examples/gdb_breakpoints.txt
    (gdb) run
"""

import torch
import logging
import os
import sys

# ============================================================================
# SETUP: Enable comprehensive logging
# ============================================================================

print("=" * 80)
print("TORCH-SPYRE EDUCATIONAL DEBUGGING WALKTHROUGH")
print("=" * 80)

# Enable C++ debug output
os.environ['SPYRE_DEBUG'] = '1'

# Enable PyTorch compilation logs
os.environ['TORCH_LOGS'] = '+dynamo,+inductor'

# Enable inductor debug mode
torch._inductor.config.debug = True
torch._inductor.config.trace.enabled = True

# Configure Python logging
from torch_spyre._inductor.logging_utils import get_inductor_logger

loggers = [
    "spyre_kernel",
    "lowering", 
    "kernel_runner",
    "passes"
]

for logger_name in loggers:
    logger = get_inductor_logger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

print("\n✓ Logging configured")
print("  - SPYRE_DEBUG=1 (C++ debug output)")
print("  - TORCH_LOGS=+dynamo,+inductor")
print("  - Python loggers: " + ", ".join(loggers))

# ============================================================================
# BREAKPOINT 1: Initial Setup
# ============================================================================

print("\n" + "=" * 80)
print("BREAKPOINT 1: Initial Setup")
print("=" * 80)
print("About to create tensors and device object...")
print("\nPress Enter to continue...")
input()

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

x = torch.rand(128, 64, dtype=torch.float16)
y = torch.rand(128, 64, dtype=torch.float16)

print(f"\n✓ Created tensors:")
print(f"  x: shape={x.shape}, dtype={x.dtype}, device={x.device}")
print(f"  y: shape={y.shape}, dtype={y.dtype}, device={y.device}")
print(f"  DEVICE: {DEVICE}")

# ============================================================================
# BREAKPOINT 2: CPU Baseline Execution
# ============================================================================

print("\n" + "=" * 80)
print("BREAKPOINT 2: CPU Baseline Execution")
print("=" * 80)
print("About to execute torch.mul on CPU for baseline...")
print("\nPress Enter to continue...")
input()

cpu_result = torch.mul(x, y)

print(f"\n✓ CPU execution complete:")
print(f"  Result shape: {cpu_result.shape}")
print(f"  Result dtype: {cpu_result.dtype}")
print(f"  Sample values: {cpu_result[0, :5]}")

# ============================================================================
# BREAKPOINT 3: Tensor Transfer to Device
# ============================================================================

print("\n" + "=" * 80)
print("BREAKPOINT 3: Tensor Transfer to Device")
print("=" * 80)
print("About to transfer tensors to Spyre device...")
print("\nKey operations that will occur:")
print("  1. SpyreAllocator::allocate() - Allocate device memory")
print("  2. SpyreTensorLayout::init() - Compute device layout")
print("  3. copy_host_to_device() - DMA transfer with layout conversion")
print("\nC++ Breakpoints to set in GDB:")
print("  - break spyre::SpyreAllocator::allocate")
print("  - break spyre::SpyreTensorLayout::init")
print("  - break spyre::copy_host_to_device")
print("\nPress Enter to continue...")
input()

x_device = x.to(DEVICE)
y_device = y.to(DEVICE)

print(f"\n✓ Device transfer complete:")
print(f"  x_device: device={x_device.device}")
print(f"  y_device: device={y_device.device}")

# Get device layout information
try:
    x_layout = x_device.device_tensor_layout()
    print(f"\n  Device Layout for x_device:")
    print(f"    {x_layout}")
    print(f"    - device_size: {x_layout.device_size}")
    print(f"    - dim_map: {x_layout.dim_map}")
    print(f"    - device_dtype: {x_layout.device_dtype}")
    print(f"    - elems_per_stick: {x_layout.elems_per_stick()}")
except Exception as e:
    print(f"  Could not get layout: {e}")

# ============================================================================
# BREAKPOINT 4: Compilation Entry
# ============================================================================

print("\n" + "=" * 80)
print("BREAKPOINT 4: Compilation Entry (torch.compile)")
print("=" * 80)
print("About to compile the lambda function...")
print("\nKey operations that will occur:")
print("  1. Dynamo - Capture FX Graph")
print("  2. Inductor - Route to torch-spyre backend")
print("  3. Decompositions - Simplify operations")
print("  4. Lowering - Convert to LoopLevelIR")
print("\nPython Breakpoints to set:")
print("  - torch_spyre/_inductor/__init__.py:96 (_uses_spyre)")
print("  - torch_spyre/_inductor/lowering.py (register_spyre_lowering)")
print("\nPress Enter to continue...")
input()

# Reset dynamo to see fresh compilation
import torch._dynamo as dynamo
dynamo.reset()

compiled = torch.compile(lambda a, b: torch.mul(a, b))

print("\n✓ Compilation object created (not yet executed)")

# ============================================================================
# BREAKPOINT 5: First Execution (Compilation Happens Here)
# ============================================================================

print("\n" + "=" * 80)
print("BREAKPOINT 5: First Execution (Triggers Compilation)")
print("=" * 80)
print("About to execute compiled function for the first time...")
print("\nThis will trigger the full compilation pipeline:")
print("\n  FRONTEND COMPILATION:")
print("    1. FX Graph capture (Dynamo)")
print("    2. Graph decomposition")
print("    3. Lowering to LoopLevelIR")
print("    4. Kernel generation (SpyreKernel)")
print("    5. Work division planning")
print("    6. SuperDSC JSON generation")
print("\n  BACKEND COMPILATION:")
print("    7. DeepTools compilation (g2.graph.cbor)")
print("\n  RUNTIME EXECUTION:")
print("    8. Kernel loading and patching")
print("    9. Device execution")
print("    10. Result transfer back to CPU")
print("\nPython Breakpoints to set:")
print("  - torch_spyre/_inductor/spyre_kernel.py (SpyreKernel class)")
print("  - torch_spyre/_inductor/codegen/superdsc.py (SuperDSC generation)")
print("  - torch_spyre/_inductor/runtime/kernel_runner.py:39 (run method)")
print("\nC++ Breakpoints to set in GDB:")
print("  - break spyre::launchKernel")
print("  - break spyre::GlobalRuntime::get")
print("\nPress Enter to continue...")
input()

print("\n>>> EXECUTING COMPILED FUNCTION <<<\n")

compiled_result = compiled(x_device, y_device).cpu()

print("\n✓ Execution complete!")
print(f"  Result shape: {compiled_result.shape}")
print(f"  Result dtype: {compiled_result.dtype}")

# ============================================================================
# BREAKPOINT 6: Results Comparison
# ============================================================================

print("\n" + "=" * 80)
print("BREAKPOINT 6: Results Comparison")
print("=" * 80)
print("Comparing CPU vs Spyre results...")
print("\nPress Enter to continue...")
input()

cpu_delta = torch.abs(compiled_result - cpu_result).max()

print(f"\n✓ Comparison complete:")
print(f"  CPU result sample: {cpu_result[0, :5]}")
print(f"  Spyre result sample: {compiled_result[0, :5]}")
print(f"  Max absolute difference: {cpu_delta}")
print(f"  Match: {'✓ PASS' if cpu_delta < 1e-3 else '✗ FAIL'}")

# ============================================================================
# BREAKPOINT 7: Second Execution (Cached)
# ============================================================================

print("\n" + "=" * 80)
print("BREAKPOINT 7: Second Execution (Should Use Cached Kernel)")
print("=" * 80)
print("About to execute compiled function again...")
print("\nThis should be much faster - no recompilation needed!")
print("The kernel binary is already loaded and ready.")
print("\nPress Enter to continue...")
input()

print("\n>>> EXECUTING COMPILED FUNCTION (2nd time) <<<\n")

compiled_result_2 = compiled(x_device, y_device).cpu()

print("\n✓ Second execution complete!")
print(f"  Results match first execution: {torch.allclose(compiled_result, compiled_result_2)}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)

print("\nExecution Flow Traced:")
print("  1. ✓ Initial setup and tensor creation")
print("  2. ✓ CPU baseline execution")
print("  3. ✓ Device memory allocation and transfer")
print("  4. ✓ Compilation object creation")
print("  5. ✓ First execution (full compilation pipeline)")
print("  6. ✓ Results comparison")
print("  7. ✓ Second execution (cached kernel)")

print("\nKey Files Involved:")
print("\n  Python Layer:")
print("    - torch_spyre/_inductor/__init__.py (backend registration)")
print("    - torch_spyre/_inductor/lowering.py (operation lowering)")
print("    - torch_spyre/_inductor/spyre_kernel.py (kernel generation)")
print("    - torch_spyre/_inductor/codegen/superdsc.py (code generation)")
print("    - torch_spyre/_inductor/runtime/kernel_runner.py (execution)")
print("\n  C++ Layer:")
print("    - torch_spyre/csrc/spyre_mem.cpp (memory management)")
print("    - torch_spyre/csrc/spyre_tensor_impl.cpp (tensor layout)")
print("    - torch_spyre/csrc/module.cpp (runtime and kernel launch)")

print("\nGenerated Artifacts:")
print(f"  Check /tmp/torchinductor_{os.getenv('USER', 'user')}/ for:")
print("    - FX graphs")
print("    - Generated Python code")
print("    - SuperDSC JSON files")
print("    - Compiled binaries (g2.graph.cbor)")

print("\n" + "=" * 80)
print("WALKTHROUGH COMPLETE!")
print("=" * 80)

# Made with Bob
