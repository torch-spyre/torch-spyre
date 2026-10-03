# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pytest configuration for model tests with automatic fallback integration.

This conftest.py automatically patches the test runner to use compilation fallbacks
when TORCH_SPYRE_MODEL_FALLBACKS environment variable is set.
"""

import os
import pytest
import sys
import torch
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def enable_compilation_fallbacks():
    """
    Automatically enable compilation fallbacks when TORCH_SPYRE_MODEL_FALLBACKS is set.
    
    This fixture patches runner._maybe_compile_call() to catch compilation errors
    and automatically fall back to CPU eager mode.
    """
    fallback_models = os.environ.get("TORCH_SPYRE_MODEL_FALLBACKS", "").strip()
    
    if not fallback_models:
        # Fallbacks not enabled - skip patching
        yield
        return
    
    # Print activation message
    msg = "\n" + "=" * 80 + "\n"
    msg += f" TORCH_SPYRE_MODEL_FALLBACKS detected: {fallback_models}\n"
    msg += "  Enabling automatic compilation fallbacks for model tests\n"
    msg += "=" * 80 + "\n"
    print(msg, file=sys.stderr)
    
    # Import modules - use sys.modules to get already imported runner
    try:
        # Get runner from sys.modules (it's already imported by test_model_ops.py)
        if 'runner' in sys.modules:
            runner = sys.modules['runner']
        else:
            # Try importing from current directory
            from . import runner
    except ImportError as e:
        print(f"Warning: Could not import runner module for fallback: {e}", file=sys.stderr)
        yield
        return
    
    # Save original function
    original_maybe_compile_call = runner._maybe_compile_call
    
    # Create wrapped version that catches compilation errors
    def wrapped_maybe_compile_call(fn, sample, device, compile_backend):
        """
        Wrapped version of _maybe_compile_call that catches compilation errors.
        
        This automatically falls back to CPU eager mode when compilation fails
        on Spyre device.
        """
        # If no compilation backend or CPU device, just run normally
        if compile_backend is None or device.type == "cpu":
            return original_maybe_compile_call(fn, sample, device, compile_backend)
        
        # Try Spyre compilation first
        try:
            return original_maybe_compile_call(fn, sample, device, compile_backend)
        except Exception as e:
            # Check if this is a compilation error we should handle
            error_type = type(e).__name__
            error_msg = str(e)
            should_fallback = any([
                "InductorError" in error_type,
                "CalledProcessError" in error_type,
                "NotImplementedError" in error_type,
                "TypeError" in error_type,
                "RuntimeError" in error_type,
                "AttributeError" in error_type,
                "DtException" in error_msg,
                "Error in codegen" in error_msg,
                "Illegal ddl" in error_msg,
                "Unexpected stick expression" in error_msg,
                "cannot restickify" in error_msg,
            ])
            
            if should_fallback:
                # Print clear fallback message
                print("\n" + "=" * 80, file=sys.stderr)
                print(f"[COMPILATION FALLBACK] Spyre compilation failed, falling back to CPU", file=sys.stderr)
                print(f"  Error Type: {error_type}", file=sys.stderr)
                print(f"  Error Message: {error_msg[:200]}...", file=sys.stderr)
                print(f"  Device: {device} -> cpu", file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)
                
                # Move inputs to CPU and retry
                def to_cpu(x):
                    return x.to("cpu") if isinstance(x, torch.Tensor) else x
                
                cpu_input = to_cpu(sample.input)
                cpu_args = [to_cpu(a) for a in sample.args]
                cpu_kwargs = {k: to_cpu(v) for k, v in sample.kwargs.items()}
                
                # Create a new sample with CPU tensors
                class CPUSample:
                    def __init__(self, input, args, kwargs):
                        self.input = input
                        self.args = args
                        self.kwargs = kwargs
                
                cpu_sample = CPUSample(cpu_input, cpu_args, cpu_kwargs)
                cpu_device = torch.device("cpu")
                
                # Run on CPU (no compilation)
                result = original_maybe_compile_call(fn, cpu_sample, cpu_device, None)
                
                # Move result back to original device
                def restore(x):
                    return x.to(device) if isinstance(x, torch.Tensor) else x
                
                if isinstance(result, (tuple, list)):
                    return type(result)(restore(r) for r in result)
                return restore(result)
            else:
                # Not a compilation error we handle, re-raise
                raise
    
    # Apply patch
    runner._maybe_compile_call = wrapped_maybe_compile_call
    
    msg = "Compilation fallback patching complete - tests will automatically fall back to CPU on errors\n"
    msg += "=" * 80 + "\n"
    print(msg, file=sys.stderr)
    
    # Run tests
    yield
    
    # Restore original function
    runner._maybe_compile_call = original_maybe_compile_call
    
    msg = "\n" + "=" * 80 + "\n"
    msg += "Compilation fallback patching removed\n"
    msg += "=" * 80 + "\n"
    print(msg, file=sys.stderr)
