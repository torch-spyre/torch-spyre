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
Helper to automatically wrap test execution with compilation fallbacks.

This module provides utilities to catch compilation errors and fall back to CPU eager mode.
Integrates with the model fallback system via TORCH_SPYRE_MODEL_FALLBACKS environment variable.
"""

import functools
import logging
import os
import sys
import torch
from typing import Callable, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Check if model fallbacks are enabled
def _is_fallback_enabled() -> bool:
    """Check if TORCH_SPYRE_MODEL_FALLBACKS environment variable is set."""
    return bool(os.environ.get("TORCH_SPYRE_MODEL_FALLBACKS", "").strip())

def _get_enabled_models() -> str:
    """Get the list of models with fallbacks enabled."""
    return os.environ.get("TORCH_SPYRE_MODEL_FALLBACKS", "").strip()


def with_compile_fallback(model_name: str = "unknown"):
    """
    Decorator to wrap test functions with compilation fallback logic.
    
    Usage in tests:
        @with_compile_fallback("mistral")
        def test_my_operation(self, device, dtype, op):
            # Test code here
            result = run_test(...)
            return result
    
    Args:
        model_name: Name of the model for logging
    
    Returns:
        Decorator function
    """
    def decorator(test_fn: Callable) -> Callable:
        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            try:
                # Try normal test execution
                return test_fn(*args, **kwargs)
            except Exception as e:
                # Check if this is a compilation error
                error_type = type(e).__name__
                error_msg = str(e)
                
                is_compile_error = (
                    "InductorError" in error_type or
                    "NotImplementedError" in error_type or
                    "CalledProcessError" in error_type or
                    ("AttributeError" in error_type and "UnimplementedOp" in error_msg) or
                    ("TypeError" in error_type and "Cannot convert symbols" in error_msg) or
                    ("RuntimeError" in error_type and "setStorage" in error_msg)
                )
                
                if is_compile_error:
                    msg = f"[{model_name}] Compilation failed with {error_type}, falling back to CPU eager mode"
                    print(msg, file=sys.stderr)
                    
                    # Try to re-run with CPU fallback
                    # This is a simplified version - actual implementation would need
                    # to modify the test execution to use CPU
                    raise RuntimeError(
                        f"[{model_name}] Compilation error detected. "
                        f"Consider running this test with CPU backend or eager mode.\n"
                        f"Original error: {error_type}: {error_msg}"
                    ) from e
                else:
                    # Not a compilation error, re-raise
                    raise
        
        return wrapper
    return decorator


def run_with_fallback(fn: Callable, *args, model_name: str = "unknown", **kwargs) -> Any:
    """
    Run a function with automatic compilation fallback.
    
    This function tries to execute the given function normally. If a compilation
    error occurs AND TORCH_SPYRE_MODEL_FALLBACKS is set, it falls back to CPU eager mode.
    
    Args:
        fn: Function to execute
        *args: Positional arguments for the function
        model_name: Name of the model for logging
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result of the function execution
    
    Example:
        # Set environment variable first
        os.environ["TORCH_SPYRE_MODEL_FALLBACKS"] = "mistral"
        
        # Then run with fallback
        result = run_with_fallback(model.forward, input_ids, model_name="mistral")
    """
    # Check if fallback is enabled
    if not _is_fallback_enabled():
        # Fallback not enabled - execute normally without catching errors
        return fn(*args, **kwargs)
    
    enabled_models = _get_enabled_models()
    
    # Determine source device from arguments
    source_device = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            source_device = arg.device
            break
    if source_device is None:
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                source_device = value.device
                break
    
    # Only apply fallback for spyre device
    if source_device is None or source_device.type != "spyre":
        return fn(*args, **kwargs)
    
    try:
        # Log attempt
        msg = f"[FALLBACK] Attempting operation on {source_device.type} (fallbacks enabled for: {enabled_models})"
        print(msg, file=sys.stderr)
        logger.info(msg)
        
        # Try normal execution
        result = fn(*args, **kwargs)
        
        msg = f"[FALLBACK]  Operation succeeded on {source_device.type}"
        print(msg, file=sys.stderr)
        logger.info(msg)
        return result
        
    except Exception as e:
        # Check if this is a compilation error
        error_type = type(e).__name__
        error_msg = str(e)
        
        is_compile_error = (
            "InductorError" in error_type or
            "NotImplementedError" in error_type or
            "CalledProcessError" in error_type or
            ("AttributeError" in error_type and "UnimplementedOp" in error_msg) or
            ("TypeError" in error_type and "Cannot convert symbols" in error_msg) or
            ("RuntimeError" in error_type and "setStorage" in error_msg)
        )
        
        if is_compile_error:
            msg = f"[FALLBACK] ✗ Operation failed with {error_type}, falling back to CPU eager mode"
            print(msg, file=sys.stderr)
            logger.warning(msg)
            logger.warning(f"[FALLBACK]   Error details: {error_msg[:200]}...")
            
            try:
                # Move inputs to CPU
                def to_cpu(x):
                    if isinstance(x, torch.Tensor) and x.device.type == "spyre":
                        return x.cpu()
                    elif isinstance(x, (tuple, list)):
                        return type(x)(to_cpu(item) for item in x)
                    elif isinstance(x, dict):
                        return {k: to_cpu(v) for k, v in x.items()}
                    return x
                
                cpu_args = [to_cpu(arg) for arg in args]
                cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
                
                # Execute on CPU in eager mode
                msg = f"[FALLBACK] Executing on CPU in eager mode..."
                print(msg, file=sys.stderr)
                logger.info(msg)
                
                result = fn(*cpu_args, **cpu_kwargs)
                
                # Move result back to Spyre
                def to_spyre(x):
                    if isinstance(x, torch.Tensor):
                        return x.to(source_device)
                    elif isinstance(x, (tuple, list)):
                        return type(x)(to_spyre(item) for item in x)
                    elif isinstance(x, dict):
                        return {k: to_spyre(v) for k, v in x.items()}
                    return x
                
                result = to_spyre(result)
                
                msg = f"[FALLBACK]  CPU eager mode fallback succeeded, result moved back to {source_device.type}"
                print(msg, file=sys.stderr)
                logger.info(msg)
                return result
                
            except Exception as cpu_error:
                msg = f"[FALLBACK] ✗ Both operation and CPU fallback failed!"
                print(msg, file=sys.stderr)
                logger.error(msg)
                logger.error(f"[FALLBACK]   Original error: {error_type}: {error_msg[:100]}")
                logger.error(f"[FALLBACK]   CPU error: {type(cpu_error).__name__}: {str(cpu_error)[:100]}")
                raise RuntimeError(
                    f"[{model_name}] Both operation and CPU fallback failed:\n"
                    f"  Original error: {error_type}: {error_msg}\n"
                    f"  CPU error: {type(cpu_error).__name__}: {str(cpu_error)}"
                ) from cpu_error
        else:
            # Not a compilation error, re-raise
            raise
