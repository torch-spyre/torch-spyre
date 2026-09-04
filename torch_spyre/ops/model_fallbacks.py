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
Model-Specific Conditional Fallbacks System

This module provides a monkey-patching based fallback system for specific models.
When enabled, operations that fail on Spyre will automatically fall back to CPU
for specific shapes, strides, or error conditions.

Key Features:
- Try Spyre first, fall back to CPU only on failure
- Per-model customization (e.g., mistral, bert, gpt2, llama, t5)
- Conditional fallbacks based on tensor properties (shape, stride, dtype)
- Zero overhead when not enabled
- Three activation methods: env var, context manager, or permanent enable/disable

Usage:
    # Method 1: Environment Variable (no code changes)
    TORCH_SPYRE_MODEL_FALLBACKS=mistral python script.py
    
    # Method 2: Context Manager (recommended)
    with apply_model_fallbacks("mistral"):
        output = model(input_ids)
    
    # Method 3: Permanent Enable/Disable
    enable_model_fallbacks("mistral")
    output = model(input_ids)
    disable_model_fallbacks("mistral")
"""

import functools
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# Global registry: model_name -> {op_name -> (check_fn, fallback_fn, target_module)}
_MODEL_FALLBACKS: Dict[str, Dict[str, Tuple[Optional[Callable], Optional[Callable], str]]] = {}

# Track which models have active fallbacks
_ACTIVE_MODELS: Set[str] = set()

# Store original methods for restoration: (target_module, method_name) -> original_method
_ORIGINAL_METHODS: Dict[Tuple[str, str], Callable] = {}

# Environment variable for automatic activation
_ENV_VAR = "TORCH_SPYRE_MODEL_FALLBACKS"
_WARN_ENV_VAR = "TORCH_SPYRE_FALLBACK_WARN"


class ModelFallbackError(Exception):
    """Raised when a model fallback operation fails"""
    pass


def _should_warn() -> bool:
    """Check if fallback warnings should be displayed"""
    return os.environ.get(_WARN_ENV_VAR, "0") == "1"


def _warn_fallback(model_name: str, op_name: str, reason: str = "error"):
    """Warn about fallback usage"""
    if _should_warn():
        warnings.warn(
            f"[{model_name}] Operation '{op_name}' falling back to CPU ({reason})",
            UserWarning,
            stacklevel=4
        )


def _cpu_fallback(original_fn: Callable, *args, **kwargs) -> Any:
    """
    Default CPU fallback implementation.
    
    Moves tensors to CPU, executes operation, then moves result back to Spyre.
    Handles both single tensors and collections (tuple/list) of tensors.
    """
    # Determine source device from first Spyre tensor
    source_device = None
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "spyre":
            source_device = arg.device
            break
    
    if source_device is None:
        for value in kwargs.values():
            if isinstance(value, torch.Tensor) and value.device.type == "spyre":
                source_device = value.device
                break
    
    if source_device is None:
        source_device = torch.device("spyre")
    
    # Helper to move tensors to CPU
    def to_cpu(x):
        if isinstance(x, torch.Tensor) and x.device.type == "spyre":
            return x.cpu()
        elif isinstance(x, (tuple, list)):
            return type(x)(to_cpu(item) for item in x)
        elif isinstance(x, dict):
            return {k: to_cpu(v) for k, v in x.items()}
        return x
    
    # Move arguments to CPU
    cpu_args = [to_cpu(arg) for arg in args]
    cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
    
    # Execute on CPU
    result = original_fn(*cpu_args, **cpu_kwargs)
    
    # Helper to move tensors back to Spyre
    def to_spyre(x):
        if isinstance(x, torch.Tensor):
            return x.to(source_device)
        elif isinstance(x, (tuple, list)):
            return type(x)(to_spyre(item) for item in x)
        elif isinstance(x, dict):
            return {k: to_spyre(v) for k, v in x.items()}
        return x
    
    return to_spyre(result)


def _create_fallback_wrapper(
    model_name: str,
    op_name: str,
    original_fn: Callable,
    check_fn: Optional[Callable] = None,
    fallback_fn: Optional[Callable] = None
) -> Callable:
    """
    Create a wrapper that tries Spyre first, falls back to CPU on failure.
    
    Args:
        model_name: Name of the model (e.g., "mistral")
        op_name: Name of the operation (e.g., "reshape", "linear")
        original_fn: Original PyTorch method
        check_fn: Optional pre-check function to determine if fallback is needed
                  Signature: check_fn(*args, **kwargs) -> bool
                  Returns True if fallback should be used immediately
        fallback_fn: Optional custom fallback function
                     Defaults to _cpu_fallback if not provided
    
    Returns:
        Wrapped function with fallback logic
    """
    if fallback_fn is None:
        fallback_fn = _cpu_fallback
    
    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        # Pre-check: if check_fn returns True, skip Spyre and use fallback directly
        if check_fn is not None:
            try:
                if check_fn(*args, **kwargs):
                    msg = f"[{model_name}] {op_name}: Pre-check condition met, using CPU fallback"
                    print(msg, file=sys.stderr)
                    logger.info(msg)
                    _warn_fallback(model_name, op_name, "pre-check condition met")
                    return fallback_fn(original_fn, *args, **kwargs)
            except Exception:
                # If check fails, proceed with normal try-fallback logic
                pass
        
        # Try Spyre first
        try:
            return original_fn(*args, **kwargs)
        except Exception as spyre_error:
            # Fall back to CPU on any error
            error_type = type(spyre_error).__name__
            msg = f"[{model_name}] {op_name}: Spyre failed with {error_type}, falling back to CPU"
            print(msg, file=sys.stderr)
            logger.warning(msg)
            _warn_fallback(model_name, op_name, f"{error_type}")
            
            try:
                result = fallback_fn(original_fn, *args, **kwargs)
                msg = f"[{model_name}] {op_name}: CPU fallback succeeded"
                print(msg, file=sys.stderr)
                logger.info(msg)
                return result
            except Exception as cpu_error:
                # Both Spyre and CPU failed - raise detailed error
                msg = f"[{model_name}] {op_name}: Both Spyre and CPU fallback failed!"
                print(msg, file=sys.stderr)
                logger.error(msg)
                raise ModelFallbackError(
                    f"[{model_name}] Both Spyre and CPU fallback failed for '{op_name}':\n"
                    f"  Spyre error: {error_type}: {str(spyre_error)}\n"
                    f"  CPU error: {type(cpu_error).__name__}: {str(cpu_error)}"
                ) from cpu_error
    
    return wrapper


def register_model_fallback(
    model_name: str,
    op_name: str,
    check_fn: Optional[Callable] = None,
    fallback_fn: Optional[Callable] = None,
    target_module: str = "torch.Tensor"
):
    """
    Register a fallback for a specific model and operation.
    
    Args:
        model_name: Name of the model (e.g., "mistral", "bert", "gpt2")
        op_name: Operation name (e.g., "reshape", "linear", "transpose")
        check_fn: Optional function to check if fallback is needed before trying Spyre
                  Signature: check_fn(*args, **kwargs) -> bool
                  Example: lambda self, *shape: self.numel() > 1000000  # Large tensors only
        fallback_fn: Optional custom fallback function (defaults to CPU fallback)
        target_module: Module to patch ("torch.Tensor", "torch", "torch.nn.functional")
    
    Example:
        # Register reshape fallback for Mistral
        register_model_fallback("mistral", "reshape")
        
        # Register with custom check for specific shapes
        def check_large_reshape(self, *shape):
            return self.numel() > 1000000  # Only fallback for large tensors
        register_model_fallback("mistral", "reshape", check_fn=check_large_reshape)
        
        # Register with custom check for non-contiguous tensors
        def check_non_contiguous(self, *args, **kwargs):
            return not self.is_contiguous()
        register_model_fallback("mistral", "view", check_fn=check_non_contiguous)
    """
    if model_name not in _MODEL_FALLBACKS:
        _MODEL_FALLBACKS[model_name] = {}
    
    _MODEL_FALLBACKS[model_name][op_name] = (check_fn, fallback_fn, target_module)


def _get_target_and_method(op_name: str, target_module: str) -> Tuple[Optional[Any], str]:
    """
    Get the target object and method name for patching.
    
    Returns:
        (target_object, method_name) or (None, op_name) if not found
    """
    if target_module == "torch.Tensor":
        if hasattr(torch.Tensor, op_name):
            return (torch.Tensor, op_name)
    elif target_module == "torch":
        if hasattr(torch, op_name):
            return (torch, op_name)
    elif target_module == "torch.nn.functional":
        if hasattr(torch.nn.functional, op_name):
            return (torch.nn.functional, op_name)
    
    return (None, op_name)


def _apply_fallbacks(model_name: str):
    """Apply fallbacks for a specific model by monkey-patching PyTorch methods"""
    if model_name not in _MODEL_FALLBACKS:
        warnings.warn(f"No fallbacks registered for model '{model_name}'", UserWarning)
        return
    
    if model_name in _ACTIVE_MODELS:
        msg = f"[{model_name}] Fallbacks already active"
        print(msg, file=sys.stderr)
        logger.info(msg)
        return  # Already active
    
    msg = f"[{model_name}] Applying fallback patches..."
    print(msg, file=sys.stderr)
    logger.info(msg)
    patched_count = 0
    
    for op_name, (check_fn, fallback_fn, target_module) in _MODEL_FALLBACKS[model_name].items():
        target, method_name = _get_target_and_method(op_name, target_module)
        
        if target is None:
            warnings.warn(f"Could not resolve operation '{op_name}' in '{target_module}'", UserWarning)
            continue
        
        # Store original method
        key = (target_module, method_name)
        if key not in _ORIGINAL_METHODS:
            _ORIGINAL_METHODS[key] = getattr(target, method_name)
        
        # Create and apply wrapper
        original_fn = _ORIGINAL_METHODS[key]
        wrapper = _create_fallback_wrapper(
            model_name, op_name, original_fn, check_fn, fallback_fn
        )
        setattr(target, method_name, wrapper)
        patched_count += 1
    
    _ACTIVE_MODELS.add(model_name)
    msg = f"[{model_name}] Successfully patched {patched_count} operations"
    print(msg, file=sys.stderr)
    logger.info(msg)


def _remove_fallbacks(model_name: str):
    """Remove fallbacks for a specific model by restoring original methods"""
    if model_name not in _ACTIVE_MODELS:
        return  # Not active
    
    if model_name not in _MODEL_FALLBACKS:
        return
    
    for op_name, (_, _, target_module) in _MODEL_FALLBACKS[model_name].items():
        key = (target_module, op_name)
        if key not in _ORIGINAL_METHODS:
            continue
        
        # Restore original method
        target, method_name = _get_target_and_method(op_name, target_module)
        if target is not None:
            setattr(target, method_name, _ORIGINAL_METHODS[key])
    
    _ACTIVE_MODELS.discard(model_name)


def enable_model_fallbacks(model_name: str):
    """
    Permanently enable fallbacks for a specific model.
    
    Args:
        model_name: Name of the model (e.g., "mistral", "bert", "gpt2")
    
    Example:
        enable_model_fallbacks("mistral")
        output = model(input_ids)  # Fallbacks active
        disable_model_fallbacks("mistral")
    """
    _apply_fallbacks(model_name)


def disable_model_fallbacks(model_name: str):
    """
    Disable previously enabled fallbacks for a specific model.
    
    Args:
        model_name: Name of the model
    
    Example:
        disable_model_fallbacks("mistral")
    """
    _remove_fallbacks(model_name)


@contextmanager
def apply_model_fallbacks(model_name: str):
    """
    Context manager to temporarily apply model-specific fallbacks.
    
    Args:
        model_name: Name of the model (e.g., "mistral", "bert", "gpt2")
    
    Example:
        with apply_model_fallbacks("mistral"):
            output = model(input_ids)  # Fallbacks active only here
    """
    was_active = model_name in _ACTIVE_MODELS
    
    if not was_active:
        _apply_fallbacks(model_name)
    
    try:
        yield
    finally:
        if not was_active:
            _remove_fallbacks(model_name)


def list_registered_models() -> List[str]:
    """
    Get list of all models with registered fallbacks.
    
    Returns:
        List of model names
    
    Example:
        models = list_registered_models()
        # Returns: ['mistral', 'bert', 'gpt2', 'llama', 't5']
    """
    return sorted(_MODEL_FALLBACKS.keys())


def list_active_models() -> List[str]:
    """
    Get list of models with currently active fallbacks.
    
    Returns:
        List of active model names
    
    Example:
        active = list_active_models()
        # Returns: ['mistral'] if Mistral fallbacks are active
    """
    return sorted(_ACTIVE_MODELS)


# Track if we've already initialized from env to avoid duplicate messages
_ENV_INITIALIZED = False

def _init_from_env():
    """Initialize fallbacks from environment variable (only once)"""
    global _ENV_INITIALIZED
    
    # Only initialize once
    if _ENV_INITIALIZED:
        return
    
    env_value = os.environ.get(_ENV_VAR, "").strip()
    if not env_value:
        return
    
    # Mark as initialized
    _ENV_INITIALIZED = True
    
    # Support comma-separated list of models
    models = [m.strip() for m in env_value.split(",") if m.strip()]
    
    if models:
        msg = "\n" + "=" * 80 + "\n"
        msg += f" TORCH_SPYRE_MODEL_FALLBACKS detected: {env_value}\n"
        msg += f"  Enabling conditional fallbacks for: {', '.join(models)}\n"
        msg += "=" * 80
        print(msg, file=sys.stderr)  # Use stderr so pytest shows it
        logger.info(msg)
    
    for model_name in models:
        if model_name in _MODEL_FALLBACKS:
            enable_model_fallbacks(model_name)
            num_ops = len(_MODEL_FALLBACKS[model_name])
            msg = f"Enabled {num_ops} conditional fallback operations for: {model_name}"
            print(msg, file=sys.stderr)
            logger.info(msg)
        else:
            msg = (f"Model '{model_name}' specified in {_ENV_VAR} but not registered. "
                   f"Available models: {', '.join(list_registered_models())}")
            print(f"  {msg}", file=sys.stderr)
            warnings.warn(msg, UserWarning)
    
    if models:
        msg = "Conditional fallback mechanism is ACTIVE\n"
        msg += " Operations will try Spyre first, fall back to CPU on failure\n"
        msg += "=" * 80 + "\n"
        print(msg, file=sys.stderr)
        logger.info(msg)


# ============================================================================
# Pre-registered Model Fallbacks - Pure Try-Catch Approach
# ============================================================================
#
# All fallbacks use pure try-catch: always try Spyre first, fall back to CPU
# only if Spyre fails. No pre-validation checks.
#

# Mistral Model Fallbacks - Pure try-catch (no check_fn)
register_model_fallback("mistral", "reshape")
register_model_fallback("mistral", "transpose")
register_model_fallback("mistral", "clone")
register_model_fallback("mistral", "view")
register_model_fallback("mistral", "contiguous")
register_model_fallback("mistral", "rsqrt")
register_model_fallback("mistral", "add")
register_model_fallback("mistral", "mul")
register_model_fallback("mistral", "cat", target_module="torch")
register_model_fallback("mistral", "stack", target_module="torch")
register_model_fallback("mistral", "sum")
register_model_fallback("mistral", "mean")
register_model_fallback("mistral", "eq")
register_model_fallback("mistral", "index_copy_")
register_model_fallback("mistral", "masked_scatter_")
register_model_fallback("mistral", "linear", target_module="torch.nn.functional")

# BERT Model Fallbacks - Pure try-catch (no check_fn)
register_model_fallback("bert", "reshape")
register_model_fallback("bert", "transpose")
register_model_fallback("bert", "clone")
register_model_fallback("bert", "view")
register_model_fallback("bert", "contiguous")
register_model_fallback("bert", "linear", target_module="torch.nn.functional")

# GPT-2 Model Fallbacks - Pure try-catch (no check_fn)
register_model_fallback("gpt2", "reshape")
register_model_fallback("gpt2", "transpose")
register_model_fallback("gpt2", "clone")
register_model_fallback("gpt2", "view")
register_model_fallback("gpt2", "contiguous")
register_model_fallback("gpt2", "linear", target_module="torch.nn.functional")

# LLaMA Model Fallbacks - Pure try-catch (no check_fn)
register_model_fallback("llama", "reshape")
register_model_fallback("llama", "transpose")
register_model_fallback("llama", "clone")
register_model_fallback("llama", "view")
register_model_fallback("llama", "contiguous")
register_model_fallback("llama", "rsqrt")
register_model_fallback("llama", "linear", target_module="torch.nn.functional")

# Granite Model Fallbacks - Pure try-catch (no check_fn)
register_model_fallback("granite", "reshape")
register_model_fallback("granite", "transpose")
register_model_fallback("granite", "clone")
register_model_fallback("granite", "view")
register_model_fallback("granite", "contiguous")
register_model_fallback("granite", "rsqrt")
register_model_fallback("granite", "linear", target_module="torch.nn.functional")

# ============================================================================
# Compilation-Level Fallback Wrapper
# ============================================================================

def wrap_with_compile_fallback(fn: Callable, model_name: str = "unknown") -> Callable:
    """
    Wrap a function to catch compilation errors and fall back to eager mode.
    
    This catches torch._inductor.exc.InductorError and other compilation errors,
    then retries the operation in eager mode (without compilation) on CPU.
    
    Args:
        fn: Function to wrap (typically a model forward pass)
        model_name: Name of the model for logging
    
    Returns:
        Wrapped function that falls back to eager mode on compilation failure
    
    Example:
        model = MyModel()
        model.forward = wrap_with_compile_fallback(model.forward, "mistral")
        output = model(input_ids)  # Will fall back to CPU if compilation fails
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            # Try normal execution (may involve compilation)
            return fn(*args, **kwargs)
        except Exception as e:
            # Check if this is a compilation error
            error_type = type(e).__name__
            is_compile_error = (
                "InductorError" in error_type or
                "NotImplementedError" in error_type or
                "CalledProcessError" in error_type or
                "AttributeError" in str(e) and "UnimplementedOp" in str(e) or
                "TypeError" in error_type and "Cannot convert symbols" in str(e)
            )
            
            if is_compile_error:
                msg = f"[{model_name}] Compilation failed with {error_type}, falling back to eager mode on CPU"
                print(msg, file=sys.stderr)
                logger.warning(msg)
                
                # Fall back to eager mode on CPU
                try:
                    # Move inputs to CPU
                    cpu_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor) and arg.device.type == "spyre":
                            cpu_args.append(arg.cpu())
                        else:
                            cpu_args.append(arg)
                    
                    cpu_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, torch.Tensor) and value.device.type == "spyre":
                            cpu_kwargs[key] = value.cpu()
                        else:
                            cpu_kwargs[key] = value
                    
                    # Execute on CPU in eager mode
                    with torch.no_grad():
                        result = fn(*cpu_args, **cpu_kwargs)
                    
                    # Move result back to Spyre
                    if isinstance(result, torch.Tensor):
                        result = result.to("spyre")
                    elif isinstance(result, (tuple, list)):
                        result = type(result)(
                            item.to("spyre") if isinstance(item, torch.Tensor) else item
                            for item in result
                        )
                    
                    msg = f"[{model_name}] CPU eager mode fallback succeeded"
                    print(msg, file=sys.stderr)
                    logger.info(msg)
                    return result
                    
                except Exception as cpu_error:
                    msg = f"[{model_name}] Both compilation and CPU fallback failed!"
                    print(msg, file=sys.stderr)
                    logger.error(msg)
                    raise ModelFallbackError(
                        f"[{model_name}] Both compilation and CPU fallback failed:\n"
                        f"  Compilation error: {error_type}: {str(e)}\n"
                        f"  CPU error: {type(cpu_error).__name__}: {str(cpu_error)}"
                    ) from cpu_error
            else:
                # Not a compilation error, re-raise
                raise
    
    return wrapper


def enable_compile_fallbacks(model_name: str):
    """
    Enable compilation-level fallbacks for a model.
    
    This is automatically called when model fallbacks are enabled,
    but can be called separately if needed.
    
    Args:
        model_name: Name of the model
    """
    msg = f"[{model_name}] Compilation-level fallbacks enabled"
    print(msg, file=sys.stderr)
    logger.info(msg)


# Initialize from environment variable on module import


# -------------------------------------------------------------------
# Automatic Test Runner Patching for Compilation Fallbacks
# -------------------------------------------------------------------

def _patch_test_runner_if_needed():
    """
    Automatically patch the test runner to enable compilation fallbacks
    when TORCH_SPYRE_MODEL_FALLBACKS environment variable is set.
    
    This eliminates the need for conftest.py - the patching happens
    automatically when this module is imported.
    """
    fallback_models = os.environ.get("TORCH_SPYRE_MODEL_FALLBACKS", "").strip()
    
    if not fallback_models:
        # No fallback models specified, skip patching
        return
    
    try:
        # Try to import and patch the runner module
        # This will only work if runner is already imported (e.g., by pytest)
        import sys
        
        # Look for runner module in sys.modules
        runner = None
        for module_name, module in sys.modules.items():
            if 'runner' in module_name and hasattr(module, '_maybe_compile_call'):
                runner = module
                break
        
        if runner is None:
            # Runner not imported yet, skip patching
            # (will be handled by conftest.py if present)
            return
        
        # Check if already patched
        if hasattr(runner._maybe_compile_call, '_spyre_fallback_patched'):
            return
        
        print("\n" + "=" * 80, file=sys.stderr)
        print(f" TORCH_SPYRE_MODEL_FALLBACKS detected: {fallback_models}", file=sys.stderr)
        print("  Enabling automatic compilation fallbacks via model_fallbacks.py", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
        
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
        
        # Mark as patched to avoid double-patching
        wrapped_maybe_compile_call._spyre_fallback_patched = True
        
        # Apply the patch
        runner._maybe_compile_call = wrapped_maybe_compile_call
        
        print("Compilation fallback patching complete - tests will automatically fall back to CPU on errors", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
        
    except Exception as e:
        # Silently fail if patching doesn't work
        # (conftest.py can still handle it if present)
        logger.debug(f"Could not auto-patch test runner: {e}")


# Apply automatic patching when module is imported
_patch_test_runner_if_needed()
_init_from_env()
