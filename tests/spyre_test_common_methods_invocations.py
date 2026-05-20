"""
Utility functions for the Spyre test framework.

This module contains helper functions for creating module input generators
and other test utilities that are used by TorchTestBase.
"""

from typing import Any, Callable

import torch


def create_module_inputs_func_from_yaml(item: Any) -> Callable:
    """Create a module_inputs_func from a YAML module item.

    This function creates a generator that follows PyTorch's upstream signature:
    module_inputs_func(module_info, device, dtype, requires_grad, training, **kwargs) -> list[ModuleInput]

    Args:
        item: A module item from YAML edits.modules.include with build_module_input method

    Returns:
        A callable that generates ModuleInput instances
    """

    def module_inputs_func(
        module_info, device, dtype, requires_grad, training, **kwargs
    ):
        """Generated from YAML edits.modules.include"""
        try:
            from torch.testing._internal.common_modules import ModuleInput
            from torch.testing._internal.common_utils import FunctionInput
        except ImportError:
            return []

        # Use the build_module_input method if available
        if hasattr(item, "build_module_input"):
            test_device = torch.device(device) if isinstance(device, str) else device
            seed = kwargs.get("seed")

            # Check if forward_inputs is a list (multiple invocations)
            forward_inputs = getattr(item, "forward_inputs", None)
            if isinstance(forward_inputs, list) and len(forward_inputs) > 1:
                # Multiple invocations - create ModuleInput for each
                module_inputs = []
                for i, forward_spec in enumerate(forward_inputs):
                    # Build constructor inputs (same for all invocations)
                    constructor_spec = getattr(item, "constructor_inputs", None)
                    if constructor_spec and hasattr(constructor_spec, "build_cpu_args"):
                        constructor_args = constructor_spec.build_cpu_args(
                            seed=seed,
                            op_name=item.name,
                            test_device=test_device,
                        )
                        constructor_kwargs = constructor_spec.resolved_kwargs(
                            test_device=test_device
                        )
                    else:
                        constructor_args = []
                        constructor_kwargs = {}
                    constructor_input = FunctionInput(
                        *constructor_args, **constructor_kwargs
                    )

                    # Build forward inputs for this invocation
                    if hasattr(forward_spec, "build_cpu_args"):
                        forward_args = forward_spec.build_cpu_args(
                            seed=(None if seed is None else seed + 10000 + i * 1000),
                            op_name=item.name,
                            test_device=test_device,
                        )
                        forward_kwargs = forward_spec.resolved_kwargs(
                            test_device=test_device
                        )
                    else:
                        forward_args = []
                        forward_kwargs = {}
                    forward_input = FunctionInput(*forward_args, **forward_kwargs)

                    module_inputs.append(
                        ModuleInput(
                            constructor_input=constructor_input,
                            forward_input=forward_input,
                        )
                    )
                return module_inputs
            else:
                # Single invocation - use existing method
                return [
                    item.build_module_input(
                        seed=seed,
                        test_device=test_device,
                        FunctionInput=FunctionInput,
                        ModuleInput=ModuleInput,
                    )
                ]

        # Fallback: empty inputs
        return [
            ModuleInput(
                constructor_input=FunctionInput(),
                forward_input=FunctionInput(),
            )
        ]

    return module_inputs_func


def create_module_inputs_func_from_config(config: Any) -> Callable:
    """Create a module_inputs_func from a SupportedModuleConfig.

    This function creates a generator that uses inline input specs from the config
    and follows PyTorch's upstream signature:
    module_inputs_func(module_info, device, dtype, requires_grad, training, **kwargs) -> list[ModuleInput]

    Args:
        config: A SupportedModuleConfig instance with constructor_inputs and forward_inputs

    Returns:
        A callable that generates ModuleInput instances
    """

    def module_inputs_func(
        module_info, device, dtype, requires_grad, training, **kwargs
    ):
        """Generated module input function from YAML config."""
        try:
            from torch.testing._internal.common_modules import ModuleInput
            from torch.testing._internal.common_utils import FunctionInput
        except ImportError:
            return []

        # Get seed from global config
        seed = kwargs.get("seed")
        test_device = torch.device(device) if isinstance(device, str) else device

        # Build constructor inputs
        constructor_spec = config.constructor_inputs
        if constructor_spec and constructor_spec.has_inputs():
            constructor_args = constructor_spec.build_cpu_args(
                seed=seed,
                op_name=module_info.name,
                test_device=test_device,
            )
            constructor_kwargs = constructor_spec.resolved_kwargs(
                test_device=test_device
            )
        else:
            constructor_args = []
            constructor_kwargs = {}

        constructor_input = FunctionInput(*constructor_args, **constructor_kwargs)

        # Build forward inputs - handle both single and list formats
        forward_spec = config.forward_inputs
        module_inputs = []

        if forward_spec:
            # Handle list of forward_inputs (multiple invocations)
            if isinstance(forward_spec, list):
                for i, spec in enumerate(forward_spec):
                    if spec.has_inputs():
                        forward_args = spec.build_cpu_args(
                            seed=(None if seed is None else seed + 10000 + i * 1000),
                            op_name=module_info.name,
                            test_device=test_device,
                        )
                        forward_kwargs = spec.resolved_kwargs(test_device=test_device)
                    else:
                        forward_args = []
                        forward_kwargs = {}

                    forward_input = FunctionInput(*forward_args, **forward_kwargs)
                    module_inputs.append(
                        ModuleInput(
                            constructor_input=constructor_input,
                            forward_input=forward_input,
                        )
                    )
            # Handle single forward_inputs (backward compatibility)
            else:
                if forward_spec.has_inputs():
                    forward_args = forward_spec.build_cpu_args(
                        seed=(None if seed is None else seed + 10000),
                        op_name=module_info.name,
                        test_device=test_device,
                    )
                    forward_kwargs = forward_spec.resolved_kwargs(
                        test_device=test_device
                    )
                else:
                    forward_args = []
                    forward_kwargs = {}

                forward_input = FunctionInput(*forward_args, **forward_kwargs)
                module_inputs.append(
                    ModuleInput(
                        constructor_input=constructor_input,
                        forward_input=forward_input,
                    )
                )
        else:
            # No forward inputs specified
            forward_input = FunctionInput()
            module_inputs.append(
                ModuleInput(
                    constructor_input=constructor_input,
                    forward_input=forward_input,
                )
            )

        return module_inputs

    return module_inputs_func
