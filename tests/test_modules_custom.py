"""
Custom module tests for torch-spyre.

This file contains additional test methods for modules defined in YAML configs:
- test_eager_vs_compile: Compare eager and compile mode outputs (CPU vs Spyre eager vs Spyre compiled)
- test_layout_stride: Validate real YAML-specified SpyreTensorLayouts and strides (CPU vs Spyre)

All tests use pytree for robust handling of nested input/output structures and test only
real model configurations from YAML without artificial modifications.
"""

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_modules import modules, module_db
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import tree_map


def _extract_all_tensors(output):
    """Extract all tensors from potentially nested output structure.

    Uses pytree to handle nested structures (tuples, lists, dicts, etc.)
    and returns all tensors found. This ensures complete validation of
    module outputs including hidden states, KV caches, attention weights, etc.

    Args:
        output: Module output (can be tensor, tuple, list, dict, or nested structure)

    Returns:
        List of all tensors found in the output structure, in traversal order.
        Returns empty list if no tensors exist.
    """
    tensors = []

    def collect_tensors(x):
        if isinstance(x, torch.Tensor):
            tensors.append(x)
        return x

    tree_map(collect_tensors, output)
    return tensors


class TestModuleCustom(TestCase):
    """Custom test cases for module validation with different execution modes and layouts."""

    @modules(module_db)
    def test_eager_vs_compile(self, device, dtype, module_info, training):
        """Test eager mode vs compile mode, comparing CPU and Spyre outputs.

        This test:
        1. Runs module in eager mode on CPU
        2. Runs module in eager mode on Spyre
        3. Runs module in compile mode on Spyre
        4. Compares outputs between eager CPU, eager Spyre, and compile Spyre
        """
        module_inputs = module_info.module_inputs_func(
            module_info, device=device, dtype=dtype, requires_grad=False, training=False
        )

        for module_input in module_inputs:
            # Create module on CPU (eager)
            module_cpu = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            )
            module_cpu.eval()

            # Create module on device (eager)
            module_device_eager = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            ).to(device)
            module_device_eager.eval()

            # Copy weights from CPU to device
            module_device_eager.load_state_dict(module_cpu.state_dict())

            # Create compiled version
            module_device_compile_base = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            ).to(device)
            module_device_compile_base.eval()
            module_device_compile_base.load_state_dict(module_cpu.state_dict())
            module_device_compile = torch.compile(module_device_compile_base)

            # Prepare inputs
            args_cpu = module_input.forward_input.args
            kwargs_cpu = module_input.forward_input.kwargs

            # Move inputs to device using pytree to handle nested structures
            args_device = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, args_cpu
            )
            kwargs_device = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, kwargs_cpu
            )

            # Run forward passes
            with torch.no_grad():
                output_cpu = module_cpu(*args_cpu, **kwargs_cpu)
                output_device_eager = module_device_eager(*args_device, **kwargs_device)
                output_device_compile = module_device_compile(
                    *args_device, **kwargs_device
                )

            # Extract all tensors from outputs using pytree
            cpu_tensors = _extract_all_tensors(output_cpu)
            device_eager_tensors = _extract_all_tensors(output_device_eager)
            device_compile_tensors = _extract_all_tensors(output_device_compile)

            # Verify all outputs have the same number of tensors
            if not (
                len(cpu_tensors)
                == len(device_eager_tensors)
                == len(device_compile_tensors)
            ):
                self.fail(
                    f"{module_info.name}: Output tensor count mismatch - "
                    f"CPU: {len(cpu_tensors)}, Spyre eager: {len(device_eager_tensors)}, "
                    f"Spyre compile: {len(device_compile_tensors)}"
                )

            # Compare all tensors (hidden states, KV cache, attention weights, etc.)
            for i, (cpu_t, eager_t, compile_t) in enumerate(
                zip(cpu_tensors, device_eager_tensors, device_compile_tensors)
            ):
                # Compare CPU eager vs Spyre eager
                self.assertEqual(
                    cpu_t,
                    eager_t.cpu(),
                    msg=f"{module_info.name}: CPU eager vs Spyre eager mismatch (tensor {i})",
                )

                # Compare Spyre eager vs Spyre compile
                self.assertEqual(
                    eager_t,
                    compile_t,
                    msg=f"{module_info.name}: Spyre eager vs Spyre compile mismatch (tensor {i})",
                )

    @modules(module_db)
    def test_layout_stride(self, device, dtype, module_info, training):
        """Test module with real YAML-specified layouts and strides.

        Validates modules work correctly with actual SpyreTensorLayouts from YAML config.
        Compares CPU vs device outputs for correctness.
        """
        module_inputs = module_info.module_inputs_func(
            module_info, device=device, dtype=dtype, requires_grad=False, training=False
        )

        for module_input in module_inputs:
            # Create module on CPU
            module_cpu = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            )
            module_cpu.eval()

            # Create module on device
            module_device = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            ).to(device)
            module_device.eval()

            # Copy weights from CPU to device
            module_device.load_state_dict(module_cpu.state_dict())

            # Prepare inputs
            args_cpu = module_input.forward_input.args
            kwargs_cpu = module_input.forward_input.kwargs

            # Move inputs to device using pytree to handle nested structures
            args_device = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, args_cpu
            )
            kwargs_device = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, kwargs_cpu
            )

            # Run forward passes
            with torch.no_grad():
                output_cpu = module_cpu(*args_cpu, **kwargs_cpu)
                output_device = module_device(*args_device, **kwargs_device)

            # Extract all tensors from outputs using pytree
            cpu_tensors = _extract_all_tensors(output_cpu)
            device_tensors = _extract_all_tensors(output_device)

            # Verify both outputs have the same number of tensors
            if len(cpu_tensors) != len(device_tensors):
                self.fail(
                    f"{module_info.name}: Output tensor count mismatch - "
                    f"CPU: {len(cpu_tensors)}, Spyre: {len(device_tensors)}"
                )

            # Compare all tensors (hidden states, KV cache, attention weights, etc.)
            for i, (cpu_t, device_t) in enumerate(zip(cpu_tensors, device_tensors)):
                self.assertEqual(
                    cpu_t,
                    device_t.cpu(),
                    msg=f"{module_info.name}: layout/stride mismatch on real inputs (tensor {i})",
                )


# Instantiate tests for all device types
instantiate_device_type_tests(TestModuleCustom, globals())


if __name__ == "__main__":
    run_tests()
