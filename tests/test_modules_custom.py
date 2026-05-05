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
from torch.testing._internal.common_utils import TestCase
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import run_tests


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

            # Extract first tensor from output using pytree
            def extract_first_tensor(output):
                """Extract first tensor from potentially nested output structure."""
                tensors = []

                def collect_tensors(x):
                    if isinstance(x, torch.Tensor):
                        tensors.append(x)
                    return x

                tree_map(collect_tensors, output)
                return tensors[0] if tensors else None

            output_cpu_tensor = extract_first_tensor(output_cpu)
            output_device_eager_tensor = extract_first_tensor(output_device_eager)
            output_device_compile_tensor = extract_first_tensor(output_device_compile)

            if (
                output_cpu_tensor is not None
                and isinstance(output_cpu_tensor, torch.Tensor)
                and output_device_eager_tensor is not None
                and isinstance(output_device_eager_tensor, torch.Tensor)
                and output_device_compile_tensor is not None
                and isinstance(output_device_compile_tensor, torch.Tensor)
            ):
                # Compare CPU eager vs Spyre eager
                self.assertEqual(
                    output_cpu_tensor,
                    output_device_eager_tensor.cpu(),
                    msg=f"{module_info.name}: CPU eager vs Spyre eager mismatch",
                )

                # Compare Spyre eager vs Spyre compile
                self.assertEqual(
                    output_device_eager_tensor,
                    output_device_compile_tensor,
                    msg=f"{module_info.name}: Spyre eager vs Spyre compile mismatch",
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

            # Extract first tensor from output using pytree
            def extract_first_tensor(output):
                """Extract first tensor from potentially nested output structure."""
                tensors = []

                def collect_tensors(x):
                    if isinstance(x, torch.Tensor):
                        tensors.append(x)
                    return x

                tree_map(collect_tensors, output)
                return tensors[0] if tensors else None

            cpu_tensor = extract_first_tensor(output_cpu)
            device_tensor = extract_first_tensor(output_device)

            if (
                cpu_tensor is not None
                and isinstance(cpu_tensor, torch.Tensor)
                and device_tensor is not None
                and isinstance(device_tensor, torch.Tensor)
            ):
                # Compare CPU vs device outputs
                self.assertEqual(
                    cpu_tensor,
                    device_tensor.cpu(),
                    msg=f"{module_info.name}: layout/stride mismatch on real inputs",
                )


# Instantiate tests for all device types
instantiate_device_type_tests(TestModuleCustom, globals())


if __name__ == "__main__":
    run_tests()
