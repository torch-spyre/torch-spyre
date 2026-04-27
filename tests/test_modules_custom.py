"""
Custom module tests for torch-spyre.

This file contains additional test methods for modules defined in YAML configs:
- test_eager_vs_compile: Compare eager and compile mode outputs (CPU vs Spyre)
- test_layout: Test different memory layouts (contiguous, non-contiguous, channels_last)
- test_stride: Test different stride patterns
"""

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_modules import modules, module_db
from torch.testing._internal.common_utils import TestCase


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

            # Move inputs to device
            args_device = tuple(
                arg.to(device) if isinstance(arg, torch.Tensor) else arg
                for arg in args_cpu
            )
            kwargs_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs_cpu.items()
            }

            # Run forward passes
            with torch.no_grad():
                output_cpu = module_cpu(*args_cpu, **kwargs_cpu)
                output_device_eager = module_device_eager(*args_device, **kwargs_device)
                output_device_compile = module_device_compile(
                    *args_device, **kwargs_device
                )

            # Extract tensor from output (handle tuples/dicts)
            def extract_tensor(output):
                if isinstance(output, torch.Tensor):
                    return output
                elif isinstance(output, (tuple, list)):
                    return output[0] if len(output) > 0 else None
                elif isinstance(output, dict):
                    return next(iter(output.values())) if output else None
                return output

            output_cpu_tensor = extract_tensor(output_cpu)
            output_device_eager_tensor = extract_tensor(output_device_eager)
            output_device_compile_tensor = extract_tensor(output_device_compile)

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
    def test_layout(self, device, dtype, module_info, training):
        """Test module with different memory layouts.

        Tests:
        - Contiguous tensors
        - Non-contiguous tensors (via transpose)
        - Channels-last format (for Conv modules)
        """
        module_inputs = module_info.module_inputs_func(
            module_info, device=device, dtype=dtype, requires_grad=False, training=False
        )

        for module_input in module_inputs:
            module = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            ).to(device)
            module.eval()

            args = tuple(
                arg.to(device) if isinstance(arg, torch.Tensor) else arg
                for arg in module_input.forward_input.args
            )
            kwargs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in module_input.forward_input.kwargs.items()
            }

            # Test 1: Contiguous input (default)
            with torch.no_grad():
                output_contiguous = module(*args, **kwargs)

            # Verify output exists
            self.assertIsNotNone(
                output_contiguous, f"{module_info.name}: Contiguous layout test failed"
            )

            # Test 2: Non-contiguous input (if applicable)
            # Try to make first tensor arg non-contiguous via transpose
            if args and isinstance(args[0], torch.Tensor) and args[0].ndim >= 2:
                args_non_contig = list(args)
                # Transpose and transpose back to get non-contiguous tensor with same shape
                args_non_contig[0] = args[0].t().contiguous().t()
                self.assertFalse(
                    args_non_contig[0].is_contiguous(),
                    f"{module_info.name}: Failed to create non-contiguous tensor",
                )

                with torch.no_grad():
                    output_non_contig = module(*args_non_contig, **kwargs)

                self.assertIsNotNone(
                    output_non_contig,
                    f"{module_info.name}: Non-contiguous layout test failed",
                )

            # Test 3: Channels-last (for 4D tensors in Conv-like modules)
            if args and isinstance(args[0], torch.Tensor) and args[0].ndim == 4:
                args_channels_last = list(args)
                args_channels_last[0] = args[0].to(memory_format=torch.channels_last)
                self.assertTrue(
                    args_channels_last[0].is_contiguous(
                        memory_format=torch.channels_last
                    ),
                    f"{module_info.name}: Failed to create channels_last tensor",
                )

                with torch.no_grad():
                    output_channels_last = module(*args_channels_last, **kwargs)

                self.assertIsNotNone(
                    output_channels_last,
                    f"{module_info.name}: Channels-last layout test failed",
                )

    @modules(module_db)
    def test_stride(self, device, dtype, module_info, training):
        """Test module with different stride patterns.

        Tests:
        - Default strides
        - Custom strides (via slicing)
        - Broadcasted tensors (stride 0 in some dimensions)
        """
        module_inputs = module_info.module_inputs_func(
            module_info, device=device, dtype=dtype, requires_grad=False, training=False
        )

        for module_input in module_inputs:
            module = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            ).to(device)
            module.eval()

            args = tuple(
                arg.to(device) if isinstance(arg, torch.Tensor) else arg
                for arg in module_input.forward_input.args
            )
            kwargs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in module_input.forward_input.kwargs.items()
            }

            # Test 1: Default strides
            with torch.no_grad():
                output_default = module(*args, **kwargs)

            self.assertIsNotNone(
                output_default, f"{module_info.name}: Default stride test failed"
            )

            # Test 2: Custom strides via slicing (if tensor is large enough)
            if args and isinstance(args[0], torch.Tensor):
                tensor = args[0]
                # Try to create a strided view by taking every other element
                if tensor.shape[-1] > 2:  # Need at least 3 elements to slice
                    args_strided = list(args)
                    # Take every other element along last dimension
                    args_strided[0] = tensor[..., ::2]

                    # Verify stride changed
                    original_stride = tensor.stride()
                    new_stride = args_strided[0].stride()
                    if original_stride != new_stride:
                        with torch.no_grad():
                            output_strided = module(*args_strided, **kwargs)

                        self.assertIsNotNone(
                            output_strided,
                            f"{module_info.name}: Custom stride test failed",
                        )

            # Test 3: Broadcasted tensor (stride 0)
            if args and isinstance(args[0], torch.Tensor) and args[0].ndim >= 2:
                tensor = args[0]
                # Create a broadcasted version by expanding first dimension
                if tensor.shape[0] > 1:
                    args_broadcast = list(args)
                    # Take first element and expand
                    single = tensor[0:1]  # Keep dimensions
                    expanded = single.expand(tensor.shape[0], *tensor.shape[1:])
                    args_broadcast[0] = expanded

                    # Verify stride 0 in first dimension
                    if expanded.stride()[0] == 0:
                        with torch.no_grad():
                            output_broadcast = module(*args_broadcast, **kwargs)

                        self.assertIsNotNone(
                            output_broadcast,
                            f"{module_info.name}: Broadcast stride test failed",
                        )


# Instantiate tests for all device types
instantiate_device_type_tests(TestModuleCustom, globals())


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
