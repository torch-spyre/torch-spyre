from typing import Any

from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonScheduling

from torch_spyre._inductor.spyre_triton_kernel import SpyreTritonKernel


class SpyreTritonScheduling(TritonScheduling):
    """
    Spyre-specific Triton scheduling that uses SpyreTritonKernel.
    """

    def create_kernel_choices(  # type: ignore[override]
        self,
        kernel_features: SIMDKernelFeatures,
        kernel_args: list[Any],
        kernel_kwargs: dict[str, Any],
    ) -> list[Any]:
        self.kernel_type = SpyreTritonKernel  # type: ignore[assignment]
        return super().create_kernel_choices(
            kernel_features, kernel_args, kernel_kwargs
        )
