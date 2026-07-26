import torch

# This provides:
# 1. Proper schema registration
# 2. Automatic fake kernel registration
# 3. Better integration with torch.compile
# 4. C++ implementation via TORCH_LIBRARY_IMPL in spyre_distributed.cpp
# This file only registers the abstract (fake/meta) kernels needed by torch.compile
# for shape inference during tracing.


@torch.library.register_fake("spyre::broadcast_async")
def _(x: torch.Tensor, src_rank: int = 0, group_name: str = "default") -> torch.Tensor:
    """Fake implementation for shape inference during compilation.

    Broadcast preserves shape, dtype, and stride.
    """
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)


@torch.library.register_fake("spyre::wait_work")
def _(x: torch.Tensor) -> torch.Tensor:
    """Fake implementation — pass through the tensor."""
    return x
