import torch

# This provides:
# 1. Proper schema registration
# 2. Automatic fake kernel registration
# 3. Better integration with torch.compile
# 4. C++ implementation via TORCH_LIBRARY_IMPL in spyre_distributed.cpp
# This file only registers the abstract (fake/meta) kernels needed by torch.compile
# for shape inference during tracing.

# Schema defined in Python, not C++: the C++ side only registers PrivateUse1
# impls (spyre_distributed.cpp) and needs this schema to exist first.
# torch_spyre/__init__.py imports this module before torch_spyre._C to
# guarantee that order. Kept as a module-level var so it isn't GC'd.
_spyre_distributed_lib = torch.library.Library("spyre", "FRAGMENT")
_spyre_distributed_lib.define(
    "broadcast_async(Tensor input, int src_rank, str group_name) -> Tensor"
)
_spyre_distributed_lib.define("wait_work(Tensor(a!) tensor) -> Tensor(a)")


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
