import torch

# This provides:
# 1. Proper schema registration
# 2. Automatic fake kernel registration
# 3. Better integration with torch.compile
# 4. C++ implementation via TORCH_LIBRARY_IMPL in spyre_distributed.cpp
# This file only registers the abstract (fake/meta) kernels needed by torch.compile
# for shape inference during tracing.

# The spyre::* distributed operators are only compiled into the C++ extension
# when torch-spyre is built with USE_SPYRE_CCL=1 (see setup.py: spyre_distributed.cpp
# is added to the sources only when use_spyre_ccl is true). Registering a fake
# kernel for an operator that was not compiled in raises "operator spyre::... does
# not exist", which would break `import torch` on any USE_SPYRE_CCL=0 build. Only
# register the fakes when the real operators are actually present.
if torch._C._dispatch_has_kernel("spyre::broadcast_async"):

    @torch.library.register_fake("spyre::broadcast_async")
    def _(
        x: torch.Tensor, src_rank: int = 0, group_name: str = "default"
    ) -> torch.Tensor:
        """Fake implementation for shape inference during compilation.

        Broadcast preserves shape, dtype, and stride.
        """
        return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)

    @torch.library.register_fake("spyre::all_reduce_async")
    def _(
        x: torch.Tensor, reduce_op: str = "sum", group_name: str = "default"
    ) -> torch.Tensor:
        """In-place op — returns the same tensor (mutated on device)."""
        return x

    @torch.library.register_fake("spyre::wait_work")
    def _(x: torch.Tensor) -> torch.Tensor:
        """Fake implementation — pass through the tensor."""
        return x
