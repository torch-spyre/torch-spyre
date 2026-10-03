import torch

# This provides:
# 1. Proper schema registration
# 2. Automatic fake kernel registration
# 3. Better integration with torch.compile
# 4. C++ implementation via TORCH_LIBRARY_IMPL in spyre_distributed.cpp
# This file only registers the abstract (fake/meta) kernels needed by torch.compile
# for shape inference during tracing.

# ──────────────────────────────────────────────────────────────────────────────
# Register _c10d_functional::reduce if not already provided by PyTorch.
# Older PyTorch versions only ship all_reduce/broadcast/wait_tensor in the
# functional c10d namespace. We add `reduce` so torch.compile can trace it.
# ──────────────────────────────────────────────────────────────────────────────
if not hasattr(torch.ops._c10d_functional, "reduce"):
    _c10d_lib = torch.library.Library("_c10d_functional", "FRAGMENT")
    _c10d_lib.define("reduce(Tensor self, int dst, str reduceOp, str tag) -> Tensor")

    @torch.library.register_fake("_c10d_functional::reduce")
    def _reduce_fake(
        self: torch.Tensor, dst: int, reduceOp: str, tag: str
    ) -> torch.Tensor:
        return torch.empty_like(self)


# The spyre::* distributed operators are only compiled into the C++ extension
# when torch-spyre is built with USE_SPYRE_CCL=1 (see setup.py: spyre_distributed.cpp
# is added to the sources only when use_spyre_ccl is true). Registering a fake
# kernel for an operator that was not compiled in raises "operator spyre::... does
# not exist", which would break `import torch` on any USE_SPYRE_CCL=0 build. Only
# register the fakes when the real operators are actually present.
if torch._C._dispatch_has_kernel("spyre::broadcast_run"):

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

    @torch.library.register_fake("spyre::wait_work")
    def _(x: torch.Tensor) -> torch.Tensor:
        """Fake implementation — pass through the tensor."""
        return x

    # Plan ops use CompositeImplicitAutograd (registered in C++ via combined
    # m.def+impl), so they need no separate fake kernel here.

    # ------------------------------------------------------------------
    # Runtime run ops — shape inference matches the legacy async ops.
    # ------------------------------------------------------------------

    @torch.library.register_fake("spyre::broadcast_run")
    def _(x: torch.Tensor, plan_handle: int, src_rank: int) -> torch.Tensor:
        return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)

    @torch.library.register_fake("spyre::allreduce_run")
    def _(x: torch.Tensor, plan_handle: int) -> torch.Tensor:
        return x

    @torch.library.register_fake("spyre::allgather_run")
    def _(x: torch.Tensor, plan_handle: int, group_size: int) -> torch.Tensor:
        output_size = list(x.shape)
        output_size[0] *= group_size
        return torch.empty(output_size, dtype=x.dtype, device=x.device)

    @torch.library.register_fake("spyre::reducescatter_run")
    def _(x: torch.Tensor, plan_handle: int, group_size: int) -> torch.Tensor:
        output_size = list(x.shape)
        output_size[0] //= group_size
        return torch.empty(output_size, dtype=x.dtype, device=x.device)

    @torch.library.register_fake("spyre::all_gather_async")
    def _(
        x: torch.Tensor, group_size: int = 1, group_name: str = "default"
    ) -> torch.Tensor:
        """Fake implementation for shape inference during compilation."""
        output_size = list(x.shape)
        output_size[0] *= group_size
        return torch.empty(output_size, dtype=x.dtype, device=x.device)

    @torch.library.register_fake("spyre::reduce_scatter_async")
    def _(
        x: torch.Tensor,
        reduce_op: str = "sum",
        group_size: int = 1,
        group_name: str = "default",
    ) -> torch.Tensor:
        output_size = list(x.shape)
        output_size[0] //= group_size
        return torch.empty(output_size, dtype=x.dtype, device=x.device)
