import torch


@torch.library.register_kernel("aten::mm", ["spyre"])
def spyre__mm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2)


@torch.library.register_kernel("aten::mm.out", ["spyre"])
def spyre__mm_out(
    self: torch.Tensor, mat2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2, out=out)


@torch.library.register_kernel("aten::bmm", ["spyre"])
def spyre__bmm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    compiled_bmm = torch.compile(torch.bmm)
    return compiled_bmm(self, mat2)


@torch.library.register_kernel("aten::addmm.dtype_out", ["spyre"])
def spyre__addmm_dtype_out(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: torch.dtype,
    beta,
    alpha,
    out: torch.Tensor,
) -> torch.Tensor:
    compiled_addmm = torch.compile(torch.addmm)
    return compiled_addmm(self, mat1, mat2, out_dtype, beta, alpha, out=out)


@torch.library.register_kernel("aten::addmm.dtype", ["spyre"])
def spyre__addmm_dtype(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: torch.dtype,
    beta,
    alpha,
) -> torch.Tensor:
    compiled_addmm = torch.compile(torch.addmm)
    return compiled_addmm(self, mat1, mat2, out_dtype, beta, alpha)


@torch.library.register_kernel("aten::addmm", ["spyre"])
def spyre__addmm(
    self: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, beta, alpha
) -> torch.Tensor:
    compiled_addmm = torch.compile(torch.addmm)
    return compiled_addmm(self, mat1, mat2, beta, alpha)


@torch.library.register_kernel("aten::addmm.out", ["spyre"])
def spyre__addmm_out(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    beta,
    alpha,
    out: torch.Tensor,
) -> torch.Tensor:
    compiled_addmm = torch.compile(torch.addmm)
    return compiled_addmm(self, mat1, mat2, beta, alpha, out=out)
