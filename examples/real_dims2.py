import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Reduction
from torch_spyre._inductor import propagate_real_dims as prd
from torch_spyre._inductor.propagate_real_dims import (
    declare_real_dim,
    annotate_real_dims,
    _compute_real_layout,
    _get_buffer,
)
from torch_spyre._inductor.pass_utils import host_coordinates
from torch_spyre._inductor.views import compute_coordinates, matching_dim
from torch_spyre._inductor.constants import BATCH_MATMUL_OP


def _lone_sym(coord: sympy.Expr) -> sympy.Symbol:
    return next(iter(coord.free_symbols))


def compute_input_real_dims(dep: MemoryDep) -> dict:
    buf_real_dims = _get_buffer(dep).real_dims
    real_size, real_stride = _compute_real_layout(buf_real_dims)
    coords = compute_coordinates(real_size, real_stride, dep.ranges, dep.index)
    result = {}
    for i, coord in enumerate(coords):
        if coord.free_symbols:
            result[_lone_sym(coord)] = buf_real_dims[i]
    return result


def get_reduction_dim(dep: MemoryDep, out_coords: list) -> sympy.Symbol:
    buf_real_dims = _get_buffer(dep).real_dims
    real_size, real_stride = _compute_real_layout(buf_real_dims)
    in_coords = compute_coordinates(real_size, real_stride, dep.ranges, dep.index)
    reduction_coord = next(
        c for c in in_coords
        if c.free_symbols and matching_dim(out_coords, c) is None
    )
    return _lone_sym(reduction_coord)


def _matmul_real_dims(op: ComputedBuffer, inputs: list) -> None:
    # Part 1: compute input real dims (generic, op-agnostic)
    rdims_0 = compute_input_real_dims(inputs[0])
    rdims_1 = compute_input_real_dims(inputs[1])

    # Part 2: matmul dimension mapping
    rdims = {**rdims_0, **rdims_1}
    output_dep = next(iter(op.get_read_writes().writes))
    out_coords = host_coordinates(op.get_layout(), output_dep)
    reduction_var = get_reduction_dim(inputs[0], out_coords)

    op.real_dims = [rdims[_lone_sym(c)] for c in out_coords if c.free_symbols]
    op.real_ranges = op.real_dims + [rdims[reduction_var]]


def _compute_real_dims(op, inputs):
    if isinstance(op.data, Reduction) and op.data.reduction_type == BATCH_MATMUL_OP:
        return _matmul_real_dims(op, inputs)
    op.real_ranges = _get_buffer(inputs[0]).real_dims
    op.real_dims = _get_buffer(inputs[0]).real_dims


prd._compute_real_dims = _compute_real_dims

# --- driver (same as real_dims.py) ---

torch.manual_seed(0xAFFE)

x = torch.rand(64 * 128, dtype=torch.float16).reshape(64, 128)
y = torch.rand(128, 256, dtype=torch.float16)
z = torch.rand(8, 8, 256, dtype=torch.float16).reshape(64, 256)


def f(x, y, z):
    return x @ y + z


r = f(x, y, z)
x_dev = x.to("spyre")
y_dev = y.to("spyre")
z_dev = z.to("spyre")

declare_real_dim("a", 64)
declare_real_dim("b", 128)
declare_real_dim("c", 256)

annotate_real_dims(x_dev, ["a", "b"])
annotate_real_dims(y_dev, ["b", "c"])
annotate_real_dims(z_dev, ["a", "c"])

z = torch.compile(f)(x_dev, y_dev, z_dev).cpu()

print(r)
print(z)
print(torch.abs(r - z).amax())
