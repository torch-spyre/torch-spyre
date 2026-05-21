import torch
import torch_spyre._inductor.passes as passes
import torch_spyre._inductor.propagate_real_dims as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_real_dim = prd.declare_real_dim
annotate_real_dims = prd.annotate_real_dims

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

B, S, H, D = 2, 128, 32, 128  # granite dims: x: [2, 128, 32, 128], y: [4096, 4096]
# B, S, H, D = 1, 128, 2, 128  # smaller dims: x: [1, 128, 2, 128], y: [256, 256]


def fn(x_base, y):
    # Reproduce clone(permute(...)) chain from granite:
    # a previous op outputs 4D [B, S, H, D], then a linear projects it.
    # Inductor sees the 4D buffer and accesses it with a flat reduction
    # index over the last two dims, producing tiled host coords.
    x = x_base.permute(0, 1, 3, 2)  # [2, 128, 32, 128], non-contiguous
    x = x.clone()  # contiguous, stride=[524288, 4096, 128, 1]
    # linear: flatten last two dims implicitly via matmul
    return torch.matmul(x.reshape(B, S, H * D), y)


x_cpu = torch.randn(B, S, D, H, dtype=torch.float16)
y_cpu = torch.randn(H * D, H * D, dtype=torch.float16)

cpu_result = fn(x_cpu, y_cpu)
print(f"CPU result shape: {cpu_result.shape}")

x_dev = x_cpu.to(DEVICE)
y_dev = y_cpu.to(DEVICE)

declare_real_dim("B", B)
declare_real_dim("S", S)
declare_real_dim("H", H)
declare_real_dim("D", D)

annotate_real_dims(x_dev, ["B", "S", "D", "H"])  # [2, 128, 128, 32]
annotate_real_dims(y_dev, ["H", "D", "H", "D"])  # [4096, 4096] = [H*D, H*D]

compiled = torch.compile(fn)
result = compiled(x_dev, y_dev).cpu()
print(f"AIU result shape: {result.shape}")

torch.testing.assert_close(
    result,
    cpu_result,
    atol=0.5,
    rtol=0.1,
    msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
)
print("ANSWER CORRECT!")
