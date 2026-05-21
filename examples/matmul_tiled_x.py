import torch
import torch_spyre._inductor.passes as passes
import torch_spyre._inductor.propagate_real_dims2 as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_real_dim = prd.declare_real_dim
annotate_real_dims = prd.annotate_real_dims

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

B, S, H, D = 2, 128, 32, 128  # granite dims: x: [2, 128, 32, 128], y: [4096, 4096]

x_cpu = torch.randn(B, S, H, D, dtype=torch.float16) * 0.01  # [2, 128, 32, 128]
y_cpu = torch.randn(H * D, H * D, dtype=torch.float16) * 0.01


def fn(x, y):
    # x_base is already [B, S, H, D] contiguous. The clone produces a
    # 4D buffer with stride=[S*H*D, H*D, D, 1]; inductor sees it as 4D
    # and accesses it via a single flat reduction index over H*D, giving
    # host_coords=[d0, d1, floor(r/D), Mod(r, D)] — the tiled-coord case.
    #
    # Inductor loop vars (batch matmul: 3 output dims + 1 reduction):
    #   d0: range=2    — B (batch)
    #   d1: range=128  — S (seq len, rows of output)
    #   d2: range=4096 — H*D (cols of output)
    #   d3: range=4096 — H*D (reduction dim), tiled as [H, D] in x's host coords
    x = x.clone()
    return torch.matmul(x.reshape(B, S, H * D), y)


cpu_result = fn(x_cpu, y_cpu)
print(f"CPU result shape: {cpu_result.shape}")

x_dev = x_cpu.to(DEVICE)
y_dev = y_cpu.to(DEVICE)

declare_real_dim("B", B)
declare_real_dim("S", S)
declare_real_dim("H", H)
declare_real_dim("D", D)

annotate_real_dims(x_dev, ["B", "S", "H", "D"])   # [2, 128, 32, 128]
annotate_real_dims(y_dev, ["H", "D", "H", "D"])   # [4096, 4096] = [H*D, H*D]

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
