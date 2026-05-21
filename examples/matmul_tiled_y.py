import torch
import torch_spyre._inductor.passes as passes
import torch_spyre._inductor.propagate_real_dims as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_tensor_dim = prd.declare_tensor_dim
name_tensor_dims = prd.name_tensor_dims

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

# Exact granite dimensions
B, H_KV, GQA, S, D = 2, 8, 4, 128, 128
H = H_KV * GQA  # 32 query heads


def fn(x, kv_cache):
    # x:        [2, 32, 128, 128]
    # kv_cache: [2, 128, 8, 128]  (as stored: batch, seq, kv_heads, head_dim)
    y = kv_cache.view(B, S, H_KV, D)  # [2, 128, 8, 128]
    y = y.permute(0, 2, 1, 3)  # [2, 8, 128, 128]
    y = y.unsqueeze(2)  # [2, 8, 1, 128, 128]
    y = y.expand(-1, -1, GQA, -1, -1)  # [2, 8, 4, 128, 128]
    y = y.clone()  # contiguous [2, 8, 4, 128, 128]
    # bmm: x [2*32, 128, 128] @ y.T [2*32, 128, 128]
    return torch.bmm(
        x.reshape(B * H, S, D),
        y.reshape(B * H, S, D).transpose(1, 2),
    )


x_cpu = torch.randn(B, H, S, D, dtype=torch.float16)
kv_cpu = torch.randn(B, S * H_KV, D, dtype=torch.float16)

cpu_result = fn(x_cpu, kv_cpu)
print(f"CPU result shape: {cpu_result.shape}")

x_dev = x_cpu.to(DEVICE)
kv_dev = kv_cpu.to(DEVICE)

declare_tensor_dim("B", B)
declare_tensor_dim("H_KV", H_KV)
declare_tensor_dim("GQA", GQA)
declare_tensor_dim("S", S)
declare_tensor_dim("D", D)

name_tensor_dims(
    x_dev, ["B", "H_KV", "GQA", "S", "D"]
)  # x shape [2,32,128,128]: H=H_KV*GQA tiled
name_tensor_dims(kv_dev, ["B", "H_KV", "S", "D"])  # kv shape [2,1024,128]: H_KV*S tiled

compiled = torch.compile(fn)
result = compiled(x_dev, kv_dev).cpu()
print(f"AIU result shape: {result.shape}")

torch.testing.assert_close(
    result,
    cpu_result,
    atol=0.5,
    rtol=0.1,
    msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
)
print("ANSWER CORRECT!")
