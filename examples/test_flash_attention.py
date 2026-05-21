import math

import torch
import torch_spyre._inductor.passes as passes
import torch_spyre._inductor.propagate_real_dims as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_real_dim = prd.declare_real_dim
annotate_real_dims = prd.annotate_real_dims


def flash(Q, K, V, block_size):
    B, H, L, D = Q.shape

    output = torch.zeros_like(Q)
    M = torch.full((B, H, L), float("-inf"), device=Q.device, dtype=torch.float16)
    denominator = torch.zeros((B, H, L), device=Q.device, dtype=torch.float16)

    scale = 1.0 / math.sqrt(D)

    for start in range(0, L, block_size):
        end = start + block_size

        K_block = K[:, :, start:end, :]
        V_block = V[:, :, start:end, :]
        K_block_T = K_block.transpose(-1, -2).contiguous()  # B, H, D, Block

        scores = torch.matmul(Q, K_block_T) * scale  # B, H, L, Block
        scores = scores.transpose(-1, -2).contiguous()  # avoid stick reduction
        block_max = torch.amax(scores, dim=-2)
        max_running = torch.maximum(M, block_max)

        exp_scores = torch.exp(scores - max_running.unsqueeze(-2))  # B, H, Block, L

        correction = torch.exp(M - max_running)

        denominator = denominator * correction + exp_scores.sum(dim=-2)
        output = output * correction.unsqueeze(-1) + torch.bmm(
            exp_scores.transpose(-1, -2).flatten(0, 1), V_block.flatten(0, 1)
        ).unflatten(0, (B, H))

        M = max_running

    output = output / denominator.unsqueeze(-1)
    return output


compiled_flash = torch.compile(flash, dynamic=False)

if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, L, D = 1, 8, 256, 64
    block_size = 128

    Q = torch.randn(B, H, L, D, dtype=torch.float16)
    K = torch.randn(B, H, L, D, dtype=torch.float16)
    V = torch.randn(B, H, L, D, dtype=torch.float16)

    declare_real_dim("B", B)
    declare_real_dim("H", H)
    declare_real_dim("L", L)
    declare_real_dim("NB", L // block_size)
    declare_real_dim("BS", block_size)
    declare_real_dim("D", D)

    q_spyre = Q.to("spyre")
    k_spyre = K.to("spyre")
    v_spyre = V.to("spyre")

    annotate_real_dims(q_spyre, ["B", "H", "L", "D"])
    annotate_real_dims(k_spyre, ["B", "H", "NB", "BS", "D"])
    annotate_real_dims(v_spyre, ["B", "H", "NB", "BS", "D"])

    spyre_out = compiled_flash(q_spyre, k_spyre, v_spyre, block_size).cpu()
    cpu_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    cpu_delta = torch.abs(spyre_out - cpu_out).max()

    print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
