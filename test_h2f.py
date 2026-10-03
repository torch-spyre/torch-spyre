import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests", "inductor"))
from utils_inductor import cached_randn, _compile_and_run, DEVICE


def dl16_to_fp32_restore_cpu(x_staggered: torch.Tensor) -> torch.Tensor:
    FP16_STICK = 64
    n = x_staggered.shape[-1]
    assert n % FP16_STICK == 0, f"last dim {n} must be multiple of {FP16_STICK}"
    P = torch.zeros(n, n, dtype=torch.float32)
    for phys_j in range(n):
        win_base = (phys_j // FP16_STICK) * FP16_STICK
        local_phys = phys_j % FP16_STICK
        local_logical = (
            (local_phys // 32) * 4 + (local_phys % 4) + (local_phys % 32) // 4 * 8
        )
        P[win_base + local_logical, phys_j] = 1.0
    orig_shape = x_staggered.shape
    m = x_staggered.cpu().numel() // n
    result = torch.mm(x_staggered.cpu().float().reshape(m, n), P.t())
    return result.reshape(orig_shape)


# h2f softmax cases: fp16 input → fp32 output
# 4D non-aligned interior dims (e.g. (6,17,32,64) dim=2) are known unsupported
# due to mixed-EA clash on non-stick reduction dims.
CASES = [
    # 2D
    ("h2f 2D (24,128)     dim=0", 0, (24, 128)),
    ("h2f 2D (24,128)     dim=1", 1, (24, 128)),
    ("h2f 2D (512,1024)   dim=0", 0, (512, 1024)),
    ("h2f 2D (512,1024)   dim=1", 1, (512, 1024)),
    # 3D
    ("h2f 3D (4,24,128)   dim=0", 0, (4, 24, 128)),
    ("h2f 3D (4,24,128)   dim=1", 1, (4, 24, 128)),
    ("h2f 3D (4,24,128)   dim=2", 2, (4, 24, 128)),
    ("h2f 3D (256,64,128) dim=0", 0, (256, 64, 128)),
    ("h2f 3D (256,64,128) dim=1", 1, (256, 64, 128)),
    ("h2f 3D (256,64,128) dim=2", 2, (256, 64, 128)),
    # 4D aligned (all dims multiples of fp32 stick=32)
    ("h2f 4D (6,32,32,64) dim=0", 0, (6, 32, 32, 64)),
    ("h2f 4D (6,32,32,64) dim=1", 1, (6, 32, 32, 64)),
    ("h2f 4D (6,32,32,64) dim=2", 2, (6, 32, 32, 64)),
    ("h2f 4D (6,32,32,64) dim=3", 3, (6, 32, 32, 64)),
    # 4D non-aligned — known unsupported (mixed-EA on interior non-stick dim)
    ("h2f 4D (6,17,32,64) dim=0", 0, (6, 17, 32, 64)),
    ("h2f 4D (6,17,32,64) dim=1", 1, (6, 17, 32, 64)),
    ("h2f 4D (6,17,32,64) dim=2", 2, (6, 17, 32, 64)),
    ("h2f 4D (6,17,32,64) dim=3", 3, (6, 17, 32, 64)),
]

print("\n[h2f softmax: fp16 input → fp32 output]")
print("=" * 80)
print(f"  {'Test':<38s}  {'raw err':>10s}  {'restored':>10s}  Status")
print("-" * 80)

for label, dim, shape in CASES:
    x = cached_randn(shape, dtype=torch.float16)
    fn = lambda a, d=dim: torch.softmax(a, dim=d, dtype=torch.float32)
    cpu_ref = fn(x)
    try:
        spyre_out = _compile_and_run(fn, [x], DEVICE, compile=True)
        err_raw = torch.abs(spyre_out.float() - cpu_ref.float()).max().item()
        last_dim = spyre_out.shape[-1]
        if last_dim % 64 == 0:
            restored = dl16_to_fp32_restore_cpu(spyre_out.float())
            err_restored = torch.abs(restored - cpu_ref.float()).max().item()
            ok = err_restored <= 0.1
            print(
                f"  {label:<38s}  {err_raw:>10.4e}  {err_restored:>10.4e}  {'PASS' if ok else 'FAIL'}"
            )
        else:
            ok = err_raw <= 0.1
            print(
                f"  {label:<38s}  {err_raw:>10.4e}  {'n/a':>10s}  {'PASS' if ok else 'FAIL'}"
            )
    except Exception as e:
        msg = str(e).split("\n")[0]
        print(f"  {label:<38s}  {'FAILED':>10s}  {'':>10s}  {msg[:60]}")

print("=" * 80)
