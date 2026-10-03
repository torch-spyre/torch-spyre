"""Minimal regression test for index_copy_ on the Spyre backend.

Covers two shapes:
  rows=1  -- dim_size==1 path (Bug 1: write was silently elided before the fix)
  rows=2  -- normal scatter path (must not be broken by the fix)

Run with:
    SENCORES=1 python3 index_copy_test.py
"""

import torch

torch.manual_seed(3)


def store(out, index, src):
    out.index_copy_(0, index, src)
    return out


for rows in (1, 2):
    # ------------------------------------------------------------------ inputs
    out = torch.zeros(rows, 8, 128, dtype=torch.float16)
    src = torch.randn(rows, 8, 128, dtype=torch.float16)
    idx = torch.arange(rows, dtype=torch.int64)

    # ------------------------------------------------------------------ CPU reference
    ref = store(out.clone(), idx.clone(), src.clone())

    # ------------------------------------------------------------------ Spyre compiled
    torch._dynamo.reset()
    result = torch.compile(store, dynamic=False)(
        out.clone().to("spyre"),
        idx.to("spyre"),
        src.to("spyre"),
    ).cpu()

    # ------------------------------------------------------------------ sanity: result must not be all-zeros
    # Before the fix, rows=1 caused the write to be silently elided so the
    # output stayed all-zeros even when src was non-zero.
    assert result.abs().amax().item() > 0, (
        f"rows={rows}: output is all-zeros — write was silently elided (Bug 1 regression)"
    )

    # ------------------------------------------------------------------ compare against CPU reference
    diff = torch.abs(ref.float() - result.float()).amax().item()
    print(
        f"rows={rows}  shape={tuple(result.shape)}  max_abs_diff={diff:.4g}"
    )

    # Use PyTorch's default fp16 tolerances (atol=1e-3, rtol=1e-5).
    # index_copy_ is a pure copy but values pass through the Spyre device
    # which operates at fp16 precision, so exact bit equality is too strict.
    torch.testing.assert_close(
        result,
        ref,
        equal_nan=True,
        msg=lambda msg: f"rows={rows}: compiled Spyre <-> CPU mismatch\n\n{msg}\n",
    )

    print(f"rows={rows}: PASSED")
