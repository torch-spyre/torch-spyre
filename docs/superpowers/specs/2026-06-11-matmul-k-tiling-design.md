# Matmul K-Dimension Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable coarse-tiling of matmul/bmm in the reduction (K) dimension by adding a matmul-specific carve-out to the Stage 1 stick-dim guard in `coarse_tile.py`.

**Architecture:** Each tile of a K-tiled matmul computes a full-sized partial `[M, N]` (or `[B, M, N]`) output — no sparse/partial-stick output occurs. The fill-initialize + per-tile combine pattern from Stage 1 applies directly: identity = 0, combine = addition. Three surgical edits to `coarse_tile.py`; no changes to `padding.py` or `superdsc.py` are required.

**Tech Stack:** Python, `torch._inductor` IR (`ComputedBuffer`, `Reduction`), existing coarse-tiling infrastructure (`_validate_reduction_tiling`, `_reduction_identity_value`, `_insert_combine_op`)

---

## Background

The Stage 1 coarse-tiling pass (merged on `tile-reduction` branch) supports reduction-dim tiling for scalar reductions (`sum`, `max`, `min`, etc.). It blocks matmul K-tiling via `_validate_reduction_tiling` → `_reduction_tiling_is_on_stick_dim`, which returns `True` for matmul because K is the within-stick dimension for operand `x`.

The guard was written conservatively. For scalar reductions, tiling the stick dim produces partial-stick outputs — a genuinely hard problem deferred to Stage 2. For matmul, tiling K does not have this problem: each tile's output is a full `[M, N]` matrix regardless of how K is divided. The partial outputs are summed to produce the final result.

## Data Flow

```
outer HBM:  accum[M, N]          ← filled with 0 before loop (outside loop)
loop T times:
  x_tile[M, K/T] @ y_tile[K/T, N] → partial[M, N]   (per_tile_fixed scratch)
  accum += partial                                     (combine op, inside loop)
outside consumers read accum[M, N]
```

For bmm: `[B, M, K/T] @ [B, K/T, N] → partial[B, M, N]`, accum = `[B, M, N]`.

## Padding Pass Interaction

`insert_bmm_padding` runs before `_maybe_coarse_tile`. It pads `y`'s K rows to a stick boundary and rebuilds the matmul `inner_fn`. `_divide_reduction_ranges` divides the IR's logical K. At SDSC codegen time, `_extend_matmul_k_to_padded` pads K/T to `round_up(K/T, stick_size)` for each tile — identical to how untiled matmuls are handled today. No changes to `padding.py` or `superdsc.py` are needed.

## Files Modified

- Modify: `torch_spyre/_inductor/coarse_tile.py` (3 edits)
- Modify: `tests/inductor/test_coarse_tile_e2e.py` (convert 2 tests, add new class)

---

### Task 1: Add matmul carve-out to `_validate_reduction_tiling`

**Files:**
- Modify: `torch_spyre/_inductor/coarse_tile.py` (~line 392)

- [ ] **Step 1: Read the current guard**

Read `coarse_tile.py` lines 337–400 to confirm the exact location of the stick-dim check:

```python
for red_dim_idx in red_dims:
    if _reduction_tiling_is_on_stick_dim(op, red_dim_idx):
        raise RuntimeError(...)
```

- [ ] **Step 2: Add the matmul carve-out**

Replace the inner loop body with:

```python
        for red_dim_idx in red_dims:
            if (
                op.data.reduction_type != BATCH_MATMUL_OP
                and _reduction_tiling_is_on_stick_dim(op, red_dim_idx)
            ):
                raise RuntimeError(
                    f"coarse_tile: op {op.get_name()!r} level {i} tiles "
                    f"reduction dim {red_dim_idx} which is the stick dimension "
                    "of the primary input (stick-dim reduction tiling is not "
                    "yet implemented — Stage 2)."
                )
```

Add `BATCH_MATMUL_OP` to the imports at the top of the function (it is already imported at the module level via `from .constants import BATCH_MATMUL_OP`).

Also update the docstring of `_validate_reduction_tiling` to document the matmul exception:

```
    Supported (Stage 1):
      - A single level that tiles only a non-stick reduction dim.
      - A single level that tiles the K (reduction) dim of a BATCH_MATMUL_OP —
        K is the stick dim for operand x, but each tile's output is a full
        [M, N] matrix so no partial-stick sparsity occurs.
```

- [ ] **Step 3: Run unit tests to verify guard change is sane**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py -x -q
```

Expected: same pass count as before (guard change does not affect unit tests, which use mocked ops).

---

### Task 2: Add `BATCH_MATMUL_OP` to `_reduction_identity_value` and `_insert_combine_op`

**Files:**
- Modify: `torch_spyre/_inductor/coarse_tile.py` (~lines 1218 and 740)

- [ ] **Step 1: Update `_reduction_identity_value`**

Find:

```python
    if reduction_type in ("sum", "xor_sum", "any"):
        return 0
```

Replace with:

```python
    if reduction_type in ("sum", "xor_sum", "any", BATCH_MATMUL_OP):
        return 0
```

The identity for matmul K-tiling is 0: the accumulator starts at zero and partial products are added to it.

- [ ] **Step 2: Update `_insert_combine_op`**

Find:

```python
        if reduction_type == "sum":
            return vops.add(accum, partial)
```

Replace with:

```python
        if reduction_type in ("sum", BATCH_MATMUL_OP):
            return vops.add(accum, partial)
```

The partial matrix products are summed element-wise into the accumulator.

- [ ] **Step 3: Add `BATCH_MATMUL_OP` import to the function scope (if needed)**

`BATCH_MATMUL_OP` is already imported at module level. No change needed.

- [ ] **Step 4: Run unit tests**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py -x -q
```

Expected: same pass count.

---

### Task 3: Convert existing matmul K-tiling rejection tests to correctness tests

**Files:**
- Modify: `tests/inductor/test_coarse_tile_e2e.py` (~lines 999–1041)

Context: `TestCoarseTileReductionE2E` contains two tests that currently assert K-tiling raises a Stage 2 error:
- `test_hint_tiled_reduction_matmul_loopspec`
- `test_hint_tiled_reduction_matmul_rejects`

After the carve-out lands, these tests will fail (no exception is raised). Convert them to check correct output and LoopSpec respectively.

- [ ] **Step 1: Convert `test_hint_tiled_reduction_matmul_loopspec` to a LoopSpec assertion**

Replace the `assertRaisesRegex` with an assertion that the compiled source contains a `LoopSpec` with count 4. Use `run_and_get_code` with `mock_patch` (the pattern used by all other loopspec tests in this file):

```python
    def test_hint_tiled_reduction_matmul_loopspec(self):
        """torch.matmul tiled over K produces a LoopSpec with count 4."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return a @ b

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for K-tiled matmul")
        self.assertIn("sympify('4')", src, "Expected loop count 4")
```

- [ ] **Step 2: Convert `test_hint_tiled_reduction_matmul_rejects` to a correctness test**

Replace with:

```python
    @config.patch({"lx_planning": False})
    def test_hint_tiled_reduction_matmul_correct(self):
        """torch.matmul tiled over K (4 tiles) produces correct results."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return a @ b

        compare_with_cpu(fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05)
```

- [ ] **Step 3: Run the converted tests**

```bash
python3 -m pytest tests/inductor/test_coarse_tile_e2e.py::TestCoarseTileReductionE2E::test_hint_tiled_reduction_matmul_loopspec tests/inductor/test_coarse_tile_e2e.py::TestCoarseTileReductionE2E::test_hint_tiled_reduction_matmul_correct -x -v
```

Expected: both pass.

---

### Task 4: Add `TestCoarseTileMatmulKTilingE2E` test class

**Files:**
- Modify: `tests/inductor/test_coarse_tile_e2e.py` (append before `if __name__ == "__main__":`)

- [ ] **Step 1: Add the test class**

```python
class TestCoarseTileMatmulKTilingE2E(InductorTestCase):
    """Correctness and LoopSpec tests for matmul/bmm tiled over the K (reduction) dimension.

    K=512 tiled by 4 → 128 per tile (two sticks); shapes chosen so K/T is
    stick-aligned without padding, keeping the tests deterministic.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    @config.patch({"lx_planning": False})
    def test_mm_k_tiled_correct(self):
        """2D mm [M,K] @ [K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.mm(a, b)

        compare_with_cpu(fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05)

    @config.patch({"lx_planning": False})
    def test_bmm_k_tiled_correct(self):
        """3D bmm [B,M,K] @ [B,K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 8, 64, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(B, K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["B", "M", "K"])
            _name_tensor_dims(b, ["B", "K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.bmm(a, b)

        compare_with_cpu(fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05)

    @config.patch({"lx_planning": False})
    def test_bmm_3d2d_k_tiled_correct(self):
        """3D×2D bmm [B,M,K] @ [K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 8, 64, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["B", "M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.matmul(a, b)

        compare_with_cpu(fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05)

    @config.patch({"lx_planning": False})
    def test_mm_k_tiled_loopspec(self):
        """K-tiled mm produces a LoopSpec with count 4 in the generated source."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.mm(a, b)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for K-tiled mm")
        self.assertIn("sympify('4')", src, "Expected loop count 4")
```

- [ ] **Step 2: Run all new tests**

```bash
python3 -m pytest tests/inductor/test_coarse_tile_e2e.py::TestCoarseTileMatmulKTilingE2E -x -v
```

Expected: all 4 pass.

---

### Task 5: Full regression run and commit

**Files:** None (verification only, then commit)

- [ ] **Step 1: Run all coarse-tile tests**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py tests/inductor/test_coarse_tile_e2e.py -x -q
```

Expected: all pass.

- [ ] **Step 2: Run pre-commit checks**

```bash
pre-commit run --all-files
```

- [ ] **Step 3: Commit**

```bash
git add torch_spyre/_inductor/coarse_tile.py tests/inductor/test_coarse_tile_e2e.py
git commit -s -m "feat(coarse_tile): enable K-dimension tiling for matmul/bmm

Add a BATCH_MATMUL_OP carve-out to _validate_reduction_tiling so that
K-tiled matmuls bypass the stick-dim guard.  K is the stick dim for
operand x, but each tile's output is a full [M,N] matrix — no
partial-stick sparsity — so the fill+combine pattern applies directly.

_reduction_identity_value and _insert_combine_op are extended to
recognise BATCH_MATMUL_OP: identity = 0, combine = add.

Tests: convert two existing rejection tests to correctness tests;
add TestCoarseTileMatmulKTilingE2E covering mm, bmm 3D×3D, bmm 3D×2D,
and a LoopSpec assertion."
```
