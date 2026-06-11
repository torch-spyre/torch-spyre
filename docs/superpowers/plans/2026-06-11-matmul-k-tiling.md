# Matmul K-Dimension Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable coarse-tiling of matmul/bmm in the K (reduction) dimension by adding a matmul-specific carve-out to the Stage 1 stick-dim guard in `coarse_tile.py`.

**Architecture:** The existing fill-initialize + per-tile combine pattern handles matmul K-tiling directly: each tile computes a full `[M, N]` partial matrix product, and partial products are summed into an HBM accumulator. Three surgical edits to `coarse_tile.py` (import, guard carve-out, identity/combine support) plus test changes.

**Tech Stack:** Python, `torch._inductor` IR (`ComputedBuffer`, `Reduction`), `coarse_tile.py` infrastructure.

---

## Files Changed

| File | Change |
|---|---|
| `torch_spyre/_inductor/coarse_tile.py` | Add `BATCH_MATMUL_OP` import; carve-out in `_validate_reduction_tiling`; extend `_reduction_identity_value` and `_insert_combine_op` |
| `tests/inductor/test_coarse_tile_e2e.py` | Convert 2 rejection tests to correctness/loopspec tests; add `TestCoarseTileMatmulKTilingE2E` class |

---

### Task 1: Add `BATCH_MATMUL_OP` import to `coarse_tile.py`

**Files:**
- Modify: `torch_spyre/_inductor/coarse_tile.py:68-72`

- [ ] **Step 1: Add the import**

Find the existing imports block (around line 68):

```python
from .logging_utils import get_inductor_logger
from .loop_info import CoarseTileInfo
from .propagate_hints import get_op_hints
from .pass_utils import op_out_coords
```

Replace with:

```python
from .constants import BATCH_MATMUL_OP
from .logging_utils import get_inductor_logger
from .loop_info import CoarseTileInfo
from .propagate_hints import get_op_hints
from .pass_utils import op_out_coords
```

- [ ] **Step 2: Verify the import is correct**

```bash
python3 -c "from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling; print('ok')"
```

Expected output: `ok`

---

### Task 2: Add matmul carve-out to `_validate_reduction_tiling`

**Files:**
- Modify: `torch_spyre/_inductor/coarse_tile.py` (~line 337)

The function `_validate_reduction_tiling` currently raises a `RuntimeError` for any reduction dim that is the stick dim of the primary input. For `BATCH_MATMUL_OP`, K is the stick dim for operand `x` but tiling K is safe: each tile's output is a full `[M, N]` matrix with no partial-stick sparsity.

- [ ] **Step 1: Write a failing unit test first**

In `tests/inductor/test_coarse_tiling.py`, find `class TestValidateReductionTiling` (around line 2000). The class has a `_make_op(loop_tiled_dims, loop_tiled_reduction_dims)` helper that creates a mock `ComputedBuffer` with `reduction_type = "sum"`. Add a new test at the end of the class that creates a matmul variant:

```python
def test_batchmatmul_k_tiling_allowed(self):
    """BATCH_MATMUL_OP tiling on the stick (K) dim is allowed — no Stage 2 error."""
    from torch._inductor.ir import ComputedBuffer, Reduction
    from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP

    data = MagicMock(spec=Reduction)
    data.ranges = [Integer(64), Integer(32)]   # [M, N]
    data.reduction_ranges = [Integer(512)]     # [K]
    data.reduction_type = BATCH_MATMUL_OP
    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.get_name.return_value = "test_matmul"
    op.loop_info = CoarseTileInfo(
        loop_group_id=(0,),
        loop_count=[Integer(4)],
        loop_tiled_dims=[[]],
        loop_tiled_reduction_dims=[[0]],
    )
    # Must not raise: BATCH_MATMUL_OP is exempt from the stick-dim guard.
    _validate_reduction_tiling(op)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py -k "test_batchmatmul_k_tiling_allowed" -v
```

Expected: FAIL — the current guard raises `RuntimeError` for stick-dim reduction tiling.

- [ ] **Step 3: Add the carve-out**

In `coarse_tile.py`, find the inner loop in `_validate_reduction_tiling` (around line 392):

```python
        for red_dim_idx in red_dims:
            if _reduction_tiling_is_on_stick_dim(op, red_dim_idx):
                raise RuntimeError(
                    f"coarse_tile: op {op.get_name()!r} level {i} tiles "
                    f"reduction dim {red_dim_idx} which is the stick dimension "
                    "of the primary input (stick-dim reduction tiling is not "
                    "yet implemented — Stage 2)."
                )
```

Replace with:

```python
        for red_dim_idx in red_dims:
            if (
                data.reduction_type != BATCH_MATMUL_OP
                and _reduction_tiling_is_on_stick_dim(op, red_dim_idx)
            ):
                raise RuntimeError(
                    f"coarse_tile: op {op.get_name()!r} level {i} tiles "
                    f"reduction dim {red_dim_idx} which is the stick dimension "
                    "of the primary input (stick-dim reduction tiling is not "
                    "yet implemented — Stage 2)."
                )
```

Also update the docstring of `_validate_reduction_tiling` — replace the `Supported (Stage 1):` paragraph with:

```python
    """Raise RuntimeError for Reduction tiling configurations not yet implemented.

    Supported (Stage 1):
      - A single level that tiles only a non-stick reduction dim.
      - A single level that tiles the K (reduction) dim of a BATCH_MATMUL_OP.
        K is the stick dim for operand x, but each tile's output is a full
        [M, N] matrix so no partial-stick sparsity occurs.

    Deferred to Stage 2 (raises):
      - Reduction tiling on the stick dimension (except BATCH_MATMUL_OP above).
      - Mixed output+reduction tiling at the same nesting level.
      - Multiple nesting levels where both output-dim and reduction-dim levels
        appear (e.g. outer tiles output dim, inner tiles reduction dim).
      - Multiple reduction range indices tiled at one level.
    """
```

- [ ] **Step 4: Run the new test to confirm it passes**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py -k "test_batchmatmul_k_tiling_allowed" -v
```

Expected: PASS

- [ ] **Step 5: Run all coarse-tiling unit tests to check for regressions**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py -x -q
```

Expected: same count as before + 1 new pass.

---

### Task 3: Extend `_reduction_identity_value` and `_insert_combine_op` for `BATCH_MATMUL_OP`

**Files:**
- Modify: `torch_spyre/_inductor/coarse_tile.py` (~lines 740 and 1218)

The identity for matmul K-tiling is 0 (the accumulator starts empty and partial products are added). The combine operation is element-wise addition.

- [ ] **Step 1: Write a failing unit test**

In `tests/inductor/test_coarse_tiling.py`, find `class TestReductionIdentityValues` (around line 2767). The class has a `_identity(reduction_type)` helper. Add at the end of the class:

```python
def test_batchmatmul(self):
    """BATCH_MATMUL_OP identity value is 0 — partial products are summed."""
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP
    self.assertEqual(self._identity(BATCH_MATMUL_OP), 0)
```

- [ ] **Step 2: Run the failing test**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py -k "TestReductionIdentityValues and test_batchmatmul" -v
```

Expected: FAIL — `_reduction_identity_value` raises `RuntimeError` for `"batchmatmul"`.

- [ ] **Step 3: Update `_reduction_identity_value`**

Find (around line 1218):

```python
    if reduction_type in ("sum", "xor_sum", "any"):
        return 0
```

Replace with:

```python
    if reduction_type in ("sum", "xor_sum", "any", BATCH_MATMUL_OP):
        return 0
```

- [ ] **Step 4: Update `_insert_combine_op`**

Find (around line 740):

```python
        if reduction_type == "sum":
            return vops.add(accum, partial)
```

Replace with:

```python
        if reduction_type in ("sum", BATCH_MATMUL_OP):
            return vops.add(accum, partial)
```

- [ ] **Step 5: Run the new unit test**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py -k "TestReductionIdentityValues and test_batchmatmul" -v
```

Expected: PASS

- [ ] **Step 6: Run all coarse-tiling unit tests**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py -x -q
```

Expected: all pass.

- [ ] **Step 7: Commit the three `coarse_tile.py` changes**

```bash
git add torch_spyre/_inductor/coarse_tile.py tests/inductor/test_coarse_tiling.py
git commit -s -m "feat(coarse_tile): enable K-dimension tiling for BATCH_MATMUL_OP

Add BATCH_MATMUL_OP import and carve-out in _validate_reduction_tiling
so K-tiled matmuls bypass the stick-dim guard.  K is the stick dim for
operand x but each tile produces a full [M,N] output — no partial-stick
sparsity.  Extend _reduction_identity_value (identity=0) and
_insert_combine_op (combine=add) to handle BATCH_MATMUL_OP."
```

---

### Task 4: Convert existing matmul K-tiling rejection tests

**Files:**
- Modify: `tests/inductor/test_coarse_tile_e2e.py` (~lines 999–1041)

These two tests in `TestCoarseTileReductionE2E` currently assert that K-tiling raises Stage 2. After the carve-out they must be converted to positive tests.

- [ ] **Step 1: Confirm the tests currently fail (now that the guard is gone)**

```bash
python3 -m pytest tests/inductor/test_coarse_tile_e2e.py::TestCoarseTileReductionE2E::test_hint_tiled_reduction_matmul_loopspec tests/inductor/test_coarse_tile_e2e.py::TestCoarseTileReductionE2E::test_hint_tiled_reduction_matmul_rejects -v
```

Expected: FAIL — the `assertRaisesRegex` no longer sees an exception.

- [ ] **Step 2: Replace `test_hint_tiled_reduction_matmul_loopspec`**

Find the full existing method body (which wraps in `assertRaisesRegex`) and replace entirely with:

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

- [ ] **Step 3: Replace `test_hint_tiled_reduction_matmul_rejects`**

Find and replace entirely with:

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

- [ ] **Step 4: Run the converted tests**

```bash
python3 -m pytest "tests/inductor/test_coarse_tile_e2e.py::TestCoarseTileReductionE2E::test_hint_tiled_reduction_matmul_loopspec" "tests/inductor/test_coarse_tile_e2e.py::TestCoarseTileReductionE2E::test_hint_tiled_reduction_matmul_correct" -v
```

Expected: both PASS.

---

### Task 5: Add `TestCoarseTileMatmulKTilingE2E` test class

**Files:**
- Modify: `tests/inductor/test_coarse_tile_e2e.py` (append before the final `if __name__ == "__main__":` block)

- [ ] **Step 1: Add the test class**

Insert immediately before `if __name__ == "__main__":`:

```python
class TestCoarseTileMatmulKTilingE2E(InductorTestCase):
    """Correctness and LoopSpec tests for matmul/bmm tiled over the K (reduction) dimension.

    K=512 tiled by 4 gives 128 per tile (two sticks at fp16); shapes are chosen
    so K/T is stick-aligned without padding, keeping results deterministic.
    Use small weight scale (0.01) to keep fp16 accumulation error bounded.
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
        """3D×2D matmul [B,M,K] @ [K,N] tiled over K produces correct results."""
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
        """K-tiled mm produces a LoopSpec with count 4 in generated source."""
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

- [ ] **Step 2: Run the new test class**

```bash
python3 -m pytest tests/inductor/test_coarse_tile_e2e.py::TestCoarseTileMatmulKTilingE2E -v
```

Expected: all 4 pass.

---

### Task 6: Full regression run and final commit

**Files:** None (verification + commit)

- [ ] **Step 1: Run all coarse-tile tests**

```bash
python3 -m pytest tests/inductor/test_coarse_tiling.py tests/inductor/test_coarse_tile_e2e.py -x -q
```

Expected: all pass (previous count + new tests).

- [ ] **Step 2: Run pre-commit checks**

```bash
pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 3: Commit the test changes**

```bash
git add tests/inductor/test_coarse_tile_e2e.py
git commit -s -m "test(coarse_tile): add matmul K-tiling correctness and LoopSpec tests

Convert test_hint_tiled_reduction_matmul_loopspec and
test_hint_tiled_reduction_matmul_rejects from Stage-2-rejection tests
to positive tests now that K-tiling is supported.

Add TestCoarseTileMatmulKTilingE2E covering:
- mm 2D×2D correctness
- bmm 3D×3D correctness
- bmm 3D×2D (weight-sharing) correctness
- mm LoopSpec assertion (count=4)"
```
