import pytest
import torch

from utils_inductor import compare_with_cpu, _compile_and_run, DEVICE

PAGE_SIZE = 16
KV_HEADS = 8
HEAD_DIM = 128
GRANITE_KV_HEADS = 8
GRANITE_HEAD_DIM = 128


def _cache(num_pages, kv_heads=KV_HEADS, head_dim=HEAD_DIM, dtype=torch.bfloat16):
    """Return a zero-filled [num_pages, PAGE_SIZE, kv_heads, head_dim] cache."""
    return torch.zeros(num_pages, PAGE_SIZE, kv_heads, head_dim, dtype=dtype)


# ---------------------------------------------------------------------------
# Single-Token Decode (Autoregressive Hot Path)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
)
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_decode_write_one_new_kv_token_per_sequence(execution_mode):
    """
    Write one new K/V token per sequence into the paged KV cache using slot-based `index_put_` writes.
    Validates correct placement, no side effects, shape/dtype preservation, and CPU/Spyre consistency.
    """

    def _kernel(key_cache, val_cache, new_keys, new_vals, slots):
        """Flatten caches and scatter new K/V at given slots."""
        key_cache_flat = key_cache.view(-1, KV_HEADS, HEAD_DIM)
        val_cache_flat = val_cache.view(-1, KV_HEADS, HEAD_DIM)
        key_cache_flat.index_put_((slots,), new_keys)
        val_cache_flat.index_put_((slots,), new_vals)
        return key_cache, val_cache

    B, num_pages = 4, 512
    slots = torch.tensor([47, 63, 80, 95], dtype=torch.int32)
    new_keys = (
        torch.arange(B * KV_HEADS * HEAD_DIM, dtype=torch.float32)
        .reshape(B, KV_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
    )
    new_vals = (new_keys + 1).to(torch.bfloat16)

    # --- Spyre execution (primary) ---
    kc_arg = _cache(num_pages)
    vc_arg = torch.zeros_like(kc_arg)
    sp_kc, sp_vc = _compile_and_run(
        _kernel,
        [kc_arg, vc_arg, new_keys, new_vals, slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    kc_flat = sp_kc.view(-1, KV_HEADS, HEAD_DIM)
    vc_flat = sp_vc.view(-1, KV_HEADS, HEAD_DIM)

    # key_cache_flat[47] == new_keys[0], key_cache_flat[63] == new_keys[1], etc.
    for i, slot in enumerate(slots.tolist()):
        assert torch.equal(kc_flat[slot], new_keys[i]), (
            f"key_cache_flat[{slot}] does not match new_keys[{i}]"
        )
        assert torch.equal(vc_flat[slot], new_vals[i]), (
            f"val_cache_flat[{slot}] does not match new_vals[{i}]"
        )

    # All other cache rows remain zero (no side effects)
    written = set(slots.tolist())
    for row in range(kc_flat.shape[0]):
        if row not in written:
            assert torch.all(kc_flat[row] == 0), (
                f"key_cache_flat[{row}] should be zero but was modified"
            )
            assert torch.all(vc_flat[row] == 0), (
                f"val_cache_flat[{row}] should be zero but was modified"
            )

    # Dtype preserved as BF16
    assert sp_kc.dtype == torch.bfloat16, "key_cache dtype should be BF16"
    assert sp_vc.dtype == torch.bfloat16, "val_cache dtype should be BF16"

    # Shape of cache unchanged
    assert sp_kc.shape == (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), (
        "key_cache shape changed"
    )
    assert sp_vc.shape == (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), (
        "val_cache shape changed"
    )

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages),
        torch.zeros_like(kc_arg),
        new_keys,
        new_vals,
        slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_kc, sp_vc),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.xfail(
    reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
)
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_decode_fp8_quantized_kv(execution_mode):
    """
    Write FP8 K/V values into the paged KV cache using `index_put_`.
    Validates bit-exact writes, FP8 dtype preservation, and CPU/Spyre consistency.
    """

    def _kernel(key_cache, val_cache, new_keys, new_vals, slots):
        key_cache.view(-1, KV_HEADS, HEAD_DIM).index_put_((slots,), new_keys)
        val_cache.view(-1, KV_HEADS, HEAD_DIM).index_put_((slots,), new_vals)
        return key_cache, val_cache

    B, num_pages = 4, 512
    slots = torch.tensor([47, 63, 80, 95], dtype=torch.int32)
    new_keys = (
        torch.tensor([0.0, 1.0, -1.0, 448.0], dtype=torch.float32)
        .reshape(B, 1, 1)
        .expand(B, KV_HEADS, HEAD_DIM)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    new_vals = (
        torch.tensor([0.5, 2.0, -2.0, 240.0], dtype=torch.float32)
        .reshape(B, 1, 1)
        .expand(B, KV_HEADS, HEAD_DIM)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )

    # --- Spyre execution (primary) ---
    kc_arg = _cache(num_pages, dtype=torch.float8_e4m3fn)
    vc_arg = torch.zeros_like(kc_arg)
    sp_kc, sp_vc = _compile_and_run(
        _kernel,
        [kc_arg, vc_arg, new_keys, new_vals, slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    kc_flat = sp_kc.view(-1, KV_HEADS, HEAD_DIM)
    vc_flat = sp_vc.view(-1, KV_HEADS, HEAD_DIM)

    # Written slot values match input bit-for-bit
    for i, slot in enumerate(slots.tolist()):
        assert torch.equal(
            kc_flat[slot].view(torch.uint8), new_keys[i].view(torch.uint8)
        ), f"key_cache_flat[{slot}] FP8 bits do not match new_keys[{i}]"
        assert torch.equal(
            vc_flat[slot].view(torch.uint8), new_vals[i].view(torch.uint8)
        ), f"val_cache_flat[{slot}] FP8 bits do not match new_vals[{i}]"

    # key_cache.dtype == torch.float8_e4m3fn (no implicit conversion)
    assert sp_kc.dtype == torch.float8_e4m3fn, (
        "key_cache dtype should be float8_e4m3fn — implicit upcast occurred"
    )
    assert sp_vc.dtype == torch.float8_e4m3fn, (
        "val_cache dtype should be float8_e4m3fn — implicit upcast occurred"
    )

    # No NaN or inf introduced (check via uint8: FP8 NaN=0x7F/0xFF, inf not representable)
    kc_u8 = sp_kc.view(torch.uint8)
    assert not torch.any((kc_u8 & 0x7F) == 0x7F), (
        "key_cache contains FP8 NaN after scatter"
    )

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages, dtype=torch.float8_e4m3fn),
        torch.zeros_like(kc_arg),
        new_keys,
        new_vals,
        slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_kc, sp_vc),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_decode_large_batch_to_non_contiguous_pages(execution_mode):
    """
    Scatter a large batch of K/V data to unique, non-contiguous cache slots using `index_put_`.
    Uses model-specific KV-head shapes and validates both caches are written correctly with no side effects.
    """

    def _kernel(key_cache, new_keys, slots):
        key_cache.view(-1, KV_HEADS, HEAD_DIM).index_put_((slots,), new_keys)
        return key_cache

    torch.manual_seed(0)
    B, num_pages = 64, 4096
    slots = torch.randperm(num_pages * PAGE_SIZE, dtype=torch.int64)[:B].to(torch.int32)
    new_keys = (
        torch.arange(1, B + 1, dtype=torch.float32)
        .reshape(B, 1, 1)
        .expand(B, KV_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
        .contiguous()
    )

    # --- Spyre execution (primary) ---
    kc_arg = _cache(num_pages)
    sp_kc = _compile_and_run(
        _kernel,
        [kc_arg, new_keys, slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    kc_flat = sp_kc.view(-1, KV_HEADS, HEAD_DIM)
    written = set(slots.tolist())

    # All 64 writes land at correct slots / no write is lost or overwritten
    for i, slot in enumerate(slots.tolist()):
        expected_val = float(i + 1)  # sentinel: arange(1, B+1)
        assert torch.all(kc_flat[slot].float() == expected_val), (
            f"Write at slot {slot} (batch {i}) lost or overwritten"
        )

    # Un-written rows are exactly zero
    for row in range(kc_flat.shape[0]):
        if row not in written:
            assert torch.all(kc_flat[row] == 0), (
                f"key_cache_flat[{row}] should be zero but was modified"
            )

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages),
        new_keys,
        slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_kc,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        "compiled",
    ],
)
def test_decode_split_kv_wide_row_gemma_26b(execution_mode):
    """
    Scatter separate K/V rows into a wide [4, 2816] cache using shared slots.
    Validates correct K/V placement, no cross-contamination, and correct handling of the wide row stride.
    """

    def _kernel(key_cache, val_cache, k, v, slots):
        key_cache.index_put_((slots,), k)
        val_cache.index_put_((slots,), v)
        return key_cache, val_cache

    B, num_pages, N = 4, 512, 2816
    slots = torch.tensor([47, 63, 80, 95], dtype=torch.int32)
    k = (
        torch.arange(1, B + 1, dtype=torch.float32)
        .reshape(B, 1)
        .expand(B, N)
        .to(torch.bfloat16)
        .contiguous()
    )
    v = (k + 10).to(torch.bfloat16)

    # --- Spyre execution (primary) ---
    kc_arg = torch.zeros(num_pages, N, dtype=torch.bfloat16)
    vc_arg = torch.zeros_like(kc_arg)
    sp_kc, sp_vc = _compile_and_run(
        _kernel,
        [kc_arg, vc_arg, k, v, slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    written = set(slots.tolist())

    # Both caches written correctly at all 4 slots; K and V do not mix
    for i, slot in enumerate(slots.tolist()):
        assert torch.equal(sp_kc[slot], k[i]), (
            f"key_cache[{slot}] does not match k[{i}]"
        )
        assert torch.equal(sp_vc[slot], v[i]), (
            f"val_cache[{slot}] does not match v[{i}]"
        )
        # No cross-contamination: key_cache[slot] ≠ val_cache[slot]
        assert not torch.equal(sp_kc[slot], sp_vc[slot]), (
            f"key_cache[{slot}] == val_cache[{slot}]: K/V cross-contamination detected"
        )

    # Non-targeted slots remain zero in both caches
    for row in range(num_pages):
        if row not in written:
            assert torch.all(sp_kc[row] == 0), (
                f"key_cache[{row}] should be zero but was modified"
            )
            assert torch.all(sp_vc[row] == 0), (
                f"val_cache[{row}] should be zero but was modified"
            )

    # N=2816 stride correct — shape unchanged
    assert sp_kc.shape == (num_pages, N), "key_cache shape changed"
    assert sp_vc.shape == (num_pages, N), "val_cache shape changed"

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        torch.zeros(num_pages, N, dtype=torch.bfloat16),
        torch.zeros_like(kc_arg),
        k,
        v,
        slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_kc, sp_vc),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_decode_split_kv_large_page_pool_granite_41_20b(execution_mode):
    """
    Scatter K/V data into a larger paged cache using widely spaced slots.
    Validates independent K/V writes, unchanged shape/dtype, and correct flattened KV row layout.
    """

    def _kernel(key_cache, val_cache, new_keys, new_vals, slots):
        key_cache.view(-1, KV_HEADS, HEAD_DIM).index_put_((slots,), new_keys)
        val_cache.view(-1, KV_HEADS, HEAD_DIM).index_put_((slots,), new_vals)
        return key_cache, val_cache

    B, num_pages = 8, 1024
    slots = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int32)
    new_keys = (
        torch.arange(B, dtype=torch.float32)
        .reshape(B, 1, 1)
        .expand(B, KV_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
        .contiguous()
    )
    new_vals = (new_keys + 10).to(torch.bfloat16)

    # --- Spyre execution (primary) ---
    kc_arg = _cache(num_pages)
    vc_arg = torch.zeros_like(kc_arg)
    sp_kc, sp_vc = _compile_and_run(
        _kernel,
        [kc_arg, vc_arg, new_keys, new_vals, slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    kc_flat = sp_kc.view(-1, KV_HEADS, HEAD_DIM)
    vc_flat = sp_vc.view(-1, KV_HEADS, HEAD_DIM)

    # Correct K and V written at all 8 slots; K and V independent (no cross-contamination)
    for i, slot in enumerate(slots.tolist()):
        assert torch.equal(kc_flat[slot], new_keys[i]), (
            f"key_cache_flat[{slot}] does not match new_keys[{i}]"
        )
        assert torch.equal(vc_flat[slot], new_vals[i]), (
            f"val_cache_flat[{slot}] does not match new_vals[{i}]"
        )
        assert not torch.equal(kc_flat[slot], vc_flat[slot]), (
            f"key_cache_flat[{slot}] == val_cache_flat[{slot}]: cross-contamination detected"
        )

    # Shape and dtype unchanged
    assert sp_kc.shape == (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), (
        "key_cache shape changed"
    )
    assert sp_vc.shape == (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), (
        "val_cache shape changed"
    )
    assert sp_kc.dtype == torch.bfloat16, "key_cache dtype should be BF16"
    assert sp_vc.dtype == torch.bfloat16, "val_cache dtype should be BF16"

    # N=1024 row width correct: kc_flat row width = KV_HEADS * HEAD_DIM = 8 * 128 = 1024
    assert kc_flat.shape == (num_pages * PAGE_SIZE, KV_HEADS, HEAD_DIM), (
        "Flattened key_cache shape wrong — N=1024 row width not preserved"
    )

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages),
        torch.zeros_like(kc_arg),
        new_keys,
        new_vals,
        slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_kc, sp_vc),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# Prefill & Chunked Prefill
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_prefill_dense_scatter(execution_mode):
    """
    Write ragged packed batch K/V values into non-contiguous slots using `index_put_`.
    Validates non-monotonic scatter, lack of cross-sequence contamination, and CPU/Spyre consistency.
    """

    def _kernel(key_cache, new_keys, slot_mapping):
        cache_flat = key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        cache_flat.index_put_((slot_mapping,), new_keys)
        return key_cache

    T, num_pages = 512, 2048  # T tokens, 32 pages used (512 // 16)

    key_cache_arg = _cache(num_pages)
    new_keys = (
        torch.arange(T * GRANITE_KV_HEADS * GRANITE_HEAD_DIM, dtype=torch.float32)
        .reshape(T, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        .to(torch.bfloat16)
    )
    slot_mapping = torch.arange(T, dtype=torch.int32)

    # --- Spyre execution (primary) ---
    sp_key_cache = _compile_and_run(
        _kernel,
        [key_cache_arg, new_keys, slot_mapping],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    cache_flat = sp_key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)

    # Expected: slot 0..511 equals new_keys
    assert torch.equal(cache_flat[0:T], new_keys), (
        "cache_flat[0:T] does not match new_keys"
    )

    # Expected: all-zero past slot 511
    assert torch.equal(
        cache_flat[T:],
        torch.zeros(
            num_pages * PAGE_SIZE - T,
            GRANITE_KV_HEADS,
            GRANITE_HEAD_DIM,
            dtype=torch.bfloat16,
        ),
    ), "Non-targeted slots past T are not zero"

    # Shape and Dtype preserved
    assert sp_key_cache.dtype == torch.bfloat16, "key_cache dtype should be BF16"
    assert sp_key_cache.shape == (
        num_pages,
        PAGE_SIZE,
        GRANITE_KV_HEADS,
        GRANITE_HEAD_DIM,
    ), "key_cache shape changed"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages),
        new_keys,
        slot_mapping,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_key_cache,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_prefill_ragged_packed_batch(execution_mode):
    """
    Write long-context prefill K/V values across a large pool of pages using `index_put_`.
    Validates page-boundary alignments, position-encoded mapping, and CPU/Spyre consistency.
    """

    def _kernel(key_cache, new_keys, slot_mapping):
        cache_flat = key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        cache_flat.index_put_((slot_mapping,), new_keys)
        return key_cache

    seq_lens = [32, 48, 16]
    T = sum(seq_lens)  # 96
    num_pages = 512

    # Each sequence gets a distinct sentinel value so contamination is visible
    new_keys = torch.zeros(T, GRANITE_KV_HEADS, GRANITE_HEAD_DIM, dtype=torch.bfloat16)
    offset = 0
    for seq_id, length in enumerate(seq_lens):
        new_keys[offset : offset + length] = float(seq_id + 1)
        offset += length

    # Non-overlapping page ranges per sequence
    page_offsets = [0, 10, 20]
    slot_parts = []
    for length, page_start in zip(seq_lens, page_offsets):
        slot_parts.append(
            torch.arange(
                page_start * PAGE_SIZE,
                page_start * PAGE_SIZE + length,
                dtype=torch.int32,
            )
        )
    slot_mapping = torch.cat(slot_parts)  # [96]

    # --- Spyre execution (primary) ---
    key_cache_arg = _cache(num_pages)
    sp_key_cache = _compile_and_run(
        _kernel,
        [key_cache_arg, new_keys, slot_mapping],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    cache_flat = sp_key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)

    # Each sequence's tokens appear at its assigned slots only
    for seq_id, (length, page_start) in enumerate(zip(seq_lens, page_offsets)):
        slots = torch.arange(
            page_start * PAGE_SIZE, page_start * PAGE_SIZE + length, dtype=torch.int32
        )
        expected_val = float(seq_id + 1)
        assert torch.equal(
            cache_flat[slots],
            torch.full(
                (length, GRANITE_KV_HEADS, GRANITE_HEAD_DIM),
                expected_val,
                dtype=torch.bfloat16,
            ),
        ), f"seq {seq_id} values incorrect at its assigned slots"

    # No cross-sequence contamination: unwritten slots remain zero
    all_flat_slots = torch.arange(num_pages * PAGE_SIZE, dtype=torch.int32)
    unwritten_mask = ~torch.isin(all_flat_slots, slot_mapping)
    assert torch.equal(
        cache_flat[unwritten_mask],
        torch.zeros(
            unwritten_mask.sum().item(),
            GRANITE_KV_HEADS,
            GRANITE_HEAD_DIM,
            dtype=torch.bfloat16,
        ),
    ), "Unwritten cache slots are not zero — cross-sequence contamination detected"

    # Exactly T non-zero rows
    written_rows = (~(cache_flat == 0).all(dim=(1, 2))).sum().item()
    assert written_rows == T, f"Expected {T} written rows, got {written_rows}"

    # Shape and Dtype preserved
    assert sp_key_cache.dtype == torch.bfloat16, "key_cache dtype should be BF16"
    assert sp_key_cache.shape == (
        num_pages,
        PAGE_SIZE,
        GRANITE_KV_HEADS,
        GRANITE_HEAD_DIM,
    ), "key_cache shape changed"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages),
        new_keys,
        slot_mapping,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_key_cache,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_prefill_long_context_8192_tokens(execution_mode):
    """
    Write long-context prefill K/V values across a large pool of pages using `index_put_`.
    Validates page-boundary alignments, position-encoded mapping, and CPU/Spyre consistency.
    """

    def _kernel(key_cache, new_keys, slot_mapping):
        cache_flat = key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        cache_flat.index_put_((slot_mapping,), new_keys)
        return key_cache

    T, num_pages = 8192, 1024
    pages_used = T // PAGE_SIZE  # 512

    pos = torch.arange(T, dtype=torch.float32) / T
    new_keys = (
        pos.reshape(T, 1, 1)
        .expand(T, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        .to(torch.bfloat16)
        .contiguous()
    )
    slot_mapping = torch.arange(T, dtype=torch.int32)

    # --- Spyre execution (primary) ---
    key_cache_arg = _cache(num_pages)
    sp_key_cache = _compile_and_run(
        _kernel,
        [key_cache_arg, new_keys, slot_mapping],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    cache_flat = sp_key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)

    # Verify boundary slots 15 and 16
    assert torch.equal(cache_flat[15], new_keys[15]), "slot 15 value mismatch"
    assert torch.equal(cache_flat[16], new_keys[16]), "slot 16 value mismatch"

    # All written slots match
    assert torch.equal(cache_flat[0:T], new_keys), (
        "cache_flat[0:T] does not match new_keys"
    )

    # Pages beyond the written range are zero
    assert torch.equal(
        cache_flat[T:],
        torch.zeros(
            (num_pages - pages_used) * PAGE_SIZE,
            GRANITE_KV_HEADS,
            GRANITE_HEAD_DIM,
            dtype=torch.bfloat16,
        ),
    ), "Unwritten pages beyond T are not zero"

    # Shape and Dtype preserved
    assert sp_key_cache.dtype == torch.bfloat16, "key_cache dtype should be BF16"
    assert sp_key_cache.shape == (
        num_pages,
        PAGE_SIZE,
        GRANITE_KV_HEADS,
        GRANITE_HEAD_DIM,
    ), "key_cache shape changed"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages),
        new_keys,
        slot_mapping,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_key_cache,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_prefill_full_16384_token_context(execution_mode):
    """16 384 tokens fill all 1 024 pages — 100% cache occupancy."""

    def _kernel(key_cache, new_keys, slot_mapping):
        cache_flat = key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        cache_flat.index_put_((slot_mapping,), new_keys)
        return key_cache

    T, num_pages = 16384, 1024

    pos = (torch.arange(T, dtype=torch.float32) + 1) / T
    new_keys = (
        pos.reshape(T, 1, 1)
        .expand(T, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        .to(torch.bfloat16)
        .contiguous()
    )
    slot_mapping = torch.arange(T, dtype=torch.int32)

    # --- Spyre execution (primary) ---
    key_cache_arg = _cache(num_pages)
    sp_key_cache = _compile_and_run(
        _kernel,
        [key_cache_arg, new_keys, slot_mapping],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    cache_flat = sp_key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)

    # Verify last page boundary (16367, 16368) and last slot (16383)
    assert torch.equal(cache_flat[16367], new_keys[16367]), "slot 16367 value mismatch"
    assert torch.equal(cache_flat[16368], new_keys[16368]), "slot 16368 value mismatch"
    assert torch.equal(cache_flat[16383], new_keys[16383]), "slot 16383 value mismatch"

    # All slots match new_keys
    assert torch.equal(cache_flat[0:T], new_keys), "cache_flat does not match new_keys"

    # 100% cache occupancy (no all-zero rows)
    assert (cache_flat == 0).all(dim=(1, 2)).sum().item() == 0, (
        "Expected 100% occupancy"
    )

    # Shape and Dtype preserved
    assert sp_key_cache.dtype == torch.bfloat16, "key_cache dtype should be BF16"
    assert sp_key_cache.shape == (
        num_pages,
        PAGE_SIZE,
        GRANITE_KV_HEADS,
        GRANITE_HEAD_DIM,
    ), "key_cache shape changed"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages),
        new_keys,
        slot_mapping,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_key_cache,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_chunked_prefill_8_chunks(execution_mode):
    """Chunked prefill with 8 × 512-token sequential chunks."""

    def _kernel(key_cache, new_keys_chunk, slots_chunk):
        cache_flat = key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        cache_flat.index_put_((slots_chunk,), new_keys_chunk)
        return key_cache

    CHUNKS = 8
    CHUNK_SIZE = 512
    T_total = CHUNKS * CHUNK_SIZE  # 4096
    num_pages = 512
    new_keys_all = (
        torch.arange(T_total * GRANITE_KV_HEADS * GRANITE_HEAD_DIM, dtype=torch.float32)
        .reshape(T_total, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        .to(torch.bfloat16)
    )
    slot_mapping_all = torch.arange(T_total, dtype=torch.int32)

    # --- Spyre execution (chunk by chunk) ---
    cache_chunked = _cache(num_pages)
    for chunk_id in range(CHUNKS):
        start = chunk_id * CHUNK_SIZE
        end = start + CHUNK_SIZE
        keys_chunk = new_keys_all[start:end]
        slots_chunk = slot_mapping_all[start:end]

        cache_chunked = _compile_and_run(
            _kernel,
            [cache_chunked, keys_chunk, slots_chunk],
            DEVICE,
            compile=(execution_mode == "compiled"),
        )

    cache_flat = cache_chunked.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)

    # All chunks present and no overwrite
    assert torch.equal(cache_flat[0:T_total], new_keys_all), (
        "Accumulated chunked cache does not match full sequence"
    )

    # Shape and Dtype preserved
    assert cache_chunked.dtype == torch.bfloat16, "cache dtype should be BF16"
    assert cache_chunked.shape == (
        num_pages,
        PAGE_SIZE,
        GRANITE_KV_HEADS,
        GRANITE_HEAD_DIM,
    ), "cache shape changed"

    # --- Compare single Spyre run against CPU reference ---
    # Construct full sequence reference via single scatter to compare against accumulated chunked cache
    def _full_prefill_ref(key_cache, new_keys, slot_mapping):
        cache_flat = key_cache.view(-1, GRANITE_KV_HEADS, GRANITE_HEAD_DIM)
        cache_flat.index_put_((slot_mapping,), new_keys)
        return key_cache

    compare_with_cpu(
        _full_prefill_ref,
        _cache(num_pages),
        new_keys_all,
        slot_mapping_all,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=cache_chunked,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# Block Table Two-Level Indirection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_block_table_two_level_page_lookup(execution_mode):
    """
    Write K/V values into a 4-D paged cache using a two-level indirection lookup through a block table.
    Validates two-level index resolution, correct page-offset mapping, and CPU/Spyre consistency.
    """

    def _kernel(key_cache, new_keys, flat_slots):
        """
        Scatter new_keys into key_cache at the pre-computed flat slot indices.
        flat_slots is built on CPU from the block_table before calling the kernel.
        """
        key_flat = key_cache.view(-1, KV_HEADS, HEAD_DIM)
        key_flat.index_put_((flat_slots,), new_keys)
        return key_cache

    num_pages = 256
    batch = 2
    seq_len = 32
    block_table = torch.tensor([[5, 12], [3, 99]], dtype=torch.int32)

    sentinel = torch.tensor(
        [b * 1000 + pos for b in range(batch) for pos in range(seq_len)],
        dtype=torch.float32,
    )
    new_keys = (
        sentinel.reshape(batch, seq_len, 1, 1)
        .expand(batch, seq_len, KV_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
        .contiguous()
    )

    # Derive flat slot indices on CPU (nonzero / block_table lookup not on Spyre)
    flat_slots_list = []
    for b in range(batch):
        for pos in range(seq_len):
            phys_page = block_table[b, pos // PAGE_SIZE].item()
            page_offset = pos % PAGE_SIZE
            flat_slots_list.append(phys_page * PAGE_SIZE + page_offset)
    flat_slots = torch.tensor(flat_slots_list, dtype=torch.int32)

    # new_keys reshaped to [batch*seq_len, KV_HEADS, HEAD_DIM]
    new_keys_flat = new_keys.reshape(batch * seq_len, KV_HEADS, HEAD_DIM)

    # --- Spyre execution (primary) ---
    kc_arg = _cache(num_pages)
    sp_key = _compile_and_run(
        _kernel,
        [kc_arg, new_keys_flat, flat_slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    assert torch.equal(sp_key[5], new_keys[0, 0:PAGE_SIZE]), "key_cache[5] wrong"
    assert torch.equal(sp_key[12], new_keys[0, PAGE_SIZE:seq_len]), (
        "key_cache[12] wrong"
    )
    assert torch.equal(sp_key[3], new_keys[1, 0:PAGE_SIZE]), "key_cache[3] wrong"
    assert torch.equal(sp_key[99], new_keys[1, PAGE_SIZE:seq_len]), (
        "key_cache[99] wrong"
    )
    used = torch.tensor([5, 12, 3, 99], dtype=torch.int64)
    mask = ~torch.isin(torch.arange(num_pages), used)
    assert (sp_key[mask] == 0).all(), "unexpected pages modified"

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        _cache(num_pages),
        new_keys_flat,
        flat_slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_key,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/1219"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="known issue- https://github.com/torch-spyre/torch-spyre/issues/4450"
            ),
        ),
    ],
)
def test_block_copy_duplicate_pages_for_prefix_caching(execution_mode):
    """
    Write entire pages of cache data (prefix cache block copies) into destination pages.
    Validates page-granular bulk copies, destination page resolution, and CPU/Spyre consistency.
    """

    def _kernel(key_cache, dst_rows, src_rows):
        """Copy src_rows to dst_rows in the flattened key cache via index_copy_"""
        cache_flat = key_cache.view(-1, KV_HEADS, HEAD_DIM)
        src_data = cache_flat[src_rows].clone()
        cache_flat.index_copy_(0, dst_rows, src_data)
        return key_cache

    num_pages = 512
    src_pages = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    dst_pages = torch.tensor(
        [200, 201, 202, 203, 204, 205, 206, 207], dtype=torch.int32
    )

    # Build flat row indices on CPU
    src_rows = (src_pages.long() * PAGE_SIZE).unsqueeze(1) + torch.arange(PAGE_SIZE)
    src_rows = src_rows.reshape(-1)
    dst_rows = (dst_pages.long() * PAGE_SIZE).unsqueeze(1) + torch.arange(PAGE_SIZE)
    dst_rows = dst_rows.reshape(-1)

    def prefilled_cache():
        kc = _cache(num_pages)
        for i, p in enumerate(src_pages.tolist()):
            kc[p] = float(i)
        return kc

    # --- Spyre execution (primary) ---
    kc_arg = prefilled_cache()
    src_snapshot = kc_arg[src_pages].clone()
    sp_key = _compile_and_run(
        _kernel,
        [kc_arg, dst_rows, src_rows],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    for i in range(len(src_pages)):
        assert torch.equal(sp_key[dst_pages[i].item()], sp_key[src_pages[i].item()]), (
            f"dst page {dst_pages[i].item()} does not match src page {src_pages[i].item()}"
        )
    assert torch.equal(sp_key[src_pages], src_snapshot), "Source pages modified"
    written_t = torch.cat([src_pages.long(), dst_pages.long()])
    mask = ~torch.isin(torch.arange(num_pages), written_t)
    assert (sp_key[mask] == 0).all(), "unexpected pages modified"

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        prefilled_cache(),
        dst_rows,
        src_rows,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_key,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# ACCUMULATING SCATTER
# ---------------------------------------------------------------------------


def _kernel(cache, slots, delta):
    cache.index_add_(0, slots, delta)
    return cache


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/3507"
            ),
        ),
        "compiled",
    ],
)
def test_index_add_large_magnitude_delta_unique_slots(execution_mode):
    """
    Adds each delta vector to its corresponding unique target row in the BF16 cache using index_add_.
    Validates that targeted rows accumulate original + delta without overwriting, non-targeted rows remain unchanged, BF16 dtype is preserved, and the Spyre result matches the CPU reference.
    """
    torch.manual_seed(0)

    cache_cpu = torch.rand(512, 1024, dtype=torch.bfloat16)
    original = cache_cpu.clone()
    delta = torch.rand(4, 1024, dtype=torch.bfloat16)
    slots = torch.randperm(512, dtype=torch.int32)[:4]

    # --- CPU reference ---
    cache_cpu = _kernel(cache_cpu, slots, delta)

    # --- Spyre execution (single run) ---
    sp_result = _compile_and_run(
        _kernel,
        [original.clone(), slots, delta],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    assert len(slots.tolist()) == len(set(slots.tolist())), (
        "slots must be unique for this test variant"
    )

    for i in range(4):
        assert torch.allclose(
            sp_result[slots[i]],
            original[slots[i]] + delta[i],
            rtol=1.6e-2,
            atol=1e-5,
        ), f"slot {slots[i].item()}: cache value does not equal original + delta[{i}]"

    slots_set = set(slots.tolist())
    for s in range(512):
        if s not in slots_set:
            assert torch.equal(sp_result[s], original[s]), (
                f"row {s} was not targeted but was modified"
            )

    assert sp_result.dtype == torch.bfloat16

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        original.clone(),
        slots,
        delta,
        atol=1e-5,
        rtol=1.6e-2,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/3507"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4411"
            ),
        ),
    ],
)
def test_index_add_large_magnitude_delta_duplicate_slots(execution_mode):
    """
    Adds each delta vector to its corresponding cache row, accumulating both contributions when multiple deltas target the same slot.
    Validates correct accumulation for duplicate and unique slots, preservation of non-targeted rows and BF16 dtype, and that the Spyre result matches the CPU reference.
    """
    torch.manual_seed(0)

    cache_cpu = torch.rand(512, 1024, dtype=torch.bfloat16)
    original = cache_cpu.clone()
    delta = torch.rand(4, 1024, dtype=torch.bfloat16)
    slots = torch.tensor([10, 20, 30, 10], dtype=torch.int32)

    # --- CPU reference ---
    cache_cpu = _kernel(cache_cpu, slots, delta)

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [original.clone(), slots, delta],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # slot 10 appears at index 0 and 3 — both deltas must be accumulated
    assert torch.allclose(
        sp_result[10], original[10] + delta[0] + delta[3], atol=1e-5, rtol=1.6e-2
    ), "slot 10 (duplicate): expected original + delta[0] + delta[3]"

    assert torch.allclose(
        sp_result[20], original[20] + delta[1], atol=1e-5, rtol=1.6e-2
    ), "slot 20: expected original + delta[1]"

    assert torch.allclose(
        sp_result[30], original[30] + delta[2], atol=1e-5, rtol=1.6e-2
    ), "slot 30: expected original + delta[2]"

    slots_set = {10, 20, 30}
    for s in range(512):
        if s not in slots_set:
            assert torch.equal(sp_result[s], original[s]), (
                f"row {s} was not targeted but was modified"
            )

    assert sp_result.dtype == torch.bfloat16

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        original.clone(),
        slots,
        delta,
        atol=1e-5,
        rtol=1.6e-2,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/3507"
            ),
        ),
        "compiled",
    ],
)
def test_index_add_fp8_quantization_error_correction(execution_mode):
    """
    Adds the FP8 quantization correction (delta) to approximate KV values in the selected cache slots to recover the true BF16 values.
    Validates that the correction restores the original BF16 values exactly, non-targeted slots remain unchanged, and the Spyre result matches the CPU reference.
    """

    def _kernel(cache, slots, delta):
        cache.index_add_(0, slots, delta)
        return cache

    torch.manual_seed(1)
    slots = torch.tensor([10, 100, 200, 300], dtype=torch.int32)

    true_kv = torch.rand(4, 1024, dtype=torch.bfloat16)
    approximate_kv = true_kv.to(torch.float8_e4m3fn).to(torch.bfloat16)
    delta = true_kv - approximate_kv

    assert delta.float().abs().amax().item() < 1.0, (
        "delta should be small (bounded by FP8 quantization error)"
    )

    # --- CPU reference ---
    cache_cpu = torch.zeros(512, 1024, dtype=torch.bfloat16)
    cache_cpu[slots] = approximate_kv
    cache_cpu = _kernel(cache_cpu, slots, delta)

    # --- Spyre (compile) ---
    cache_sp_init = torch.zeros(512, 1024, dtype=torch.bfloat16)
    cache_sp_init[slots] = approximate_kv
    sp_result = _compile_and_run(
        _kernel,
        [cache_sp_init, slots, delta],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    assert torch.equal(sp_result[slots], true_kv), (
        "cache[slots] does not equal true_kv after FP8 correction"
    )

    slots_set = set(slots.tolist())
    for s in range(512):
        if s not in slots_set:
            assert torch.equal(sp_result[s], torch.zeros(1024, dtype=torch.bfloat16)), (
                f"non-targeted row {s} is not zero"
            )

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    def _kernel_with_init(slots, delta):
        cache = torch.zeros(512, 1024, dtype=torch.bfloat16)
        cache[slots.cpu()] = approximate_kv
        return _kernel(cache, slots, delta)

    compare_with_cpu(
        _kernel_with_init,
        slots,
        delta,
        atol=1e-5,
        rtol=1.6e-2,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4396"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4409"
            ),
        ),
    ],
)
def test_scatter_add_accumulate_attention_scores_gqa(execution_mode):
    """
    Adds the FP8 quantization correction (delta) to approximate KV values in the selected cache slots to recover the true BF16 values.
    Validates that the correction restores the original BF16 values exactly, non-targeted slots remain unchanged, and the Spyre result matches the CPU reference.
    """

    def _kernel(score_accum, index, partial_scores):
        score_accum.scatter_add_(0, index, partial_scores)
        return score_accum

    Q_HEADS = 32
    KV_HEADS = 8
    HEAD_DIM = 128

    partial_scores = (
        torch.arange(Q_HEADS, dtype=torch.float32)
        .reshape(Q_HEADS, 1)
        .expand(Q_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
        .contiguous()
    )

    head_mapping = torch.arange(Q_HEADS, dtype=torch.int64) // 4
    index = head_mapping.unsqueeze(1).expand_as(partial_scores)

    # --- CPU reference ---
    score_accum_cpu = torch.zeros(KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    score_accum_cpu = _kernel(score_accum_cpu, index, partial_scores)

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [torch.zeros(KV_HEADS, HEAD_DIM, dtype=torch.bfloat16), index, partial_scores],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    for k in range(KV_HEADS):
        expected = partial_scores[k * 4 : (k + 1) * 4].sum(dim=0)
        assert torch.allclose(sp_result[k], expected, rtol=1.6e-2, atol=1e-5), (
            f"KV head {k}: scatter_add_ result does not match manual sum"
        )

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        torch.zeros(KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        index,
        partial_scores,
        atol=1e-5,
        rtol=1.6e-2,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
    ],
)
def test_scatter_reduce_sum_softmax_denominator(execution_mode):
    """
    Accumulates the page-level softmax denominators into the corresponding sequence entries using scatter_reduce_ with sum reduction.
    Validates that each sequence receives the sum of its own pages without cross-sequence contamination, FP32 dtype is preserved, and the Spyre result matches the CPU reference.
    """

    def _kernel(seq_denom, page_to_seq, page_denom):
        seq_denom.scatter_reduce_(0, page_to_seq, page_denom, reduce="sum")
        return seq_denom

    page_denom = torch.ones(16, dtype=torch.float32)
    page_to_seq = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int64
    )

    # --- CPU reference ---
    seq_denom_cpu = torch.zeros(4, dtype=torch.float32)
    seq_denom_cpu = _kernel(seq_denom_cpu, page_to_seq, page_denom)

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [torch.zeros(4, dtype=torch.float32), page_to_seq, page_denom],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    assert torch.equal(sp_result, torch.tensor([4.0, 4.0, 4.0, 4.0])), (
        f"seq_denom = {sp_result.tolist()}, expected [4.0, 4.0, 4.0, 4.0]"
    )
    assert sp_result.unique().numel() == 1, "Cross-sequence contamination detected"
    assert sp_result.dtype == torch.float32

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        torch.zeros(4, dtype=torch.float32),
        page_to_seq,
        page_denom,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_amax_running_max_softmax(execution_mode):
    """
    Reduces the page-level maximum attention scores into the corresponding sequence entries using scatter_reduce_ with amax.
    Validates that each sequence receives the maximum of its own page values, not their sum, with no cross-sequence contamination, and that the Spyre result matches the CPU reference.
    """

    def _kernel(seq_max, page_to_seq, page_max):
        seq_max.scatter_reduce_(0, page_to_seq, page_max, reduce="amax")
        return seq_max

    seq_max_init = torch.full((4,), float("-inf"), dtype=torch.float32)

    page_max = torch.tensor(
        [
            1.0,
            3.0,
            2.0,
            4.0,
            0.5,
            1.5,
            2.5,
            3.5,
            2.0,
            4.0,
            6.0,
            8.0,
            1.0,
            1.0,
            1.0,
            2.0,
        ],
        dtype=torch.float32,
    )
    expected_max = torch.tensor([4.0, 3.5, 8.0, 2.0], dtype=torch.float32)
    expected_sum = torch.tensor([10.0, 8.0, 20.0, 5.0], dtype=torch.float32)

    page_to_seq = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int64
    )

    # --- CPU reference ---
    seq_max_cpu = seq_max_init.clone()
    seq_max_cpu = _kernel(seq_max_cpu, page_to_seq, page_max)

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [seq_max_init.clone(), page_to_seq, page_max],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    assert torch.equal(sp_result, expected_max), (
        f"seq_max = {sp_result.tolist()}, expected {expected_max.tolist()}"
    )
    assert not torch.equal(sp_result, expected_sum), (
        "Result equals the sum — amax reduction not working correctly"
    )
    assert sp_result.dtype == torch.float32

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        seq_max_init.clone(),
        page_to_seq,
        page_max,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_amax_kv_quantization_scale_tracking(execution_mode):
    """
    Updates each sequence's KV quantization scale tracker with the maximum absolute KV value from its corresponding pages using scatter_reduce_ with amax.
    Validates that the maximum value is retained, smaller values do not overwrite an existing maximum, FP32 dtype is preserved, and the Spyre result matches the CPU reference.
    """

    def _kernel(seq_kv_max, slot_to_seq, new_kv_abs):
        seq_kv_max.scatter_reduce_(0, slot_to_seq, new_kv_abs, reduce="amax")
        return seq_kv_max

    B, N = 4, 1024
    FP8_MAX = 448.0

    new_kv = torch.zeros(B, N, dtype=torch.bfloat16)
    new_kv[0, 0] = 5.0
    new_kv[1, 0] = 2.0
    new_kv[2, 0] = -3.0
    new_kv[3, 0] = 4.0
    new_kv[0, 1:] = 1.0
    new_kv[1, 1:] = 1.0
    new_kv[2, 1:] = 1.0
    new_kv[3, 1:] = 1.0
    known_max_abs = torch.tensor([5.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    new_kv_abs = new_kv.abs().float()
    slots = torch.arange(B, dtype=torch.int64)
    slot_to_seq = slots.unsqueeze(1).expand_as(new_kv_abs)

    # --- CPU reference ---
    seq_kv_max_cpu = torch.zeros(B, N, dtype=torch.float32)
    seq_kv_max_cpu = _kernel(seq_kv_max_cpu, slot_to_seq, new_kv_abs)

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [torch.zeros(B, N, dtype=torch.float32), slot_to_seq, new_kv_abs],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    for s in range(B):
        assert abs(sp_result[s, 0].item() - known_max_abs[s].item()) < 1e-3, (
            f"seq {s}: seq_kv_max[s, 0] = {sp_result[s, 0].item()}, expected {known_max_abs[s].item()}"
        )
        assert sp_result[s, 0].item() >= 1.0, (
            f"seq {s}: max was reduced below the smaller fill value"
        )

    scale = sp_result.amax(dim=-1) / FP8_MAX
    for s in range(B):
        assert abs((scale[s] * FP8_MAX).item() - known_max_abs[s].item()) < 0.1, (
            f"seq {s}: scale * fp8_max != known max abs"
        )

    assert sp_result.dtype == torch.float32

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        torch.zeros(B, N, dtype=torch.float32),
        slot_to_seq,
        new_kv_abs,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_amax_sliding_window_filtered_pages(execution_mode):
    """
    Reduces the scores of window-active pages into each sequences sliding-window maximum using scatter_reduce_ with amax.
    Validates that only window-active pages contribute to each sequences maximum, evicted-page scores do not affect the result, and the Spyre result matches the CPU reference.
    """

    def _kernel(seq_window_max, page_to_seq, window_page_max):
        seq_window_max.scatter_reduce_(0, page_to_seq, window_page_max, reduce="amax")
        return seq_window_max

    BATCH = 4
    PAGES_PER_SEQ = 8
    ACTIVE_PER_SEQ = 4
    total_pages = BATCH * PAGES_PER_SEQ  # 32

    all_page_max = torch.zeros(total_pages, dtype=torch.float32)
    all_page_to_seq = torch.zeros(total_pages, dtype=torch.int64)
    for s in range(BATCH):
        base = s * PAGES_PER_SEQ
        for p in range(PAGES_PER_SEQ):
            all_page_to_seq[base + p] = s
            if p < ACTIVE_PER_SEQ:
                all_page_max[base + p] = 99.0
            else:
                all_page_max[base + p] = float(s + 1) + (p - ACTIVE_PER_SEQ) * 0.5

    window_mask = torch.zeros(total_pages, dtype=torch.bool)
    for s in range(BATCH):
        base = s * PAGES_PER_SEQ
        window_mask[base + ACTIVE_PER_SEQ : base + PAGES_PER_SEQ] = True

    window_page_max = all_page_max[window_mask]
    page_to_seq = all_page_to_seq[window_mask]
    W = window_mask.sum().item()
    assert W < total_pages, "W should be < total_pages (sparse index)"

    # --- CPU reference ---
    seq_window_max_cpu = torch.full((BATCH,), float("-inf"), dtype=torch.float32)
    seq_window_max_cpu = _kernel(seq_window_max_cpu, page_to_seq, window_page_max)

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [
            torch.full((BATCH,), float("-inf"), dtype=torch.float32),
            page_to_seq,
            window_page_max,
        ],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    for s in range(BATCH):
        base = s * PAGES_PER_SEQ
        active_scores = all_page_max[base + ACTIVE_PER_SEQ : base + PAGES_PER_SEQ]
        expected = active_scores.max().item()
        assert abs(sp_result[s].item() - expected) < 1e-5, (
            f"seq {s}: window max incorrect"
        )
        assert sp_result[s].item() < 99.0, (
            f"seq {s}: evicted page score (99.0) contaminated window max"
        )

    assert sp_result.dtype == torch.float32

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        torch.full((BATCH,), float("-inf"), dtype=torch.float32),
        page_to_seq,
        window_page_max,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_amax_gqa_fixed_head_mapping(execution_mode):
    """
    Reduces attention scores from multiple Q heads into one maximum score for each corresponding KV head using scatter_reduce_(reduce="amax").
    Validates each KV head receives the maximum of its mapped Q-head scores, not their sum, with the fixed GQA mapping preserved and Spyre matching the CPU reference.
    """

    def _kernel(kv_head_max, head_mapping, q_head_max):
        kv_head_max.scatter_reduce_(0, head_mapping, q_head_max, reduce="amax")
        return kv_head_max

    # --- Granite / Mistral: 32Q → 8KV ---
    Q_HEADS, KV_HEADS_GM = 32, 8
    q_head_max_gm = torch.tensor(
        [1.0, 3.0, 2.0, 4.0] * KV_HEADS_GM, dtype=torch.float32
    )
    head_mapping_gm = torch.arange(Q_HEADS, dtype=torch.int64) // 4

    # --- CPU reference ---
    kv_head_max_cpu = torch.full((KV_HEADS_GM,), float("-inf"), dtype=torch.float32)
    kv_head_max_cpu = _kernel(kv_head_max_cpu, head_mapping_gm, q_head_max_gm)

    # rerun with different q_head_max, same fixed mapping (CPU-only reuse check)
    q_head_max_gm2 = torch.arange(1, Q_HEADS + 1, dtype=torch.float32)
    kv_head_max_cpu2 = torch.full((KV_HEADS_GM,), float("-inf"), dtype=torch.float32)
    kv_head_max_cpu2 = _kernel(kv_head_max_cpu2, head_mapping_gm, q_head_max_gm2)
    for k in range(KV_HEADS_GM):
        expected2 = q_head_max_gm2[k * 4 : (k + 1) * 4].max().item()
        assert abs(kv_head_max_cpu2[k].item() - expected2) < 1e-5, (
            f"Reuse KV head {k}: max incorrect"
        )

    # --- Gemma: 32Q → 16KV (2 Q per KV) --- CPU-only validation
    KV_HEADS_GEMMA = 16
    q_head_max_gemma = torch.tensor([1.0, 3.0] * KV_HEADS_GEMMA, dtype=torch.float32)
    head_mapping_gemma = torch.arange(Q_HEADS, dtype=torch.int64) // 2
    kv_head_max_gemma = torch.full(
        (KV_HEADS_GEMMA,), float("-inf"), dtype=torch.float32
    )
    kv_head_max_gemma = _kernel(kv_head_max_gemma, head_mapping_gemma, q_head_max_gemma)
    for k in range(KV_HEADS_GEMMA):
        expected = q_head_max_gemma[k * 2 : (k + 1) * 2].max().item()
        assert abs(kv_head_max_gemma[k].item() - expected) < 1e-5, (
            f"Gemma KV head {k}: max incorrect"
        )

    # --- Spyre (compile) — Granite/Mistral variant ---
    sp_result = _compile_and_run(
        _kernel,
        [
            torch.full((KV_HEADS_GM,), float("-inf"), dtype=torch.float32),
            head_mapping_gm,
            q_head_max_gm,
        ],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    for k in range(KV_HEADS_GM):
        expected = q_head_max_gm[k * 4 : (k + 1) * 4].max().item()
        assert abs(sp_result[k].item() - expected) < 1e-5, (
            f"Granite/Mistral KV head {k}: max incorrect"
        )
        sum_val = q_head_max_gm[k * 4 : (k + 1) * 4].sum().item()
        assert abs(sp_result[k].item() - sum_val) > 1e-3, (
            f"KV head {k}: result equals sum (should be max)"
        )

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        torch.full((KV_HEADS_GM,), float("-inf"), dtype=torch.float32),
        head_mapping_gm,
        q_head_max_gm,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_prod_joint_probability_log_prob_pipeline(execution_mode):
    """
    Reduces per-page probabilities into a joint probability for each sequence using scatter_reduce_(reduce="prod"), while the corresponding log-probabilities are accumulated through the log-product identity.
    The sequence result is a product rather than a sum, the probability and log-probability results are mathematically consistent without underflow to zero, and Spyre matches the CPU reference.
    """

    def _kernel(seq_joint_prob, page_to_seq, page_prob):
        seq_joint_prob.scatter_reduce_(0, page_to_seq, page_prob, reduce="prod")
        return seq_joint_prob

    seq_joint_prob_init = torch.ones(4, dtype=torch.float32)

    page_log_prob = torch.tensor(
        [
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.5,
            -0.6,
            -0.7,
            -0.8,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
        ],
        dtype=torch.float32,
    )
    page_to_seq = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int64
    )
    page_prob = page_log_prob.exp()

    # --- CPU reference ---
    seq_joint_prob_cpu = seq_joint_prob_init.clone()
    seq_joint_prob_cpu = _kernel(seq_joint_prob_cpu, page_to_seq, page_prob)

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [seq_joint_prob_init.clone(), page_to_seq, page_prob],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    for s in range(4):
        expected_prod = page_prob[s * 4 : (s + 1) * 4].prod().item()
        assert abs(sp_result[s].item() - expected_prod) < 1e-5, (
            f"seq {s}: joint probability incorrect"
        )

    seq_log_joint = sp_result.log()
    for s in range(4):
        expected_sum = page_log_prob[s * 4 : (s + 1) * 4].sum().item()
        assert abs(seq_log_joint[s].item() - expected_sum) < 1e-4, (
            f"seq {s}: log-joint != sum of log-probs"
        )

    for s in range(4):
        sum_val = page_prob[s * 4 : (s + 1) * 4].sum().item()
        assert abs(sp_result[s].item() - sum_val) > 1e-3, (
            f"seq {s}: result equals sum — prod reduction not working"
        )

    assert (sp_result > 0).all(), "underflow to 0.0 detected"

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        seq_joint_prob_init.clone(),
        page_to_seq,
        page_prob,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_prod_per_page_gate_weight_runtime_index(execution_mode):
    """
    Reduces per-page gate or weight values into each sequence using scatter_reduce_(reduce="prod"), multiplying the weights mapped to the same sequence.
    Each sequence receives the product of its page weights, non-targeted positions remain at the identity value 1.0, the reduction is product rather than sum, and Spyre matches the CPU
    """

    def _kernel(seq_weight, page_to_seq, page_weight):
        seq_weight.scatter_reduce_(0, page_to_seq, page_weight, reduce="prod")
        return seq_weight

    seq_weight_init = torch.ones(4, dtype=torch.float32)

    page_weight = torch.tensor(
        [
            0.9,
            0.8,
            0.95,
            0.85,
            0.7,
            0.6,
            0.5,
            0.4,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        dtype=torch.float32,
    )
    page_to_seq = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int64
    )

    # --- CPU reference ---
    seq_weight_cpu = seq_weight_init.clone()
    seq_weight_cpu = _kernel(seq_weight_cpu, page_to_seq, page_weight)

    # Rerun with unequal pages per seq (CPU-only reuse check)
    seq_weight2 = seq_weight_init.clone()
    page_weight2 = torch.tensor(
        [
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
            0.4,
            0.3,
            0.2,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        dtype=torch.float32,
    )
    page_to_seq2 = torch.tensor(
        [0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3], dtype=torch.int64
    )
    seq_weight2 = _kernel(seq_weight2, page_to_seq2, page_weight2)
    assert abs(seq_weight2[0].item() - 0.9 * 0.8) < 1e-4, (
        "rerun seq 0: 2-page product incorrect"
    )
    assert abs(seq_weight2[2].item() - 1.0) < 1e-5, (
        "rerun seq 2: non-targeted must remain 1.0"
    )

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [seq_weight_init.clone(), page_to_seq, page_weight],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    for s in range(4):
        expected = page_weight[s * 4 : (s + 1) * 4].prod().item()
        assert abs(sp_result[s].item() - expected) < 1e-4, f"seq {s}: product incorrect"

    assert abs(sp_result[2].item() - 1.0) < 1e-5, "seq 2: product of 1.0s should be 1.0"

    for s in range(4):
        sum_val = page_weight[s * 4 : (s + 1) * 4].sum().item()
        assert abs(sp_result[s].item() - sum_val) > 1e-3, (
            f"seq {s}: result equals sum — prod reduction not working"
        )

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        seq_weight_init.clone(),
        page_to_seq,
        page_weight,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_prod_per_layer_quantization_scale_fixed_index(execution_mode):
    """
    Reduces the 32 per-layer quantization scales for each sequence using scatter_reduce_(reduce="prod") with a fixed layer-to-sequence mapping.
    Each sequence receives the product of its 32 layer scales, confirming product rather than sum semantics, reusable fixed indexing, and Spyre agreement with the CPU reference.
    """

    def _kernel(seq_scale_prod, layer_to_seq, layer_scale):
        seq_scale_prod.scatter_reduce_(0, layer_to_seq, layer_scale, reduce="prod")
        return seq_scale_prod

    LAYERS = 32
    BATCH = 4

    scale_per_seq = [0.5, 0.9, 0.8, 0.7]
    layer_scale = torch.tensor(
        [scale_per_seq[s] for s in range(BATCH) for _ in range(LAYERS)],
        dtype=torch.float32,
    )
    layer_to_seq = torch.arange(LAYERS * BATCH, dtype=torch.int64) // LAYERS

    # --- CPU reference ---
    seq_scale_prod_cpu = torch.ones(BATCH, dtype=torch.float32)
    seq_scale_prod_cpu = _kernel(seq_scale_prod_cpu, layer_to_seq, layer_scale)

    # Rerun with different layer_scale, same fixed mapping (CPU-only reuse check)
    scale_per_seq2 = [0.95, 0.85, 0.75, 0.65]
    layer_scale2 = torch.tensor(
        [scale_per_seq2[s] for s in range(BATCH) for _ in range(LAYERS)],
        dtype=torch.float32,
    )
    seq_scale_prod2 = torch.ones(BATCH, dtype=torch.float32)
    seq_scale_prod2 = _kernel(seq_scale_prod2, layer_to_seq, layer_scale2)
    for s in range(BATCH):
        expected2 = layer_scale2[s * LAYERS : (s + 1) * LAYERS].prod().item()
        assert abs(seq_scale_prod2[s].item() - expected2) < 1e-4, (
            f"reuse seq {s}: layer scale product incorrect"
        )

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [torch.ones(BATCH, dtype=torch.float32), layer_to_seq, layer_scale],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    for s in range(BATCH):
        expected = layer_scale[s * LAYERS : (s + 1) * LAYERS].prod().item()
        assert abs(sp_result[s].item() - expected) < 1e-4, (
            f"seq {s}: layer scale product incorrect"
        )
        sum_val = layer_scale[s * LAYERS : (s + 1) * LAYERS].sum().item()
        assert abs(sp_result[s].item() - sum_val) > 0.01, (
            f"seq {s}: result equals sum — prod reduction not working"
        )

    assert sp_result.dtype == torch.float32

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        torch.ones(BATCH, dtype=torch.float32),
        layer_to_seq,
        layer_scale,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_amin_fp8_symmetric_quantization_range(execution_mode):
    """
    Reduces KV values for each sequence to the minimum signed value using scatter_reduce_(reduce="amin") for FP8 symmetric quantization range tracking.
    The minimum KV value is preserved for each sequence and the resulting symmetric scale covers both negative and positive extremes, with Spyre matching the CPU reference.
    """

    def _kernel_min(seq_kv_min, slot_to_seq, new_kv_f32):
        seq_kv_min.scatter_reduce_(0, slot_to_seq, new_kv_f32, reduce="amin")
        return seq_kv_min

    def _kernel_max(seq_kv_max, slot_to_seq, new_kv_f32):
        seq_kv_max.scatter_reduce_(0, slot_to_seq, new_kv_f32, reduce="amax")
        return seq_kv_max

    B, N = 4, 1024
    FP8_MAX = 448.0

    new_kv = torch.zeros(B, N, dtype=torch.bfloat16)
    min_vals = [-3.0, -6.0, -1.0, -4.0]
    max_vals = [5.0, 2.0, 1.0, 4.0]
    for s in range(B):
        new_kv[s, 0] = min_vals[s]
        new_kv[s, 1] = max_vals[s]
        new_kv[s, 2:] = 0.5

    new_kv_f32 = new_kv.float()
    slots = torch.arange(B, dtype=torch.int64)
    slot_to_seq = slots.unsqueeze(1).expand_as(new_kv_f32)

    # --- CPU reference ---
    seq_kv_min_cpu = torch.full((B, N), float("inf"), dtype=torch.float32)
    seq_kv_min_cpu = _kernel_min(seq_kv_min_cpu, slot_to_seq, new_kv_f32)

    seq_kv_max_cpu = torch.full((B, N), float("-inf"), dtype=torch.float32)
    seq_kv_max_cpu = _kernel_max(seq_kv_max_cpu, slot_to_seq, new_kv_f32)

    # --- Spyre (compile) — amin kernel ---
    sp_result = _compile_and_run(
        _kernel_min,
        [
            torch.full((B, N), float("inf"), dtype=torch.float32),
            slot_to_seq,
            new_kv_f32,
        ],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    for s in range(B):
        assert abs(sp_result[s, 0].item() - min_vals[s]) < 0.05, (
            f"seq {s}: seq_kv_min[s,0] incorrect"
        )
        assert sp_result[s, 0].item() <= sp_result[s, 2].item(), (
            f"seq {s}: filler value reduced the running min below known minimum"
        )

    sym_range = torch.maximum(seq_kv_max_cpu.abs(), sp_result.abs())
    scale = sym_range.amax(dim=-1) / FP8_MAX
    expected_sym = [max(abs(min_vals[s]), abs(max_vals[s])) for s in range(B)]
    for s in range(B):
        assert abs((scale[s] * FP8_MAX).item() - expected_sym[s]) < 0.05, (
            f"seq {s}: scale * fp8_max incorrect"
        )

    assert sp_result.dtype == torch.float32

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel_min,
        torch.full((B, N), float("inf"), dtype=torch.float32),
        slot_to_seq,
        new_kv_f32,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4399"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_mean_include_self_false(execution_mode):
    """
    Reduces page-level attention scores into the average score for each sequence using scatter_reduce_(reduce="mean", include_self=False).
    Validates each sequence receives the mean of its mapped page scores, confirming mean rather than sum or max semantics, with Spyre matching the CPU reference.
    """

    def _kernel(seq_mean_score, page_to_seq, page_score):
        seq_mean_score.scatter_reduce_(
            0, page_to_seq, page_score, reduce="mean", include_self=False
        )
        return seq_mean_score

    page_score = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            4.0,
            6.0,
            8.0,
            0.5,
            0.5,
            0.5,
            0.5,
            1.0,
            3.0,
            5.0,
            7.0,
        ],
        dtype=torch.float32,
    )
    page_to_seq = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int64
    )

    # --- CPU reference ---
    seq_mean_score_cpu = torch.zeros(4, dtype=torch.float32)
    seq_mean_score_cpu = _kernel(seq_mean_score_cpu, page_to_seq, page_score)

    # Count sensitivity: 2 pages per seq (CPU-only reuse check)
    seq_mean_2 = torch.zeros(4, dtype=torch.float32)
    page_score_2 = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32
    )
    page_to_seq_2 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    seq_mean_2.scatter_reduce_(
        0, page_to_seq_2, page_score_2, reduce="mean", include_self=False
    )
    for s in range(4):
        expected_2pg = page_score_2[s * 2 : (s + 1) * 2].mean().item()
        assert abs(seq_mean_2[s].item() - expected_2pg) < 1e-4, (
            f"count-sensitivity seq {s}: mean with 2 pages incorrect"
        )
        assert abs(seq_mean_2[s].item() - seq_mean_score_cpu[s].item()) > 1e-3, (
            f"seq {s}: mean did not change when page count changed from 4 to 2"
        )

    # --- Spyre (compile) ---
    sp_result = _compile_and_run(
        _kernel,
        [torch.zeros(4, dtype=torch.float32), page_to_seq, page_score],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions on Spyre result ---
    expected_means = [2.5, 5.0, 0.5, 4.0]
    expected_sums = [10.0, 20.0, 2.0, 16.0]
    expected_maxs = [4.0, 8.0, 0.5, 7.0]

    for s in range(4):
        assert abs(sp_result[s].item() - expected_means[s]) < 1e-4, (
            f"seq {s}: mean incorrect"
        )
        assert abs(sp_result[s].item() - expected_sums[s]) > 1e-3, (
            f"seq {s}: result equals sum (should be mean)"
        )
        if expected_means[s] != expected_maxs[s]:
            assert abs(sp_result[s].item() - expected_maxs[s]) > 1e-3, (
                f"seq {s}: result equals max (should be mean)"
            )

    assert sp_result.dtype == torch.float32

    # --- compare_with_cpu: reuse compile result; also run eager on Spyre vs CPU ---
    compare_with_cpu(
        _kernel,
        torch.zeros(4, dtype=torch.float32),
        page_to_seq,
        page_score,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        "eager",
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4629"
            ),
        ),
    ],
)
def test_scatter_reduce_mean_include_self_true(execution_mode):
    """
    Computes the mean attention score for each sequence using scatter_reduce_(reduce="mean", include_self=True), combining the existing output value with the mapped page scores.
    The result uses (initial value + sum of page scores) / (count + 1), differs from the include_self=False case when the initial value is nonzero, and captures the Spyre backend’s unsupported-op failure in eager mode.
    """

    def _kernel(seq_mean_score, page_to_seq, page_score):
        seq_mean_score.scatter_reduce_(
            0, page_to_seq, page_score, reduce="mean", include_self=True
        )
        return seq_mean_score

    page_score = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            1.0,
            2.0,
            3.0,
            4.0,
            1.0,
            2.0,
            3.0,
            4.0,
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        dtype=torch.float32,
    )
    page_to_seq = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int64
    )

    # --- CPU reference ---
    seq_mean_score_cpu = torch.full((4,), 10.0, dtype=torch.float32)
    seq_mean_score_cpu = _kernel(seq_mean_score_cpu, page_to_seq, page_score)

    # With out_initial=0.0, include_self=True uses denom=count+1 (CPU-only check)
    seq_zero = torch.zeros(4, dtype=torch.float32)
    seq_zero.scatter_reduce_(
        0, page_to_seq, page_score, reduce="mean", include_self=True
    )
    for s in range(4):
        src_sum = page_score[s * 4 : (s + 1) * 4].sum().item()
        expected_zero_init = src_sum / 5.0
        assert abs(seq_zero[s].item() - expected_zero_init) < 1e-4, (
            f"seq {s}: include_self=True with init=0 should equal sum/(count+1)"
        )

    # --- Spyre: NotImplementedError expected in eager mode ---
    if execution_mode == "eager":
        with pytest.raises(NotImplementedError, match="aten::scatter_reduce.two_out"):
            _compile_and_run(
                _kernel,
                [torch.full((4,), 10.0, dtype=torch.float32), page_to_seq, page_score],
                DEVICE,
                compile=False,
            )
    else:
        # In compiled mode, run and compare against CPU reference
        sp_result = _compile_and_run(
            _kernel,
            [torch.full((4,), 10.0, dtype=torch.float32), page_to_seq, page_score],
            DEVICE,
            compile=True,
        )
        compare_with_cpu(
            _kernel,
            torch.full((4,), 10.0, dtype=torch.float32),
            page_to_seq,
            page_score,
            target=sp_result,
            run_compile=True,
            run_eager=False,
        )


# ---------------------------------------------------------------------------
# Sliding Window / Local Attention
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="known issue-https://github.com/torch-spyre/torch-spyre/issues/4498"
)
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_ministral_sliding_window_masked_scatter(execution_mode):
    """
    Write Ministral SWA-masked K/V values into the paged KV cache using `index_put_`.
    Validates sliding-window mask filtering, non-targeted slot preservation, and CPU/Spyre consistency
    in both eager and compiled modes.
    """

    def _kernel(
        key_cache, val_cache, new_keys, new_vals, slots, window_mask, kv_heads, head_dim
    ):
        """Scatter K/V at given slots only where the window mask is True."""
        key_cache_flat = key_cache.view(-1, kv_heads, head_dim)
        val_cache_flat = val_cache.view(-1, kv_heads, head_dim)

        # Filter keys/values/slots based on the sliding window mask
        masked_slots = slots[window_mask]
        masked_keys = new_keys[window_mask]
        masked_vals = new_vals[window_mask]

        key_cache_flat.index_put_((masked_slots,), masked_keys)
        val_cache_flat.index_put_((masked_slots,), masked_vals)
        return key_cache, val_cache

    B, num_pages = 4, 512
    slots = torch.tensor([47, 63, 80, 95], dtype=torch.int32)
    # Let tokens at indices 1 and 3 be out-of-window (mask = False)
    window_mask = torch.tensor([True, False, True, False], dtype=torch.bool)

    new_keys = (
        torch.arange(B * KV_HEADS * HEAD_DIM, dtype=torch.float32)
        .reshape(B, KV_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
    )
    new_vals = (new_keys + 1).to(torch.bfloat16)

    def kernel(kc, vc, nk, nv, sl, wm):
        return _kernel(kc, vc, nk, nv, sl, wm, KV_HEADS, HEAD_DIM)

    # --- Spyre execution (primary) ---
    kc_arg = _cache(num_pages)
    vc_arg = torch.zeros_like(kc_arg)
    sp_kc, sp_vc = _compile_and_run(
        kernel,
        [kc_arg, vc_arg, new_keys, new_vals, slots, window_mask],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    kc_flat = sp_kc.view(-1, KV_HEADS, HEAD_DIM)
    vc_flat = sp_vc.view(-1, KV_HEADS, HEAD_DIM)

    # Only masked slots (where window_mask is True) should be written
    for i, slot in enumerate(slots.tolist()):
        if window_mask[i].item():
            assert torch.equal(kc_flat[slot], new_keys[i]), (
                f"key_cache_flat[{slot}] should have been written with new_keys[{i}]"
            )
            assert torch.equal(vc_flat[slot], new_vals[i]), (
                f"val_cache_flat[{slot}] should have been written with new_vals[{i}]"
            )
        else:
            assert torch.all(kc_flat[slot] == 0), (
                f"key_cache_flat[{slot}] is out-of-window and must remain zero"
            )
            assert torch.all(vc_flat[slot] == 0), (
                f"val_cache_flat[{slot}] is out-of-window and must remain zero"
            )

    # All non-targeted rows remain zero (no side effects)
    mask = ~torch.isin(torch.arange(kc_flat.shape[0]), slots)
    assert torch.all(kc_flat[mask] == 0), "key_cache: non-targeted rows modified"
    assert torch.all(vc_flat[mask] == 0), "val_cache: non-targeted rows modified"

    # Shape and Dtype preserved
    assert sp_kc.dtype == torch.bfloat16, "key_cache dtype should be BF16"
    assert sp_vc.dtype == torch.bfloat16, "val_cache dtype should be BF16"
    assert sp_kc.shape == (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), (
        "key_cache shape changed"
    )

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        kernel,
        _cache(num_pages),
        torch.zeros_like(kc_arg),
        new_keys,
        new_vals,
        slots,
        window_mask,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_kc, sp_vc),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.xfail(
    reason="known issue-https://github.com/torch-spyre/torch-spyre/issues/4332"
)
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_gemma_interleaved_global_local_caches(execution_mode):
    """Write interleaved Gemma global/local K/V values into separate caches based on layer types.
    Validates routing lookup, dual target scatter correctness, and CPU/Spyre consistency
    in both eager and compiled modes.
    """

    def _kernel(
        global_cache, local_cache, new_kv, slots, layer_types, kv_heads, head_dim
    ):
        """Route K/V values to separate global and local caches depending on layer types."""
        g_cache_flat = global_cache.view(-1, kv_heads, head_dim)
        l_cache_flat = local_cache.view(-1, kv_heads, head_dim)

        # Route based on layer_types (1 = global, 0 = local)
        g_slots = slots[layer_types == 1]
        g_kv = new_kv[layer_types == 1]

        l_slots = slots[layer_types == 0]
        l_kv = new_kv[layer_types == 0]

        g_cache_flat.index_put_((g_slots,), g_kv)
        l_cache_flat.index_put_((l_slots,), l_kv)
        return global_cache, local_cache

    B, num_pages = 4, 512
    slots = torch.tensor([47, 63, 80, 95], dtype=torch.int32)
    # Alternate layer types: 0 = Local/Sliding, 1 = Global
    layer_types = torch.tensor([1, 0, 1, 0], dtype=torch.int32)

    new_kv = (
        torch.arange(B * KV_HEADS * HEAD_DIM, dtype=torch.float32)
        .reshape(B, KV_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
    )

    def kernel(gc, lc, n_kv, sl, lt):
        return _kernel(gc, lc, n_kv, sl, lt, KV_HEADS, HEAD_DIM)

    # --- Spyre execution (primary) ---
    gc_arg = _cache(num_pages)
    lc_arg = _cache(num_pages)
    sp_gc, sp_lc = _compile_and_run(
        kernel,
        [gc_arg, lc_arg, new_kv, slots, layer_types],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # --- Spec assertions performed directly on the Spyre result ---
    gc_flat = sp_gc.view(-1, KV_HEADS, HEAD_DIM)
    lc_flat = sp_lc.view(-1, KV_HEADS, HEAD_DIM)

    for i, slot in enumerate(slots.tolist()):
        if layer_types[i].item() == 1:
            # Global layer -> Global cache written, Local cache remains zero
            assert torch.equal(gc_flat[slot], new_kv[i]), (
                f"global_cache[{slot}] should have been written with new_kv[{i}]"
            )
            assert torch.all(lc_flat[slot] == 0), (
                f"local_cache[{slot}] should remain zero for global layer type"
            )
        else:
            # Local layer -> Local cache written, Global cache remains zero
            assert torch.equal(lc_flat[slot], new_kv[i]), (
                f"local_cache[{slot}] should have been written with new_kv[{i}]"
            )
            assert torch.all(gc_flat[slot] == 0), (
                f"global_cache[{slot}] should remain zero for local layer type"
            )

    # All non-targeted rows remain zero
    mask = ~torch.isin(torch.arange(gc_flat.shape[0]), slots)
    assert torch.all(gc_flat[mask] == 0), "global_cache: non-targeted rows modified"
    assert torch.all(lc_flat[mask] == 0), "local_cache: non-targeted rows modified"

    # Shape and Dtype preserved
    assert sp_gc.dtype == torch.bfloat16 and sp_lc.dtype == torch.bfloat16
    assert sp_gc.shape == (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM)

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        kernel,
        _cache(num_pages),
        _cache(num_pages),
        new_kv,
        slots,
        layer_types,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_gc, sp_lc),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# Edge Cases & Boundary Conditions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        "compiled",
    ],
)
def test_duplicate_slots_last_writer_wins(execution_mode):
    """
    Overwrites cache slots with the provided values when multiple writes target the same slot.
    Validates last-writer-wins overwrite semantics for duplicate slots, with no accumulation, and confirms the Spyre result matches the CPU reference.
    """

    def _kernel(cache_flat, slots, new_keys):
        cache_flat.index_put_((slots,), new_keys, accumulate=False)
        return cache_flat

    slots = torch.tensor([10, 10, 20, 20], dtype=torch.int32)
    new_keys = torch.zeros(4, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    new_keys[0] = 1.0
    new_keys[1] = 2.0
    new_keys[2] = 3.0
    new_keys[3] = 4.0

    # --- Spyre execution (primary) ---
    cache_arg = torch.zeros(8192, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    sp_result = _compile_and_run(
        _kernel,
        [cache_arg, slots, new_keys],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    assert (sp_result[10] == 2.0).all(), "cache[10] should be 2.0 (last writer wins)"
    assert (sp_result[20] == 4.0).all(), "cache[20] should be 4.0"
    assert not (sp_result[10] == 3.0).all(), "values must not be summed"

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        torch.zeros(8192, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        slots,
        new_keys,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        "compiled",
    ],
)
def test_empty_batch_scatter_is_noop(execution_mode):
    """
    Performs no writes when the input contains zero tokens, leaving the KV cache unchanged.
    Validates that an empty scatter is a no-op and the Spyre result matches the CPU reference.
    """

    def _kernel(cache_flat, slots, new_keys):
        cache_flat.index_put_((slots,), new_keys)
        return cache_flat

    torch.manual_seed(42)
    new_keys = torch.zeros(0, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    slots = torch.zeros(0, dtype=torch.int32)

    cache_initial = torch.rand(8192, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    checksum_before = cache_initial.sum().item()

    # --- Spyre execution (primary) ---
    sp_result = _compile_and_run(
        _kernel,
        [cache_initial.clone(), slots, new_keys],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    assert torch.equal(sp_result, cache_initial), "Cache modified by empty scatter"
    assert sp_result.sum().item() == checksum_before, "Cache modified by empty scatter"

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        cache_initial.clone(),
        slots,
        new_keys,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_out_of_bounds_slot_raises_error(execution_mode):
    """
    Out-of-bounds slot index raises an IndexError or RuntimeError on Spyre.
    """

    def _kernel(cache_flat, slots, new_keys):
        cache_flat.index_put_((slots,), new_keys)
        return cache_flat

    cache_flat = torch.zeros(8192, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    new_keys = torch.ones(1, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    slots = torch.tensor([99999], dtype=torch.int32)

    with pytest.raises((IndexError, RuntimeError, Exception)):
        _compile_and_run(
            _kernel,
            [cache_flat, slots, new_keys],
            DEVICE,
            compile=(execution_mode == "compiled"),
        )


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_scatter_at_page_boundary(execution_mode):
    """
    Writes KV vectors to tokens at the last slot of one page and the first slot of the next page.
    Validates correct page-boundary mapping with no spill into adjacent page offsets, and confirms the Spyre result matches the CPU reference.
    """

    def _kernel(cache_flat, slots, new_keys):
        cache_flat.index_put_((slots,), new_keys)
        return cache_flat

    num_pages = 128
    key_cache = torch.zeros(
        num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    )
    new_keys = torch.zeros(2, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    new_keys[0] = 1.0
    new_keys[1] = 2.0
    slots = torch.tensor([15, 16], dtype=torch.int32)

    # --- Spyre execution (primary) ---
    cache_flat_arg = key_cache.view(-1, KV_HEADS, HEAD_DIM).clone()
    sp_result = _compile_and_run(
        _kernel,
        [cache_flat_arg, slots, new_keys],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    key_cache_sp = sp_result.view(num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    assert (key_cache_sp[0, 15] == 1.0).all(), "key_cache[0,15] should be 1.0"
    assert (key_cache_sp[1, 0] == 2.0).all(), "key_cache[1,0] should be 2.0"
    assert (key_cache_sp[0, 0:15] == 0.0).all(), "page 0 offsets 0..14 should be zero"
    assert (key_cache_sp[1, 1:16] == 0.0).all(), "page 1 offsets 1..15 should be zero"

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        key_cache.view(-1, KV_HEADS, HEAD_DIM).clone(),
        slots,
        new_keys,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_near_max_slot_index_gemma(execution_mode):
    """Near-max slot index — writes at end of large Gemma-4-26b cache."""

    def _kernel(cache, slots, new_kv):
        cache.index_put_((slots,), new_kv)
        return cache

    GEMMA_N = 2816
    TOTAL = 32768
    slots = torch.tensor([32764, 32765, 32766, 32767], dtype=torch.int32)
    new_kv = (
        torch.arange(1, 5, dtype=torch.float32)
        .reshape(4, 1)
        .expand(4, GEMMA_N)
        .to(torch.bfloat16)
        .contiguous()
    )

    # --- Spyre execution (primary) ---
    cache_arg = torch.zeros(TOTAL, GEMMA_N, dtype=torch.bfloat16)
    sp_result = _compile_and_run(
        _kernel,
        [cache_arg, slots, new_kv],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    for i, s in enumerate(slots.tolist()):
        assert torch.equal(sp_result[s], new_kv[i]), f"Slot {s} not written correctly"
    assert (sp_result[0] == 0.0).all(), "Slot 0 was unexpectedly modified"

    # --- Spyre vs CPU comparison via compare_with_cpu using single-run target ---
    compare_with_cpu(
        _kernel,
        torch.zeros(TOTAL, GEMMA_N, dtype=torch.bfloat16),
        slots,
        new_kv,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# DTYPE & MIXED-PRECISION SCENARIOS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_bf16_scatter_preserves_dtype_no_silent_promotion(execution_mode):
    """
    Scatters BF16 KV vectors into the BF16 cache while preserving their original BF16 representation.
    Validates that the scatter does not silently promote or round the BF16 values, the cache remains BF16, and the Spyre result matches the CPU reference bit-for-bit.
    """

    def _kernel(key_cache, slots, new_keys):
        """BF16 scatter: flatten cache and write new_keys at slots."""
        cache_flat = key_cache.view(-1, KV_HEADS, HEAD_DIM)
        cache_flat.index_put_((slots,), new_keys)
        return key_cache

    num_pages = 512
    B = 4
    slots = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    new_keys = (
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.bfloat16)
        .reshape(B, 1, 1)
        .expand(B, KV_HEADS, HEAD_DIM)
        .contiguous()
    )

    # --- Spyre execution (single run) ---
    kc_sp_arg = torch.zeros(
        num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    )
    sp_result = _compile_and_run(
        _kernel,
        [kc_sp_arg, slots, new_keys],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    cache_flat = sp_result.view(-1, KV_HEADS, HEAD_DIM)

    # Read back as BF16 and compare bit-for-bit via int16 view
    for i, s in enumerate(slots.tolist()):
        readback = cache_flat[s]
        assert torch.equal(
            readback.view(torch.int16),
            new_keys[i].view(torch.int16),
        ), f"Slot {s}: BF16 bits not preserved (silent promotion suspected)"

    # No rounding via upcast path
    upcast_downcast = new_keys.float().to(torch.bfloat16)
    for i, s in enumerate(slots.tolist()):
        assert torch.equal(cache_flat[s], upcast_downcast[i]), (
            f"Slot {s}: result differs from upcast-downcast reference"
        )

    assert sp_result.dtype == torch.bfloat16

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        torch.zeros(num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        slots,
        new_keys,
        atol=1e-5,
        rtol=1.6e-2,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_fp8_cache_bf16_compute_quantize_then_scatter(execution_mode):
    """
    Quantizes BF16 KV vectors to FP8 and scatters the resulting FP8 values into the FP8 KV cache.
    Validates that the quantize–scatter–dequantize pipeline preserves values within the expected FP8 quantization tolerance, applies the scale correctly, and the Spyre result matches the CPU reference.
    """

    def _kernel(key_cache, slots, new_keys_fp8):
        """FP8 scatter: flatten cache and write FP8 keys at slots."""
        cache_flat = key_cache.view(-1, KV_HEADS, HEAD_DIM)
        cache_flat.index_put_((slots,), new_keys_fp8)
        return key_cache

    num_pages = 512
    B = 4
    slots = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    torch.manual_seed(0)
    new_keys_bf16 = torch.rand(B, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16) * 2 - 1

    FP8_MAX = 448.0
    kv_scale = new_keys_bf16.float().abs().amax().item() / FP8_MAX
    kv_scale = max(kv_scale, 1e-6)

    new_keys_fp8 = (new_keys_bf16.float() / kv_scale).to(torch.float8_e4m3fn)

    # --- Spyre execution (single run) ---
    kc_sp_arg = torch.zeros(
        num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn
    )
    sp_result = _compile_and_run(
        _kernel,
        [kc_sp_arg, slots, new_keys_fp8],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    cache_flat = sp_result.view(-1, KV_HEADS, HEAD_DIM)

    # Dequantize and compare against original BF16 within tolerance
    read_back = cache_flat[slots].to(torch.float32) * kv_scale
    max_err = (read_back - new_keys_bf16.float()).abs().amax().item()
    max_abs_input = new_keys_bf16.float().abs().amax().item()
    tolerance = 0.15 * max_abs_input + kv_scale
    assert max_err < tolerance, (
        f"Max dequantization error {max_err:.4g} exceeds tolerance {tolerance:.4g}"
    )

    assert sp_result.dtype == torch.float8_e4m3fn

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        torch.zeros(
            num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn
        ),
        slots,
        new_keys_fp8,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# Split K/V Cache & Head-Parallel Scatter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_split_kv_scatter_shared_slot_mapping(execution_mode):
    """
    Writes new_keys and new_vals independently into their respective flattened KV caches at the shared slots.
    Verifies correct K/V values at target slots, no cross-cache contamination, and unchanged non-target slots.
    """

    def _kernel(key_cache, val_cache, slots, new_keys, new_vals):
        """Scatter new_keys and new_vals into their respective caches at shared slots."""
        key_flat = key_cache.view(-1, KV_HEADS, HEAD_DIM)
        val_flat = val_cache.view(-1, KV_HEADS, HEAD_DIM)
        key_flat.index_put_((slots,), new_keys)
        val_flat.index_put_((slots,), new_vals)
        return key_cache, val_cache

    B, num_pages = 8, 512
    slots = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32)

    new_keys = torch.full((B, KV_HEADS, HEAD_DIM), 1.0, dtype=torch.bfloat16)
    new_vals = torch.full((B, KV_HEADS, HEAD_DIM), -1.0, dtype=torch.bfloat16)

    # --- Spyre execution (single run) ---
    kc_sp_arg = torch.zeros(
        num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    )
    vc_sp_arg = torch.zeros_like(kc_sp_arg)
    sp_kc, sp_vc = _compile_and_run(
        _kernel,
        [kc_sp_arg, vc_sp_arg, slots, new_keys, new_vals],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    key_flat = sp_kc.view(-1, KV_HEADS, HEAD_DIM)
    val_flat = sp_vc.view(-1, KV_HEADS, HEAD_DIM)

    for s in slots.tolist():
        assert (key_flat[s] == 1.0).all(), f"key_flat[{s}] should be 1.0"
        assert (val_flat[s] == -1.0).all(), f"val_flat[{s}] should be -1.0"
        assert not (key_flat[s] == -1.0).all(), (
            f"key_flat[{s}] must not contain val value (-1.0)"
        )
        assert not (val_flat[s] == 1.0).all(), (
            f"val_flat[{s}] must not contain key value (1.0)"
        )

    non_slot_mask = ~torch.isin(torch.arange(num_pages * PAGE_SIZE), slots)
    assert (key_flat[non_slot_mask] == 0.0).all(), (
        "Non-targeted key_cache slots are not zero"
    )
    assert (val_flat[non_slot_mask] == 0.0).all(), (
        "Non-targeted val_cache slots are not zero"
    )

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        torch.zeros(num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        slots,
        new_keys,
        new_vals,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_kc, sp_vc),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# PA-KV-002
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4334"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4471"
            ),
        ),
    ],
)
def test_per_head_scatter_specific_kv_heads(execution_mode):
    """
    Writes new KV values to selected heads [0, 3, 7] at the specified slots.
    Verifies selected heads are updated while all other heads and non-target slots remain unchanged.
    """

    def _kernel(key_cache, slots, head_indices, new_head_keys):
        """
        Write new_head_keys only to the selected head_indices at the given slots.
        Reads current rows, updates only the specified heads, then writes back.
        """
        cache_flat = key_cache.view(-1, KV_HEADS, HEAD_DIM)
        rows = cache_flat[slots].clone()  # [B, 8, 128]
        rows[:, head_indices, :] = new_head_keys  # update only selected heads
        cache_flat[slots] = rows
        return key_cache

    B, num_pages = 4, 512
    head_indices = torch.tensor([0, 3, 7], dtype=torch.int64)
    slots = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    new_head_keys = torch.full((B, 3, HEAD_DIM), 1.0, dtype=torch.bfloat16)

    # --- Spyre execution (single run) ---
    kc_sp_arg = torch.full(
        (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), 99.0, dtype=torch.bfloat16
    )
    sp_result = _compile_and_run(
        _kernel,
        [kc_sp_arg, slots, head_indices, new_head_keys],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    cache_flat = sp_result.view(-1, KV_HEADS, HEAD_DIM)
    updated_heads = [0, 3, 7]
    unchanged_heads = [1, 2, 4, 5, 6]

    for s in slots.tolist():
        row = cache_flat[s]
        for h in updated_heads:
            assert (row[h] == 1.0).all(), (
                f"slot {s}, head {h}: expected 1.0 after scatter"
            )
        for h in unchanged_heads:
            assert (row[h] == 99.0).all(), (
                f"slot {s}, head {h}: expected 99.0 (should be unchanged)"
            )

    non_slot_mask = ~torch.isin(torch.arange(num_pages * PAGE_SIZE), slots)
    assert (cache_flat[non_slot_mask] == 99.0).all(), "Non-target slots were modified"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        torch.full(
            (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), 99.0, dtype=torch.bfloat16
        ),
        slots,
        head_indices,
        new_head_keys,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# Memory Efficiency & Scale
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4451"
            ),
        ),
    ],
)
def test_fp8_vs_bf16_two_x_memory_reduction(execution_mode):
    """
    Scatters matching-dtype KV vectors into selected slots of both BF16 and FP8 KV caches.
    Validates the expected 2× memory-size difference between BF16 and FP8 caches, correct writes to target slots, unchanged non-target slots, and matching Spyre/CPU results.
    """

    def _kernel_bf16(cache, slots, new_kv):
        """BF16 scatter: flatten and write new_kv at slots."""
        cache.view(-1, KV_HEADS, HEAD_DIM).index_put_((slots,), new_kv)
        return cache

    def _kernel_fp8(cache, slots, new_kv):
        """FP8 scatter: flatten and write new_kv at slots."""
        cache.view(-1, KV_HEADS, HEAD_DIM).index_put_((slots,), new_kv)
        return cache

    num_pages = 512
    B = 4
    slots = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    bf16_cache_init = torch.zeros(
        num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    )
    fp8_cache_init = torch.zeros(
        num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn
    )

    ratio = bf16_cache_init.nbytes / fp8_cache_init.nbytes
    assert abs(ratio - 2.0) < 0.01, (
        f"Expected BF16/FP8 memory ratio 2.0, got {ratio:.4f}"
    )

    new_kv_bf16 = torch.full((B, KV_HEADS, HEAD_DIM), 1.0, dtype=torch.bfloat16)
    new_kv_fp8 = torch.full((B, KV_HEADS, HEAD_DIM), 1.0, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )

    # --- Spyre execution (BF16 variant, single run) ---
    sp_result = _compile_and_run(
        _kernel_bf16,
        [bf16_cache_init.clone(), slots, new_kv_bf16],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    bf16_flat = sp_result.view(-1, KV_HEADS, HEAD_DIM)
    non_slot_mask = ~torch.isin(torch.arange(num_pages * PAGE_SIZE), slots)
    assert (bf16_flat[non_slot_mask] == 0.0).all(), (
        "BF16 cache: non-targeted slots not zero"
    )
    assert sp_result.nbytes == 512 * 16 * 8 * 128 * 2

    # --- Compare BF16 against CPU reference ---
    compare_with_cpu(
        _kernel_bf16,
        bf16_cache_init.clone(),
        slots,
        new_kv_bf16,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )

    # --- Spyre execution (FP8 variant, single run) ---
    sp_result_fp8 = _compile_and_run(
        _kernel_fp8,
        [fp8_cache_init.clone(), slots, new_kv_fp8],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    fp8_flat = sp_result_fp8.view(-1, KV_HEADS, HEAD_DIM)
    assert (fp8_flat[non_slot_mask].to(torch.float32) == 0.0).all(), (
        "FP8 cache: non-targeted slots not zero"
    )
    assert sp_result_fp8.nbytes == 512 * 16 * 8 * 128 * 1

    # --- Compare FP8 against CPU reference ---
    compare_with_cpu(
        _kernel_fp8,
        fp8_cache_init.clone(),
        slots,
        new_kv_fp8,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result_fp8,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        "compiled",
    ],
)
def test_gemma_scatter_near_maximum_slot_no_overflow(execution_mode):
    """
    Scatters new KV vectors with value 1.0 into slots near the maximum valid cache index.
    Validates that near-maximum int32 slot indices are handled correctly without overflow or corruption, non-target slots remain unchanged, and the Spyre result matches the CPU reference.
    """

    def _kernel(cache_flat, slots, new_keys):
        """Scatter new_keys into cache_flat at near-max slots."""
        cache_flat.index_put_((slots,), new_keys)
        return cache_flat

    GEMMA_KV_HEADS, GEMMA_HEAD_DIM = 16, 176
    num_pages = 8192
    TOTAL_SLOTS = num_pages * PAGE_SIZE  # 131072

    slots = torch.tensor([131056, 131057, 131058, 131059], dtype=torch.int32)
    new_keys = torch.full(
        (4, GEMMA_KV_HEADS, GEMMA_HEAD_DIM), 1.0, dtype=torch.bfloat16
    )

    # --- Spyre execution (single run) ---
    cache_flat_sp = torch.zeros(
        TOTAL_SLOTS, GEMMA_KV_HEADS, GEMMA_HEAD_DIM, dtype=torch.bfloat16
    )
    sp_result = _compile_and_run(
        _kernel,
        [cache_flat_sp, slots, new_keys],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    for s in slots.tolist():
        assert (sp_result[s] == 1.0).all(), f"Near-max slot {s} not written correctly"
    assert (sp_result[0] == 0.0).all(), "Slot 0 was unexpectedly modified"
    assert 131071 < 2**31, "Max slot index must fit in int32"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        torch.zeros(TOTAL_SLOTS, GEMMA_KV_HEADS, GEMMA_HEAD_DIM, dtype=torch.bfloat16),
        slots,
        new_keys,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# Speculative & Multi-Generation Decode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/1219"
            ),
        ),
        "compiled",
    ],
)
def test_beam_search_copy_surviving_beam_kv_pages(execution_mode):
    """
    Copies KV-cache pages from the surviving parent beams to newly allocated destination pages after beam top-K pruning
    Validates that surviving beam pages are copied correctly in both key and value caches, source and eliminated pages remain unchanged, and the deterministic top-K selection and Spyre output match the CPU reference.
    """

    def _kernel(key_flat, val_flat, dst_rows, src_rows):
        """Copy KV rows from src_rows into dst_rows via index_copy_.
        top-K pruning produces src_rows and dst_rows on CPU before the kernel.
        torch.topk is called outside the compiled region and the resulting
        row-index tensors are passed in as explicit arguments.
        """
        key_flat.index_copy_(0, dst_rows, key_flat[src_rows])
        val_flat.index_copy_(0, dst_rows, val_flat[src_rows])
        return key_flat, val_flat

    num_pages = 512
    batch = 2
    beam_width = 4
    num_beams = batch * beam_width  # 8 total beams
    vocab_size = 100

    # sentinel values per beam page
    key_cache = torch.zeros(
        num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    )
    val_cache = torch.zeros_like(key_cache)
    for i in range(num_beams):
        key_cache[i] = float(i + 1)
        val_cache[i] = float(i + 1)

    # deterministic beam_scores
    beam_scores = torch.full((num_beams, vocab_size), -999.0, dtype=torch.float32)
    for b in range(num_beams):
        best_tok = b * 10 % vocab_size
        beam_scores[b, best_tok] = -(b * 0.1)

    # topk
    topk_scores, topk_tokens = torch.topk(beam_scores, k=4, dim=1)

    #  topk is deterministic
    for b in range(num_beams):
        expected_best = b * 10 % vocab_size
        assert topk_tokens[b, 0].item() == expected_best, (
            f"beam {b}: topk best token should be {expected_best}"
        )

    # src/dst pages (disjoint)
    src_pages = torch.arange(num_beams, dtype=torch.int32)  # pages 0-7
    dst_pages = torch.arange(num_beams, num_beams * 2, dtype=torch.int32)  # pages 8-15
    assert len(set(src_pages.tolist()) & set(dst_pages.tolist())) == 0, (
        "src_pages and dst_pages must not overlap"
    )

    # Snapshot src pages before copy
    key_src_snapshot = key_cache[src_pages].clone()
    val_src_snapshot = val_cache[src_pages].clone()

    # expand page indices to flat row indices
    src_rows = (src_pages.long() * PAGE_SIZE).unsqueeze(1) + torch.arange(PAGE_SIZE)
    src_rows = src_rows.reshape(-1)
    dst_rows = (dst_pages.long() * PAGE_SIZE).unsqueeze(1) + torch.arange(PAGE_SIZE)
    dst_rows = dst_rows.reshape(-1)

    # --- Spyre execution (single run) ---
    key_cache_sp = torch.zeros(
        num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    )
    val_cache_sp = torch.zeros_like(key_cache_sp)
    for i in range(num_beams):
        key_cache_sp[i] = float(i + 1)
        val_cache_sp[i] = float(i + 1)

    key_flat_sp = key_cache_sp.view(-1, KV_HEADS, HEAD_DIM)
    val_flat_sp = val_cache_sp.view(-1, KV_HEADS, HEAD_DIM)

    sp_kf, sp_vf = _compile_and_run(
        _kernel,
        [key_flat_sp, val_flat_sp, dst_rows, src_rows],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # dst pages match src pages
    sp_key_cache = sp_kf.view(num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    sp_val_cache = sp_vf.view(num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    for i in range(num_beams):
        assert torch.equal(
            sp_key_cache[dst_pages[i].item()], sp_key_cache[src_pages[i].item()]
        ), (
            f"key_cache: dst page {dst_pages[i].item()} != src page {src_pages[i].item()}"
        )
        assert torch.equal(
            sp_val_cache[dst_pages[i].item()], sp_val_cache[src_pages[i].item()]
        ), (
            f"val_cache: dst page {dst_pages[i].item()} != src page {src_pages[i].item()}"
        )

    # source pages unmodified (copy, not move)
    assert torch.equal(sp_key_cache[src_pages], key_src_snapshot), (
        "key_cache source pages were modified — index_copy_ must be a copy, not a move"
    )
    assert torch.equal(sp_val_cache[src_pages], val_src_snapshot), (
        "val_cache source pages were modified"
    )

    #  eliminated beam pages (16+) untouched
    assert (sp_key_cache[num_beams * 2 :] == 0.0).all(), (
        "Eliminated beam pages were unexpectedly modified"
    )

    assert sp_kf.shape == (num_pages * PAGE_SIZE, KV_HEADS, HEAD_DIM)
    assert sp_kf.dtype == torch.bfloat16
    assert sp_vf.dtype == torch.bfloat16

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        key_cache.view(-1, KV_HEADS, HEAD_DIM),
        val_cache.view(-1, KV_HEADS, HEAD_DIM),
        dst_rows,
        src_rows,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_kf, sp_vf),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# index_fill_ (Slot Invalidation & Masking)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4414"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4446"
            ),
        ),
    ],
)
def test_slot_invalidation_zero_freed_pages(execution_mode):
    """
    Zeros out the KV-cache rows corresponding to freed slots before those pages are reused.
    Validates that only the freed slots in both key and value caches are cleared, all other rows remain unchanged, and the Spyre result matches the CPU reference.
    """

    def _kernel(key_cache_flat, val_cache_flat, freed_slots):
        """Zero freed KV cache slots in both key and val caches via index_fill_."""
        key_cache_flat.index_fill_(0, freed_slots, 0.0)
        val_cache_flat.index_fill_(0, freed_slots, 0.0)
        return key_cache_flat, val_cache_flat

    N = 512 * PAGE_SIZE  # 8192
    freed_slots = torch.tensor(
        [10, 50, 100, 200, 300, 350, 400, 450], dtype=torch.int64
    )

    non_freed_mask = ~torch.isin(torch.arange(N), freed_slots)

    key_init = torch.ones(N, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    val_init = torch.ones(N, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)

    key_checksum_before = key_init[non_freed_mask].sum().item()
    val_checksum_before = val_init[non_freed_mask].sum().item()

    # --- Spyre execution (single run) ---
    sp_key, sp_val = _compile_and_run(
        _kernel,
        [key_init.clone(), val_init.clone(), freed_slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # each freed slot is exactly 0.0
    for s in freed_slots.tolist():
        assert sp_key[s].eq(0.0).all(), (
            f"key_cache_flat[{s}] is not all zeros after index_fill_"
        )

    # non-freed rows unchanged
    key_checksum_after = sp_key[non_freed_mask].sum().item()
    assert key_checksum_before == key_checksum_after, (
        "Non-freed key_cache rows changed after index_fill_"
    )

    # val cache also zeroed at freed slots, non-freed unchanged
    for s in freed_slots.tolist():
        assert sp_val[s].eq(0.0).all(), (
            f"val_cache_flat[{s}] is not all zeros after index_fill_"
        )
    val_checksum_after = sp_val[non_freed_mask].sum().item()
    assert val_checksum_before == val_checksum_after, (
        "Non-freed val_cache rows changed after index_fill_"
    )

    # dtype and shape unchanged
    assert sp_key.dtype == torch.bfloat16, "key_cache dtype should be BF16"
    assert sp_key.shape == (N, KV_HEADS, HEAD_DIM), "key_cache shape changed"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        key_init.clone(),
        val_init.clone(),
        freed_slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_key, sp_val),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4414"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4446"
            ),
        ),
    ],
)
def test_speculative_decode_rollback_zero_rejected_slots(execution_mode):
    """
    Zeros out the KV-cache rows corresponding to rejected speculative draft tokens while preserving accepted token slots.
    Validates that only rejected slots are cleared, accepted slots and all other rows remain unchanged, with no cross-contamination, and the Spyre result matches the CPU reference.
    """

    def _kernel(key_cache_flat, val_cache_flat, rejected_slots):
        """Zero rejected draft token slots in both key and val caches."""
        key_cache_flat.index_fill_(0, rejected_slots, 0.0)
        val_cache_flat.index_fill_(0, rejected_slots, 0.0)
        return key_cache_flat, val_cache_flat

    N = 1024 * PAGE_SIZE  # 16384
    rejected_slots = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    accepted_slots = torch.tensor([50, 60, 70, 80], dtype=torch.int64)

    key_sp = torch.zeros(N, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    val_sp = torch.zeros(N, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    key_sp[rejected_slots] = 1.0
    val_sp[rejected_slots] = 1.0
    key_sp[accepted_slots] = 2.0
    val_sp[accepted_slots] = 2.0

    # --- Spyre execution (single run) ---
    sp_key, sp_val = _compile_and_run(
        _kernel,
        [key_sp.clone(), val_sp.clone(), rejected_slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # rejected slots are 0.0 in both caches
    for s in rejected_slots.tolist():
        assert sp_key[s].eq(0.0).all(), (
            f"key_cache_flat[{s}] should be 0.0 after rollback"
        )
        assert sp_val[s].eq(0.0).all(), (
            f"val_cache_flat[{s}] should be 0.0 after rollback"
        )

    # accepted slots intact (2.0)
    for s in accepted_slots.tolist():
        assert sp_key[s].eq(2.0).all(), (
            f"key_cache_flat[{s}] should still be 2.0 (accepted)"
        )

    # no cross-contamination: rejected and accepted sets must not overlap
    for s in rejected_slots.tolist():
        assert s not in accepted_slots.tolist(), (
            "rejected and accepted sets must not overlap"
        )

    # dtype unchanged
    assert sp_key.dtype == torch.bfloat16, "key_cache dtype should be BF16"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        key_sp.clone(),
        val_sp.clone(),
        rejected_slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=(sp_key, sp_val),
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4414"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4441"
            ),
        ),
    ],
)
def test_attention_mask_padding_fill_with_neg_inf(execution_mode):
    """
    Fills attention-bias positions corresponding to padded tokens with -inf before softmax.
    Validates that only padding positions are masked, non-padding positions remain unchanged, softmax suppresses padded positions, and the Spyre result matches the CPU reference.
    """

    def _kernel(attn_bias, pad_slots):
        """Fill pad positions in attn_bias with -inf using index_fill_ on the flat view."""
        B = attn_bias.shape[0]
        bias_2d = attn_bias.view(B, -1)
        bias_2d.index_fill_(1, pad_slots, float("-inf"))
        return attn_bias

    B = 4
    S = 64  # small S for speed; concept identical to S=8192 in production
    seq_lens = [32, 16, 24, 8]

    # compute pad_slots: positions seq_len..S-1 across all sequences
    pad_slots_list = []
    for sl in seq_lens:
        pad_slots_list.extend(range(sl, S))
    pad_slots = torch.tensor(pad_slots_list, dtype=torch.int64).unique()

    # --- Spyre execution (single run) ---
    attn_sp_init = torch.zeros(B, 1, S, S, dtype=torch.float32)
    original_shape = attn_sp_init.shape
    sp_result = _compile_and_run(
        _kernel,
        [attn_sp_init.clone(), pad_slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    bias_2d = sp_result.view(B, -1)

    # all pad_slots positions hold -inf
    for idx in pad_slots.tolist():
        assert (bias_2d[:, idx] == float("-inf")).all(), (
            f"pad column {idx} does not hold -inf"
        )

    # non-pad positions remain 0.0
    non_pad = torch.ones(S * S, dtype=torch.bool)
    non_pad[pad_slots] = False
    assert (bias_2d[:, non_pad] == 0.0).all(), (
        "Non-pad positions were modified (should remain 0.0)"
    )

    # softmax at pad positions < 1e-6
    sm = torch.softmax(sp_result, dim=-1)  # [B, 1, S, S]
    sm_2d = sm.view(B, -1)
    for idx in pad_slots.tolist():
        assert (sm_2d[:, idx] < 1e-6).all(), (
            f"softmax at pad column {idx} not near zero"
        )

    # FP32 dtype and shape unchanged
    assert sp_result.dtype == torch.float32, "attn_bias dtype should be FP32"
    assert sp_result.shape == original_shape, "attn_bias shape changed"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        attn_sp_init.clone(),
        pad_slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ---------------------------------------------------------------------------
# Mask-Based scenarios
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4441"
            ),
        ),
    ],
)
def test_masked_scatter_valid_non_padding_token_positions(execution_mode):
    """
    Writes each packed KV vector from new_kv_packed into the corresponding valid (non-padding) row of cache_flat using slots.
    Validates that only valid rows are updated with the correct KV values, padding rows remain unchanged, and the compiled Spyre result matches the CPU reference.
    """

    def _kernel(cache_flat, slots, new_kv_packed):
        """Write new_kv_packed into cache_flat at the given row slots"""
        T = slots.shape[0]
        cache_flat_2d = cache_flat.view(cache_flat.shape[0], -1)  # [N, H*Dh]
        cache_flat_2d.index_put_((slots,), new_kv_packed.view(T, -1))
        return cache_flat

    N = 512 * PAGE_SIZE  # 8192
    T = 64  # first 64 positions are valid

    valid_mask = torch.zeros(N, dtype=torch.bool)
    valid_mask[:T] = True
    new_kv_packed = (
        torch.arange(1, T + 1, dtype=torch.float32)
        .reshape(T, 1, 1)
        .expand(T, KV_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
        .contiguous()
    )

    # slots derived on CPU — aten::nonzero is not available on Spyre
    slots = valid_mask.nonzero(as_tuple=False).squeeze(1)  # [T] int64

    # --- Spyre execution (single run) ---
    cache_sp = torch.zeros(N, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    sp_result = _compile_and_run(
        _kernel,
        [cache_sp, slots, new_kv_packed],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # True positions hold corresponding source values in order
    for i in range(T):
        assert torch.equal(sp_result[i], new_kv_packed[i]), (
            f"Valid position {i}: sp_result does not match new_kv_packed[{i}]"
        )

    # False (padding) positions remain 0.0
    assert (sp_result[T:] == 0.0).all(), (
        "Padding positions are not 0.0 after index_put_"
    )

    # exactly T rows written
    written_count = slots.shape[0]
    assert written_count == T, f"Expected {T} True rows, got {written_count}"

    # dtype and shape unchanged
    assert sp_result.dtype == torch.bfloat16, "cache dtype should be BF16"
    assert sp_result.shape == (N, KV_HEADS, HEAD_DIM), "cache shape changed"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        torch.zeros(N, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        slots,
        new_kv_packed,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/692"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4441"
            ),
        ),
    ],
)
def test_masked_scatter_threshold_driven_selective_update(execution_mode):
    """
    Writes replacement KV vectors into cache rows whose attention scores fall below the specified threshold.
    Validates that only stale slots are updated, fresh slots retain their original values, the updated slot set changes with the threshold, and the Spyre output matches the CPU reference.
    """

    def _kernel(cache_flat, slots, new_kv_updates):
        """Write new_kv_updates into cache_flat at the given row slots."""
        T = slots.shape[0]
        cache_flat_2d = cache_flat.view(cache_flat.shape[0], -1)  # [N, H*Dh]
        cache_flat_2d.index_put_((slots,), new_kv_updates.view(T, -1))
        return cache_flat

    N = 8192
    threshold = 0.5

    attn_scores = torch.zeros(N, dtype=torch.float32)
    attn_scores[0::2] = 0.8  # even indices: above threshold
    attn_scores[1::2] = 0.2  # odd indices: below threshold

    update_mask = attn_scores < threshold
    T = int(update_mask.sum().item())
    # slots derived on CPU — aten::nonzero is not available on Spyre
    slots = update_mask.nonzero(as_tuple=False).squeeze(1)  # [T] int64
    new_kv_updates = torch.full((T, KV_HEADS, HEAD_DIM), 9.0, dtype=torch.bfloat16)

    # --- Spyre execution (single run) ---
    cache_sp = torch.full((N, KV_HEADS, HEAD_DIM), 5.0, dtype=torch.bfloat16)
    sp_result = _compile_and_run(
        _kernel,
        [cache_sp, slots, new_kv_updates],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # ßstale slots (odd indices) updated to 9.0
    stale_indices = update_mask.nonzero(as_tuple=True)[0]
    for s in stale_indices[:10].tolist():
        assert (sp_result[s] == 9.0).all(), f"Stale slot {s} should be 9.0 after update"

    # fresh slots (even indices) still 5.0
    fresh_indices = (~update_mask).nonzero(as_tuple=True)[0]
    for s in fresh_indices[:10].tolist():
        assert (sp_result[s] == 5.0).all(), (
            f"Fresh slot {s} should still be 5.0 (high score, not updated)"
        )

    assert sp_result.dtype == torch.bfloat16, "cache dtype should be BF16"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        torch.full((N, KV_HEADS, HEAD_DIM), 5.0, dtype=torch.bfloat16),
        slots,
        new_kv_updates,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4356"
            ),
        ),
        "compiled",
    ],
)
def test_masked_fill_zero_evicted_slots_boolean_mask(execution_mode):
    """
    Zeros out all KV-cache elements in rows marked as evicted by eviction_mask.
    Validates that only evicted rows are cleared, non-evicted rows remain unchanged, and the result matches the equivalent index_fill_ operation and CPU reference.
    """

    def _kernel(cache_flat, eviction_mask):
        """Zero evicted slots in cache_flat using a boolean mask."""
        cache_flat.masked_fill_(eviction_mask, 0.0)
        return cache_flat

    N = 512 * PAGE_SIZE  # 8192
    evicted_rows = [10, 50, 100, 200, 300, 350, 400, 450]

    eviction_mask = torch.zeros(N, 1, 1, dtype=torch.bool)
    for r in evicted_rows:
        eviction_mask[r] = True

    # Pre-compute non-evicted mask for checksum
    non_evicted = torch.ones(N, dtype=torch.bool)
    for r in evicted_rows:
        non_evicted[r] = False

    cache_init = torch.ones(N, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    checksum_before = cache_init[non_evicted].sum().item()

    # --- Spyre execution (single run) ---
    sp_result = _compile_and_run(
        _kernel,
        [cache_init.clone(), eviction_mask],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # evicted rows are 0.0
    for r in evicted_rows:
        assert (sp_result[r] == 0.0).all(), (
            f"Evicted row {r} should be 0.0 after masked_fill_"
        )

    #  non-evicted rows unchanged
    checksum_after = sp_result[non_evicted].sum().item()
    assert checksum_before == checksum_after, (
        "Non-evicted rows changed after masked_fill_"
    )

    # cross-check with index_fill_ on a fresh copy
    cache_copy = torch.ones(N, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    freed_slots_tensor = torch.tensor(evicted_rows, dtype=torch.int64)
    cache_copy.index_fill_(0, freed_slots_tensor, 0.0)
    assert torch.equal(sp_result, cache_copy), (
        "masked_fill_ result differs from equivalent index_fill_ result"
    )

    assert sp_result.dtype == torch.bfloat16, "cache dtype should be BF16"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        cache_init.clone(),
        eviction_mask,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4356"
            ),
        ),
        pytest.param(
            "compiled",
            marks=pytest.mark.xfail(
                reason="https://github.com/torch-spyre/torch-spyre/issues/4073"
            ),
        ),
    ],
)
def test_pa_masked_fill_causal_attention_mask(execution_mode):
    """
    Fills the upper-triangular positions of the attention-score matrix with -inf to prevent attention to future tokens.
    Validates that only future-token positions are masked, valid positions remain unchanged, softmax assigns zero weight to masked positions, and the reusable mask produces the same CPU and Spyre results.
    """

    def _kernel(attn_scores, causal_mask):
        """Apply causal mask by filling upper-triangle positions with -inf."""
        attn_scores.masked_fill_(causal_mask, float("-inf"))
        return attn_scores

    B, HEADS, S = 4, 32, 32

    causal_mask = torch.ones(S, S, dtype=torch.bool).triu(diagonal=1)

    # --- Spyre execution (single run) ---
    attn_init = torch.ones(B, HEADS, S, S, dtype=torch.float32)
    sp_result = _compile_and_run(
        _kernel,
        [attn_init.clone(), causal_mask],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # upper-triangle positions (j > i) hold -inf
    for i in range(S):
        for j in range(i + 1, min(i + 3, S)):  # spot-check a few per row
            assert (sp_result[:, :, i, j] == float("-inf")).all(), (
                f"Upper-triangle [{i},{j}] should be -inf"
            )

    # lower-triangle and diagonal (j <= i) remain 1.0
    for i in range(S):
        for j in range(max(0, i - 2), i + 1):
            assert (sp_result[:, :, i, j] == 1.0).all(), (
                f"Lower-triangle/diagonal [{i},{j}] should remain 1.0"
            )

    # softmax: upper-triangle → 0.0
    sm = torch.softmax(sp_result, dim=-1)
    upper = causal_mask.expand(B, HEADS, S, S)
    assert (sm[upper] == 0.0).all(), "softmax at upper-triangle positions should be 0.0"

    assert sp_result.dtype == torch.float32, "attn_scores dtype should be FP32"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        attn_init.clone(),
        causal_mask,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_masked_select_extract_active_kv_rows(execution_mode):
    """
    Extracts the KV vectors from cache rows corresponding to active (non-evicted) sequence slots using index_select.
    Validates that exactly the active rows are extracted in the correct order, no evicted rows are included, the output reshapes correctly, and the Spyre result matches the CPU reference.
    """

    def _kernel(cache_flat, slots):
        """Extract KV vectors for active (non-evicted) rows via index_select."""
        active_kv = cache_flat.index_select(0, slots)
        return active_kv.view(-1)

    N = 512 * PAGE_SIZE  # 8192
    T = 64  # 64 active rows
    active_rows = list(range(T))

    cache_flat = (
        torch.arange(1, N + 1, dtype=torch.float32)
        .reshape(N, 1, 1)
        .expand(N, KV_HEADS, HEAD_DIM)
        .to(torch.bfloat16)
        .contiguous()
    )

    active_mask = torch.zeros(N, 1, 1, dtype=torch.bool)
    for r in active_rows:
        active_mask[r] = True

    # slots derived on CPU — aten::masked_select and aten::nonzero not on Spyre
    slots = active_mask.squeeze().nonzero(as_tuple=False).squeeze(1)  # [T] int64

    # --- Spyre execution (single run) ---
    sp_result = _compile_and_run(
        _kernel,
        [cache_flat.clone(), slots],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # numel == T * H * Dh
    assert sp_result.numel() == T * KV_HEADS * HEAD_DIM, (
        f"active_kv.numel() = {sp_result.numel()}, expected {T * KV_HEADS * HEAD_DIM}"
    )

    # reshape
    active_kv_reshaped = sp_result.view(T, KV_HEADS, HEAD_DIM)

    # rows match original active rows in order
    for j in range(T):
        assert torch.equal(active_kv_reshaped[j], cache_flat[slots[j].item()]), (
            f"active_kv[{j}] does not match cache_flat[{slots[j].item()}]"
        )

    # evicted sentinel values not present
    evicted_sentinels = set(range(T + 1, N + 1))
    extracted_vals = set(active_kv_reshaped[:, 0, 0].to(torch.float32).tolist())
    overlap = evicted_sentinels & extracted_vals
    assert len(overlap) == 0, (
        f"Evicted sentinel values found in active_kv: {list(overlap)[:5]}"
    )

    assert sp_result.dtype == torch.bfloat16, "active_kv dtype should be BF16"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        cache_flat.clone(),
        slots,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.xfail(
    reason="known issue-https://github.com/torch-spyre/torch-spyre/issues/4306"
)
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_masked_select_extract_scores_above_threshold(execution_mode):
    """
    Extracts the attention scores whose values are strictly above the specified threshold using index_select.
    Validates that exactly the above-threshold scores are selected in their original order, the selected count changes correctly with the threshold, and the Spyre result matches the CPU reference.
    """

    def _kernel(attn_scores, score_indices):
        """Extract attention scores at the given indices via index_select."""
        return attn_scores.index_select(0, score_indices)

    N = 8192
    threshold = 0.5

    attn_scores = torch.zeros(N, dtype=torch.float32)
    attn_scores[0::2] = 0.8
    attn_scores[1::2] = 0.2

    score_mask = attn_scores > threshold
    # score_indices derived on CPU — aten::masked_select and aten::nonzero not on Spyre
    score_indices = score_mask.nonzero(as_tuple=False).squeeze(1)  # [K] int64

    # --- Spyre execution (single run) ---
    sp_result = _compile_and_run(
        _kernel,
        [attn_scores.clone(), score_indices],
        DEVICE,
        compile=(execution_mode == "compiled"),
    )

    # exactly K elements
    K = score_indices.shape[0]
    assert sp_result.numel() == K, (
        f"selected_scores.numel() = {sp_result.numel()}, expected {K}"
    )

    # all above threshold
    assert (sp_result > threshold).all(), "selected_scores contains values <= threshold"

    # no below-threshold values
    assert not (sp_result <= threshold).any(), (
        "selected_scores contains values <= threshold"
    )

    # values appear in original tensor order
    expected_ordered = attn_scores.index_select(0, score_indices)
    assert torch.equal(sp_result, expected_ordered), (
        "selected_scores not in original tensor order"
    )

    assert sp_result.dtype == torch.float32, "selected_scores dtype should be FP32"

    # --- Compare single Spyre run against CPU reference ---
    compare_with_cpu(
        _kernel,
        attn_scores.clone(),
        score_indices,
        atol=0,
        rtol=0,
        clone_inputs=True,
        target=sp_result,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.fixture(autouse=True)
def setup_and_teardown():
    # clear the compiler cache before each test to avoid interference between tests
    torch._dynamo.reset()
