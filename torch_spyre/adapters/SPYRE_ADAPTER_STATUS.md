# HF Adapter Spyre Status

Status of HuggingFace Transformers adapters on Spyre hardware.

## Model Compatibility Matrix

| Model | model\_type | head\_dim | D/2 | Stick Aligned | CPU Accurate | Spyre Compiles | Spyre Runs |
|-------|-----------|---------|-----|--------------|-------------|---------------|-----------|
| Qwen3 0.6B | qwen3 | 128 | 64 | Yes | Yes | Yes | Yes |
| Granite 3.3 8B | granite | 128 | 64 | Yes | Yes | Yes | Yes |
| Granite 3.3 2B | granite | 64 | 32 | **No** | Yes | **No** | — |
| Granite 4.0 1B | granitemoehybrid | 128 | 64 | Yes | Yes | Yes | Yes |
| SmolLM3 3B | smollm3 | 128 | 64 | Yes | Yes | Yes | Yes |

**CPU Accurate** = adapter produces identical greedy tokens to stock
HF on CPU.
**Spyre Compiles** = `torch.compile(block_forward)` succeeds on Spyre.
**Spyre Runs** = block produces output (no crash/NaN). Numerical
accuracy is limited by known Spyre hardware correctness issues being
fixed.

## Public API

```python
# Granite 3.3
from torch_spyre.adapters.hf_granite import load_model, generate
model = load_model("/path/to/granite-3.3-8b-instruct")

# Qwen3
from torch_spyre.adapters.hf_qwen3 import load_model, generate
model = load_model("Qwen/Qwen3-0.6B")

# Granite 4.0 (dense variants only — no Mamba layers)
from torch_spyre.adapters.hf_granitemoehybrid import load_model, generate
model = load_model("ibm-granite/granite-4.0-1b-base")

# SmolLM3
from torch_spyre.adapters.hf_smollm3 import load_model, generate
model = load_model("HuggingFaceTB/SmolLM3-3B-Base")

# Generate (same for all models)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")
outputs = generate(
    model, tokenizer, ["What is 2+2?"], max_new_tokens=128,
)
```

Each adapter also exposes `prepare_for_spyre(model)` for manual
control:

```python
from transformers import AutoModelForCausalLM
from torch_spyre.adapters.hf_granite import (
    prepare_for_spyre, generate,
)

model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.float16, device_map="cpu",
)
prepare_for_spyre(model)
model.to("spyre")
outputs = generate(
    model, tokenizer, ["Hello!"], max_new_tokens=32,
)
```

## How the Adapters Work

### Architecture

Each adapter follows the FMS `eager_spyre` compilation pattern:
compiled block functions with raw tensor KV caches, precomputed
RoPE rotation matrices, fp16 RMSNorm, and padded 64-block decode
generation loop.

```
torch_spyre/adapters/
├── hf_common.py          — shared utilities
│   PrecomputedRotaryEmbedding, apply_rope_matmul,
│   patch_rmsnorm, pad_lm_head, kv_cache_update,
│   build_prefill_mask, build_expansion_mask,
│   load_model_common, generate
├── hf_granite.py          — Granite 3.3 adapter
├── hf_qwen3.py            — Qwen3 adapter
├── hf_granitemoehybrid.py — Granite 4.0 dense adapter
├── hf_smollm3.py          — SmolLM3 adapter
├── hf_phi3.py             — Phi-4 mini adapter (blocked)
└── __init__.py
```

```
┌───────────────────────────────────────────────┐
│  generate() — Python loop (hf_common.py)      │
│  ┌─────────────────────────────────────────┐  │
│  │  _run_forward() — per-model             │  │
│  │  ┌───────────────────────────────────┐  │  │
│  │  │  compiled block_forward()         │  │  │
│  │  │  • RMSNorm (fp16, patched)        │  │  │
│  │  │  • QKV projections                │  │  │
│  │  │  • RoPE (matmul, no slicing)      │  │  │
│  │  │  • KV cache update                │  │  │
│  │  │  • SDPA (enable_gqa=True)         │  │  │
│  │  │  • Output projection              │  │  │
│  │  │  • MLP (SwiGLU)                   │  │  │
│  │  └───────────────────────────────────┘  │  │
│  │  x N layers                             │  │
│  │  Final RMSNorm + LM head                │  │
│  └─────────────────────────────────────────┘  │
│  Token selection (CPU) + buffer management    │
└───────────────────────────────────────────────┘
```

### Deviations from Stock HuggingFace Transformers

#### 1. RoPE: Precomputed Rotation Matrices

| | HF Transformers | Adapter |
|---|---|---|
| **Where** | `*RotaryEmbedding.forward()` | `PrecomputedRotaryEmbedding` |
| **sin/cos** | Computed every forward call | Precomputed once on CPU, cached as `[S, 2, 2, D/2]` rotation matrices |
| **Application** | `rotate_half()` + slices | `apply_rope_matmul()` — reshape to `[B, L, H, 2, D/2]`, broadcast multiply, sum. No slicing. |
| **Output** | `(cos, sin)` tuple | `selected_freqs` tensor `[B, L, 2, 2, D/2]` |

**Why:** Spyre has no `sin`/`cos` hardware ops and
`aten.slice.Tensor` falls back to CPU with stride mismatches
inside compiled graphs.

#### 2. RMSNorm: Class-Level Patch

| | HF Transformers | Adapter |
|---|---|---|
| **Mechanism** | Each model has its own RMSNorm class | `patch_rmsnorm(cls)` patches any RMSNorm class in-place |
| **Precision** | Casts to float32 for variance | Spyre: stays fp16. CPU: float32 (matches HF). |
| **Variance** | `hidden_states.pow(2).mean()` | Spyre: `(hidden_states * hidden_states).mean()`. CPU: same as HF. |
| **Epsilon** | Python float scalar | Spyre: `torch.ops.spyre.full((1,), eps, ...)` tensor. CPU: Python float. |

**Why:** Spyre does not support dtype conversion on-device. The
`pow(2)` op is also not well supported; element-wise multiply is
native.

#### 3. LM Head Weight: Padded

| | HF Transformers | Adapter |
|---|---|---|
| **Vocab dim** | As-is from model config | Padded to `ceil(vocab/64)*64 + 64` via `pad_lm_head()` |

**Why:** Spyre requires tensor dimensions aligned to 64-element
sticks (128 bytes at fp16) for efficient matmul work division.
The extra +64 avoids prime-number multiples that cause poor work
distribution across Spyre cores.

#### 4. Decoder Layers: Custom Compiled Blocks

| | HF Transformers | Adapter |
|---|---|---|
| **Layer forward** | `*DecoderLayer.forward()` | `block_forward()` — plain function closure wrapping the same weights |
| **KV cache** | `DynamicCache` Python object | Raw tensor lists passed as function args |
| **Cache update** | `torch.cat` inside `DynamicCache.update()` | `torch.cat` (expand) or `spyre.overwrite` (fill) |
| **Compilation** | Not compiled by default | `torch.compile(block_forward, dynamic=False)` |

**Why:** `DynamicCache` is a Python object with side effects that
causes graph breaks in `torch.compile`. Raw tensor args trace
cleanly. `torch.ops.spyre.overwrite` must execute inside the
compiled graph to produce Spyre device code.

#### 5. Generation Loop: Custom Implementation

| | HF Transformers | Adapter |
|---|---|---|
| **Entry point** | `GenerationMixin.generate()` | `generate()` in `hf_common.py` |
| **Decode protocol** | Token-by-token with dynamic cache growth | 64-block padded decode: prefill, expand, fill (x63), expand cycle |
| **Prompt handling** | Right-padded or unpadded | Left-padded to multiple of 64 |
| **KV cache growth** | Grows by 1 per token | Grows by 64 per expansion, then 63 single-slot overwrites |
| **Extras** | Full sampling, beam search, etc. | Greedy + top-k sampling, per-token timing |

**Why:** Spyre requires fixed-size block decode with
`spyre.overwrite` for KV cache updates. HF's generate has dynamic
shapes, CPU-only causal mask creation, and DynamicCache that are
all incompatible with Spyre's static-shape compilation model.

#### 6. Attention Mask: Built Externally

| | HF Transformers | Adapter |
|---|---|---|
| **Creation** | `create_causal_mask()` using `torch.tril` | `build_prefill_mask()` / `build_expansion_mask()` on CPU |
| **Dtype** | Model dtype (may be float32) | Always `float16` |
| **Transfer** | On-device | Built on CPU, moved to Spyre |

**Why:** `torch.tril` is not supported on Spyre. Masks must be
`float16` (Spyre's native dtype).

#### 7. Embedding: No Change Required

HF's `nn.Embedding` automatically falls back to CPU via
torch-spyre's fallback mechanism. The result is transparently
transferred to the Spyre device. No adapter code needed.

### What Works As-Is (No Patching)

These HF Transformer components run natively on Spyre without
modification:

| Component | HF Class/Function | Spyre Support |
|---|---|---|
| Linear projections | `nn.Linear` | Native matmul |
| MLP activation | `nn.SiLU` (SwiGLU) | Native `silu` |
| Embedding multiplier | scalar-tensor mul | Native |
| Residual multiplier | scalar-tensor mul | Native |
| Logits scaling | tensor-scalar div | Native |
| SDPA | `F.scaled_dot_product_attention` | Decomposed by torch-spyre |
| GQA head expansion | `enable_gqa=True` | SDPA decomposition |
| Embedding lookup | `nn.Embedding` | CPU fallback (automatic) |

### Model-Specific Differences

| Feature | Granite 3.3 | Qwen3 | Granite 4.0 | SmolLM3 |
|---------|------------|-------|-------------|---------|
| Embedding multiplier | Yes | No | Yes | No |
| Residual multiplier | Yes | No | Yes | No |
| Logits scaling | Yes | No | Yes | No |
| Q/K RMSNorm | No | Yes (per-head) | No | No |
| Fused MLP weights | No | No | Yes (split at prepare time) | No |
| NoPE layers | No | No | No | Yes (conditional RoPE) |
| Attention scaling | `config.attention_multiplier` | `head_dim**-0.5` | `config.attention_multiplier` | `head_dim**-0.5` |

## Adding a New Model

### Checklist

1. Check `head_dim >= 128` (see Stick Alignment below)
2. Check for fused weights that need splitting (like Granite 4.0's
   `input_linear`)
3. Check for partial RoPE (`partial_rotary_factor < 1.0`) — requires
   splitting Q/K into rotated/non-rotated portions, which hits
   stickify non-zero offset assertions (see Known Issues)
4. Check for model-specific multipliers (embedding, residual,
   attention, logits) — must be preserved in the block function
5. Check for per-layer variations (NoPE layers, sliding window,
   MoE routing)
6. Verify CPU accuracy before testing on Spyre

### Comparison with FMS `eager_spyre` Approach

| Aspect | FMS `eager_spyre` | HF Adapter |
|---|---|---|
| Model source | FMS (custom codebase) | HuggingFace Transformers (standard) |
| RoPE | Matmul with `selected_freqs` | Same approach, applied via monkey-patch |
| RMSNorm | `torch.nn.RMSNorm` | `patch_rmsnorm(cls)` patches HF's RMSNorm in-place |
| KV cache | Tuple of tensors through FMS attention | List of tensors passed to compiled block function |
| GQA | Manual expand/flatten | `F.scaled_dot_product_attention(enable_gqa=True)` |
| Compilation | `block.compile(dynamic=False)` | `torch.compile(block_forward, dynamic=False)` |
| Generation | `fms.utils.generation` | `hf_common.generate()` |
| Weight loading | FMS serialization | HF `from_pretrained` then `.to("spyre")` |
| Maintenance | Requires FMS fork | No fork; runtime monkey-patches on stock HF |

## Stick Alignment Requirement

The RoPE matmul implementation reshapes Q/K to `[B, L, H, 2, D/2]`
where `D = head_dim`. When `D/2 < 64` (sub-stick), the stickify
compiler pass generates a compound dimension expression that it
cannot decompose, causing an `AssertionError`:

```
AssertionError: Could not find a host dimension matching stick
expr d4 in [0, d0, 64*d1 + 32*d3 + d4]
```

**Rule: a model works on Spyre if `head_dim >= 128`
(i.e. `D/2 >= 64`).**

All models tested with `head_dim=128` compile and run. Granite 3.3
2B (`head_dim=64`) is the only failure.

Note that `head_dim` is not always
`hidden_size // num_attention_heads`. Some models (e.g. Qwen3)
explicitly set `head_dim` in config independent of `hidden_size`
and `num_attention_heads`.

### Checking a new model

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("org/model-name")
head_dim = getattr(
    config, "head_dim",
    config.hidden_size // config.num_attention_heads,
)
will_compile = (head_dim // 2) >= 64
```

## Known Issues

### Spyre Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| No `sin`/`cos` ops | RoPE must be precomputed | `PrecomputedRotaryEmbedding` |
| No dtype conversion | RMSNorm must stay fp16 | Patched forward with device check |
| No `aten.slice` in compiled graphs | KV cache indexing falls back to CPU | `spyre.overwrite` for fill mode |
| `head_dim/2 < 64` (sub-stick) | Stickify assertion on RoPE matmul | Only use models with `head_dim >= 128` |
| `partial_rotary_factor < 1.0` | Non-zero offset assertion in stickify | Split Q/K weights (not yet implemented; blocks Phi-4) |
| Zero-length tensors crash `copy_host_to_device` | Segfault on `.to("spyre")` | Create empty tensors directly on device |
| fp16 overflow on CPU for large multipliers | NaN logits on CPU | Test in float32; runs fine on Spyre hardware |

### Performance Issues

These affect speed but not correctness:

**Compilation overhead (first run):** The first invocation compiles
graphs per layer per mode (expand + fill). This takes several
minutes. Subsequent runs with the same shapes reuse cached compiled
graphs.

**`aten.slice` fallback in fill mode:** The KV cache fill operation
`k[:, :, token_index:token_index+1, :]` inside `spyre.overwrite`
triggers an `aten.slice.Tensor` CPU fallback per layer per fill
step.

**Recompilation per `token_index`:** Each unique `token_index`
value in fill mode triggers a new graph specialization (because
`torch.compile` specializes on Python int arguments). Over 63 fill
steps, this causes 63 recompilations on first use.

### Open Work

1. **Fix `token_index` recompilation** — pass as tensor to avoid
   specialization
2. **Fix `aten.slice` fallback in fill** — restructure overwrite
   call
3. **Multi-iteration benchmarking** — run 5+ iterations to measure
   steady-state latency (after compilation cache is warm)
4. **Phi-4 mini** — blocked by `partial_rotary_factor=0.75`
   (stickify non-zero offset assertion). Fix: split Q/K weights so
   rotated dims are separate linears.
5. **Stick alignment for small head\_dim** — Granite 3.3 2B
   (`head_dim=64`) fails stickify. Needs compiler fix or
   alternative RoPE implementation.

## Test Scripts

| Script | Purpose | Requires Spyre |
|--------|---------|---------------|
| `tests/adapters/test_adapter_cpu_accuracy.py` | CPU: adapter vs HF logit comparison | No |
| `tests/adapters/test_block_cpu_vs_spyre.py` | Per-layer CPU vs Spyre block comparison | Yes |
| `tests/adapters/test_e2e_smoke_spyre.py` | E2E: load model, generate tokens | Yes |
| `tests/adapters/test_e2e_token_compare_spyre.py` | E2E: HF CPU vs adapter Spyre tokens | Yes |
