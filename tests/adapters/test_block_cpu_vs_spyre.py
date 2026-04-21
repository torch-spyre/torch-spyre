# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Per-layer CPU vs Spyre compiled block comparison.

Creates tiny random-weight models (3 layers each, no weight download needed),
runs each decoder block on CPU (uncompiled) and Spyre (compiled), and compares
the output hidden states numerically.

Usage (on Spyre pod):
    python3 test_block_cpu_vs_spyre.py [qwen3|granite|granite4|smollm3|all]

Output: markdown tables with per-layer max/mean abs diff and NaN flags.
"""

import sys
import traceback

import torch
import torch_spyre  # noqa: F401 — registers Spyre device

DEVICE = "spyre"
SEQ_LEN = 64  # prefill sequence length (one Spyre stick)


# ---------------------------------------------------------------------------
# Model registry: tiny configs for each model family (no weight download)
# ---------------------------------------------------------------------------

def _make_qwen3_config():
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:
        from transformers import Qwen3Config
        cfg = Qwen3Config(
            hidden_size=1024, num_attention_heads=16, num_key_value_heads=2,
            intermediate_size=2560, num_hidden_layers=3, vocab_size=151936,
            rms_norm_eps=1e-6, max_position_embeddings=4096,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_granite_config():
    from transformers import AutoConfig
    try:
        # Use 8B config (head_dim=128, D/2=64 = one stick — compiles on Spyre).
        # The 2B config has head_dim=64 (D/2=32) which triggers a stickify
        # assertion with 32 attention heads.
        cfg = AutoConfig.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
    except Exception:
        from transformers import GraniteConfig
        cfg = GraniteConfig(
            hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
            intermediate_size=10944, num_hidden_layers=3, vocab_size=49152,
            rms_norm_eps=1e-5, max_position_embeddings=4096,
            embedding_multiplier=12.0, residual_multiplier=0.22,
            attention_multiplier=0.0625, logits_scaling=13.0,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_granite4_config():
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained("ibm-granite/granite-4.0-1b-base")
    except Exception:
        from transformers import GraniteMoeHybridConfig
        cfg = GraniteMoeHybridConfig(
            hidden_size=2048, num_attention_heads=16, num_key_value_heads=4,
            shared_intermediate_size=5632, num_hidden_layers=3, vocab_size=49152,
            rms_norm_eps=1e-5, max_position_embeddings=4096,
            embedding_multiplier=12.0, residual_multiplier=0.22,
            attention_multiplier=0.0625, logits_scaling=13.0,
            num_local_experts=0,
            layers_block_type=["attention"] * 3,
        )
    cfg.num_hidden_layers = 3
    # Force all-attention (no Mamba)
    if hasattr(cfg, "layers_block_type"):
        cfg.layers_block_type = ["attention"] * cfg.num_hidden_layers
    if hasattr(cfg, "num_local_experts"):
        cfg.num_local_experts = 0
    cfg._attn_implementation = "eager"
    return cfg


def _make_smollm3_config():
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
    except Exception:
        from transformers import SmolLM3Config
        cfg = SmolLM3Config(
            hidden_size=2048, num_attention_heads=16, num_key_value_heads=4,
            intermediate_size=11008, num_hidden_layers=4, vocab_size=128256,
            rms_norm_eps=1e-6, max_position_embeddings=4096,
            no_rope_layer_interval=4,
        )
    cfg.num_hidden_layers = 4  # keep 4 so NoPE pattern (interval=4) is tested
    cfg.pad_token_id = None
    # Recompute no_rope_layers for the reduced layer count
    interval = getattr(cfg, "no_rope_layer_interval", 4)
    cfg.no_rope_layers = [
        int((i + 1) % interval != 0) for i in range(cfg.num_hidden_layers)
    ]
    if hasattr(cfg, "layer_types"):
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    cfg._attn_implementation = "eager"
    return cfg


MODEL_REGISTRY = {
    "qwen3": {
        "name": "Qwen3 0.6B",
        "config_fn": _make_qwen3_config,
        "adapter": "torch_spyre.adapters.hf_qwen3",
    },
    "granite": {
        "name": "Granite 3.3 8B",
        "config_fn": _make_granite_config,
        "adapter": "torch_spyre.adapters.hf_granite",
    },
    "granite4": {
        "name": "Granite 4.0",
        "config_fn": _make_granite4_config,
        "adapter": "torch_spyre.adapters.hf_granitemoehybrid",
    },
    "smollm3": {
        "name": "SmolLM3",
        "config_fn": _make_smollm3_config,
        "adapter": "torch_spyre.adapters.hf_smollm3",
    },
}


# ---------------------------------------------------------------------------
# Input creation
# ---------------------------------------------------------------------------

def make_inputs(config, mode, seed, cache_len=64, device="cpu"):
    """Create deterministic random inputs for a block_forward call.

    Args:
        config: HF model config
        mode: "prefill" or "decode"
        seed: random seed
        cache_len: KV cache length for decode mode
        device: target device ("cpu" or "spyre")

    Returns: dict of tensors on the specified device (fp16)
    """
    torch.manual_seed(seed)
    H = config.hidden_size
    head_dim = getattr(
        config, "head_dim", H // config.num_attention_heads
    )
    num_kv_heads = config.num_key_value_heads
    half_dim = head_dim // 2

    if mode == "prefill":
        L = SEQ_LEN
        # Create on CPU first, then move (zero-length tensors crash on Spyre
        # .to(), so create empty KV caches directly on the target device)
        hidden = torch.randn(1, L, H, dtype=torch.float16).to(device)
        freqs = torch.randn(1, L, 2, 2, half_dim, dtype=torch.float16).to(
            device
        )
        mask = torch.zeros(1, 1, L, L, dtype=torch.float16)
        for i in range(L):
            mask[:, :, i, i + 1:] = -torch.inf
        mask = mask.to(device)
        # Empty KV caches: create directly on target device (avoids segfault)
        kc = torch.empty(
            1, num_kv_heads, 0, head_dim, dtype=torch.float16, device=device
        )
        vc = torch.empty(
            1, num_kv_heads, 0, head_dim, dtype=torch.float16, device=device
        )
        is_filling = False
    else:
        L = 1
        total = cache_len + 1
        hidden = torch.randn(1, L, H, dtype=torch.float16).to(device)
        freqs = torch.randn(1, L, 2, 2, half_dim, dtype=torch.float16).to(
            device
        )
        mask = torch.zeros(1, 1, L, total, dtype=torch.float16).to(device)
        kc = torch.randn(
            1, num_kv_heads, cache_len, head_dim, dtype=torch.float16
        ).to(device)
        vc = torch.randn(
            1, num_kv_heads, cache_len, head_dim, dtype=torch.float16
        ).to(device)
        is_filling = False

    return {
        "hidden_states": hidden,
        "selected_freqs": freqs,
        "attn_mask": mask,
        "key_cache": kc,
        "value_cache": vc,
        "is_filling": is_filling,
        "token_index": 0,
        "cache_position": 0,
    }


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def compare_block(uncompiled_fn, compiled_fn, inputs_cpu):
    """Run block on CPU (uncompiled) and Spyre (compiled), compare outputs."""
    # --- CPU ---
    with torch.no_grad():
        cpu_h, cpu_kc, cpu_vc = uncompiled_fn(
            inputs_cpu["hidden_states"],
            inputs_cpu["selected_freqs"],
            inputs_cpu["attn_mask"],
            inputs_cpu["key_cache"],
            inputs_cpu["value_cache"],
            inputs_cpu["is_filling"],
            inputs_cpu["token_index"],
            inputs_cpu["cache_position"],
        )

    # --- Spyre ---
    spyre_inputs = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in inputs_cpu.items()
    }
    with torch.no_grad():
        spyre_h, spyre_kc, spyre_vc = compiled_fn(
            spyre_inputs["hidden_states"],
            spyre_inputs["selected_freqs"],
            spyre_inputs["attn_mask"],
            spyre_inputs["key_cache"],
            spyre_inputs["value_cache"],
            spyre_inputs["is_filling"],
            spyre_inputs["token_index"],
            spyre_inputs["cache_position"],
        )

    spyre_h_cpu = spyre_h.to("cpu")

    # --- Metrics ---
    diff = (cpu_h - spyre_h_cpu).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cpu_nan": cpu_h.isnan().any().item(),
        "spyre_nan": spyre_h_cpu.isnan().any().item(),
        "cpu_shape": list(cpu_h.shape),
        "spyre_shape": list(spyre_h_cpu.shape),
    }


def test_model(model_key):
    """Run per-layer comparison for one model. Returns list of result dicts."""
    import importlib
    from transformers import AutoModelForCausalLM

    info = MODEL_REGISTRY[model_key]
    print(f"\n{'='*70}")
    print(f"  {info['name']}: creating tiny model with random weights")
    print(f"{'='*70}")

    config = info["config_fn"]()
    print(f"  Config: {config.num_hidden_layers} layers, "
          f"hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads}")

    # Create model with random weights
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config).to(torch.float16)
    model.eval()
    model.requires_grad_(False)

    # Prepare adapter (patches RMSNorm, creates compiled blocks, etc.)
    adapter = importlib.import_module(info["adapter"])
    print(f"  Preparing adapter ...")
    adapter.prepare_for_spyre(model)

    num_blocks = len(model._spyre_compiled_blocks)

    # --- Phase A: CPU runs ---
    print(f"  Phase A: running {num_blocks} blocks on CPU ...")
    cpu_results = {}
    for layer_idx in range(num_blocks):
        compiled_block = model._spyre_compiled_blocks[layer_idx]
        uncompiled = getattr(compiled_block, "_orig_mod", compiled_block)
        for mode in ("prefill", "decode"):
            seed = 42 + layer_idx * 100 + (0 if mode == "prefill" else 1)
            inputs = make_inputs(config, mode, seed, device="cpu")
            with torch.no_grad():
                h, kc, vc = uncompiled(
                    inputs["hidden_states"], inputs["selected_freqs"],
                    inputs["attn_mask"], inputs["key_cache"],
                    inputs["value_cache"], inputs["is_filling"],
                    inputs["token_index"], inputs["cache_position"],
                )
            cpu_results[(layer_idx, mode)] = h.clone()

    # --- Phase B: Move model to Spyre, run compiled blocks ---
    print(f"  Moving model to Spyre ...")
    model.to(DEVICE)
    print(f"  Phase B: running {num_blocks} blocks on Spyre ...")

    results = []
    for layer_idx in range(num_blocks):
        compiled_block = model._spyre_compiled_blocks[layer_idx]
        for mode in ("prefill", "decode"):
            seed = 42 + layer_idx * 100 + (0 if mode == "prefill" else 1)
            spyre_inputs = make_inputs(
                config, mode, seed, device=DEVICE,
            )

            try:
                print(f"    Layer {layer_idx} {mode} ...", end=" ", flush=True)
                with torch.no_grad():
                    spyre_h, _, _ = compiled_block(
                        spyre_inputs["hidden_states"],
                        spyre_inputs["selected_freqs"],
                        spyre_inputs["attn_mask"],
                        spyre_inputs["key_cache"],
                        spyre_inputs["value_cache"],
                        spyre_inputs["is_filling"],
                        spyre_inputs["token_index"],
                        spyre_inputs["cache_position"],
                    )
                spyre_h_cpu = spyre_h.to("cpu")
                cpu_h = cpu_results[(layer_idx, mode)]

                diff = (cpu_h - spyre_h_cpu).abs()
                r = {
                    "model": info["name"],
                    "layer": layer_idx,
                    "mode": mode,
                    "shape": (
                        f"[1,{SEQ_LEN if mode == 'prefill' else 1}"
                        f",{config.hidden_size}]"
                    ),
                    "max_abs_diff": diff.max().item(),
                    "mean_abs_diff": diff.mean().item(),
                    "cpu_nan": cpu_h.isnan().any().item(),
                    "spyre_nan": spyre_h_cpu.isnan().any().item(),
                    "error": None,
                }
                print(f"max_diff={r['max_abs_diff']:.4f}")
            except Exception as e:
                r = {
                    "model": info["name"],
                    "layer": layer_idx,
                    "mode": mode,
                    "shape": (
                        f"[1,{SEQ_LEN if mode == 'prefill' else 1}"
                        f",{config.hidden_size}]"
                    ),
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "cpu_nan": None,
                    "spyre_nan": None,
                    "error": str(e)[:80],
                }
                print(f"ERROR: {r['error']}")

            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(all_results):
    """Print markdown table of all results."""
    print(f"\n## Per-Layer CPU vs Spyre Block Comparison\n")
    print(f"| Model | Layer | Mode | Shape | Max Diff | Mean Diff "
          f"| CPU NaN | Spyre NaN | Error |")
    print(f"|-------|-------|------|-------|----------|----------- "
          f"|---------|-----------|-------|")
    for r in all_results:
        if r["error"]:
            print(f"| {r['model']} | {r['layer']} | {r['mode']} "
                  f"| {r['shape']} | — | — | — | — | {r['error']} |")
        else:
            nan_c = "Yes" if r["cpu_nan"] else "No"
            nan_s = "Yes" if r["spyre_nan"] else "No"
            print(f"| {r['model']} | {r['layer']} | {r['mode']} "
                  f"| {r['shape']} "
                  f"| {r['max_abs_diff']:.4f} | {r['mean_abs_diff']:.6f} "
                  f"| {nan_c} | {nan_s} | — |")


def print_summary(all_results):
    """Print pass/fail summary per model."""
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    print(f"\n## Summary\n")
    print(f"| Model | Layers Tested | Errors | Max Diff (worst) | Any NaN |")
    print(f"|-------|--------------|--------|------------------|---------|")
    for model, rows in by_model.items():
        n_layers = len(set(r["layer"] for r in rows))
        n_errors = sum(1 for r in rows if r["error"])
        valid = [r for r in rows if r["error"] is None]
        worst = max((r["max_abs_diff"] for r in valid), default=0.0)
        any_nan = any(r.get("spyre_nan") for r in valid)
        print(f"| {model} | {n_layers} | {n_errors} "
              f"| {worst:.4f} | {'Yes' if any_nan else 'No'} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in which:
        which = list(MODEL_REGISTRY.keys())

    all_results = []
    for key in which:
        if key not in MODEL_REGISTRY:
            print(f"Unknown model: {key}. "
                  f"Options: {list(MODEL_REGISTRY.keys())} or 'all'")
            continue
        try:
            results = test_model(key)
            all_results.extend(results)
        except Exception:
            print(f"\n!!! {MODEL_REGISTRY[key]['name']} FAILED:")
            traceback.print_exc()

    if all_results:
        print_table(all_results)
        print_summary(all_results)
