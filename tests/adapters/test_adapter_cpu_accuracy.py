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
CPU accuracy test: compare adapter forward passes against stock HF on CPU.

Tests prefill (full sequence) and 4 autoregressive decode steps, comparing
logits and greedy token selections at each step.

Usage:
    python tests/adapters/test_adapter_cpu_accuracy.py [granite|qwen3|granite4|smollm3]

Requires: transformers, torch (2.x), sentencepiece
"""

import importlib
import importlib.util
import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import adapter modules without triggering torch_spyre.__init__
# We load them as standalone modules via importlib.
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "torch_spyre", "adapters")


def _load_adapter_module(filename):
    """Load an adapter .py file as a module, bypassing torch_spyre.__init__."""
    filepath = os.path.join(ADAPTERS_DIR, filename)
    mod_name = f"_adapter_{filename.replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    return mod, spec


# Step 1: load hf_common first so adapters can import from it
_common_path = os.path.join(ADAPTERS_DIR, "hf_common.py")
_common_spec = importlib.util.spec_from_file_location(
    "torch_spyre.adapters.hf_common", _common_path
)
_common_mod = importlib.util.module_from_spec(_common_spec)
# Patch DEVICE to cpu BEFORE executing the module
sys.modules["torch_spyre.adapters.hf_common"] = _common_mod
_common_spec.loader.exec_module(_common_mod)
_common_mod.DEVICE = "cpu"

# Also register torch_spyre.adapters so relative imports work
sys.modules.setdefault("torch_spyre", type(sys)("torch_spyre"))
sys.modules.setdefault("torch_spyre.adapters", type(sys)("torch_spyre.adapters"))


def load_adapter(filename):
    """Load an adapter module by filename."""
    mod_name = f"torch_spyre.adapters.{filename.replace('.py', '')}"
    filepath = os.path.join(ADAPTERS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# HF reference: token-by-token greedy with DynamicCache
# ---------------------------------------------------------------------------

def hf_greedy_steps(model, input_ids, num_decode=4):
    """Run stock HF model for prefill + N greedy decode steps.

    Returns list of dicts with logits/token for each step.
    Step 0 = prefill, steps 1..N = decode.
    """
    from transformers import DynamicCache

    results = []
    past = DynamicCache()
    ids = input_ids.clone()
    seq_len = ids.shape[1]

    for step in range(num_decode + 1):
        if step == 0:
            position_ids = torch.arange(seq_len).unsqueeze(0)
        else:
            position_ids = torch.tensor([[seq_len + step - 1]])

        with torch.no_grad():
            out = model(
                input_ids=ids,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )

        logits = out.logits
        last_logits = logits[0, -1, :].float()
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})

        past = out.past_key_values
        ids = torch.tensor([[token]])

    return results


# ---------------------------------------------------------------------------
# Adapter: prefill + decode using adapter _run_forward with KV cache
# ---------------------------------------------------------------------------

def adapter_greedy_steps(run_forward_fn, model, input_ids, num_decode=4):
    """Run adapter forward for prefill + N greedy decode steps on CPU."""
    results = []
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    vocab_size = model.config.vocab_size

    # Empty KV caches
    key_caches = [
        torch.empty(batch_size, num_kv_heads, 0, head_dim, dtype=torch.float16)
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.empty(batch_size, num_kv_heads, 0, head_dim, dtype=torch.float16)
        for _ in range(num_layers)
    ]

    # --- Prefill ---
    position_ids = torch.arange(seq_len).unsqueeze(0)
    causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
    for i in range(seq_len):
        causal_mask[:, :, i, i + 1:] = -torch.inf

    with torch.no_grad():
        logits = run_forward_fn(
            model, input_ids, position_ids, causal_mask,
            key_caches, value_caches,
            is_filling=False, token_index=0, cache_position=0,
        )

    last_logits = logits[0, -1, :].float()[:vocab_size]
    token = last_logits.argmax().item()
    results.append({"logits": last_logits, "token": token, "step": 0})

    cache_len = seq_len

    # --- Decode steps (expand mode) ---
    for step in range(1, num_decode + 1):
        next_ids = torch.tensor([[token]])
        next_pos = torch.tensor([[seq_len + step - 1]])
        total_len = cache_len + 1
        decode_mask = torch.zeros((1, 1, 1, total_len), dtype=torch.float16)

        with torch.no_grad():
            logits = run_forward_fn(
                model, next_ids, next_pos, decode_mask,
                key_caches, value_caches,
                is_filling=False, token_index=0, cache_position=0,
            )

        last_logits = logits[0, -1, :].float()[:vocab_size]
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})
        cache_len += 1

    return results


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def run_model_test(model_name, model_path, adapter_filename, num_decode=4):
    """Load model, run HF ref vs adapter, return comparison list."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_mod = load_adapter(adapter_filename)
    run_forward_fn = adapter_mod._run_forward
    prepare_fn = adapter_mod.prepare_for_spyre

    print(f"\n{'='*70}")
    print(f"  {model_name}: loading {model_path}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)

    prompt = "The capital of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"  Prompt: {prompt!r}  ({input_ids.shape[1]} tokens)")

    # --- HF reference (BEFORE patching) ---
    print("  Running HF reference ...")
    hf_results = hf_greedy_steps(model, input_ids, num_decode=num_decode)

    # --- Adapter ---
    print("  Preparing adapter ...")
    prepare_fn(model)

    # Unwrap torch.compile for CPU (skip compilation overhead)
    if hasattr(model, "_spyre_compiled_blocks"):
        unwrapped = []
        for cb in model._spyre_compiled_blocks:
            orig = getattr(cb, "_orig_mod", None)
            unwrapped.append(orig if orig is not None else cb)
        model._spyre_compiled_blocks = unwrapped

    print("  Running adapter ...")
    adapter_results = adapter_greedy_steps(
        run_forward_fn, model, input_ids, num_decode=num_decode,
    )

    # --- Compare ---
    comparisons = []
    for hf_r, ad_r in zip(hf_results, adapter_results):
        step = hf_r["step"]
        h_logits = hf_r["logits"]
        a_logits = ad_r["logits"]

        abs_diff = (h_logits - a_logits).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        h_top5 = h_logits.topk(5).indices.tolist()
        a_top5 = a_logits.topk(5).indices.tolist()
        top1_match = h_top5[0] == a_top5[0]
        top5_overlap = len(set(h_top5) & set(a_top5))

        step_label = "prefill" if step == 0 else f"decode-{step}"
        hf_tok_str = tokenizer.decode([hf_r["token"]])
        ad_tok_str = tokenizer.decode([ad_r["token"]])

        comparisons.append({
            "step": step_label,
            "hf_token": hf_r["token"],
            "hf_tok_str": hf_tok_str,
            "adapter_token": ad_r["token"],
            "adapter_tok_str": ad_tok_str,
            "top1_match": top1_match,
            "top5_overlap": top5_overlap,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        })

    return comparisons, tokenizer


def print_results_table(model_name, comparisons):
    """Print formatted results table. Returns True if all top-1 match."""
    print(f"\n  {model_name} Results")
    print(f"  {'Step':<12} {'HF Token':<20} {'Adapter Token':<20} "
          f"{'Top1':<6} {'Top5':<5} {'MaxDiff':<10} {'MeanDiff':<10}")
    print(f"  {'-'*12} {'-'*20} {'-'*20} "
          f"{'-'*6} {'-'*5} {'-'*10} {'-'*10}")
    all_match = True
    for c in comparisons:
        match_str = "OK" if c["top1_match"] else "FAIL"
        if not c["top1_match"]:
            all_match = False
        hf_str = f"{c['hf_token']:>6} {c['hf_tok_str']!r:<12}"
        ad_str = f"{c['adapter_token']:>6} {c['adapter_tok_str']!r:<12}"
        print(f"  {c['step']:<12} {hf_str:<20} {ad_str:<20} "
              f"{match_str:<6} {c['top5_overlap']}/5  "
              f"{c['max_diff']:<10.4f} {c['mean_diff']:<10.6f}")
    return all_match


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "qwen3": {
        "name": "Qwen3 0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "adapter": "hf_qwen3.py",
    },
    "granite": {
        "name": "Granite 3.3 2B",
        "path": "ibm-granite/granite-3.3-2b-instruct",
        "adapter": "hf_granite.py",
    },
    "granite4": {
        "name": "Granite 4.0 Tiny",
        "path": "ibm-granite/granite-4.0-tiny-preview",
        "adapter": "hf_granitemoehybrid.py",
    },
    "smollm3": {
        "name": "SmolLM3 3B",
        "path": "HuggingFaceTB/SmolLM3-3B-Base",
        "adapter": "hf_smollm3.py",
    },
}


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    all_results = {}
    for key in which:
        if key not in MODELS:
            print(f"Unknown model: {key}. Options: {list(MODELS.keys())}")
            continue
        m = MODELS[key]
        try:
            comps, _ = run_model_test(
                m["name"], m["path"], m["adapter"], num_decode=4,
            )
            ok = print_results_table(m["name"], comps)
            all_results[key] = {"comparisons": comps, "all_match": ok}
        except Exception as e:
            print(f"\n!!! {m['name']} FAILED:")
            traceback.print_exc()
            all_results[key] = {"error": str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for key in which:
        if key not in MODELS or key not in all_results:
            continue
        name = MODELS[key]["name"]
        res = all_results[key]
        if "error" in res:
            print(f"  {name:<22} ERROR: {res['error']}")
        else:
            status = "PASS" if res["all_match"] else "FAIL"
            print(f"  {name:<22} {status}")
    print(f"{'='*70}")
