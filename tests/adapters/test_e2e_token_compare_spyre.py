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
E2E token-level comparison: HF stock forward (CPU) vs adapter forward (Spyre).

For each model, runs prefill + 4 greedy decode steps on both CPU (stock HF)
and Spyre (adapter), comparing logits and greedy tokens at each step.

NOTE: Spyre currently has known correctness issues. This test is expected
to show mismatches now but will pass once hardware fixes land (~1 week).

Usage (on Spyre pod):
    python3 test_e2e_token_compare_spyre.py [qwen3|granite]
"""

import importlib
import math
import sys
import traceback

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401 — registers Spyre device

from torch_spyre.adapters.hf_common import BLOCK_SIZE

DEVICE = "spyre"

MODEL_REGISTRY = {
    "qwen3": {
        "name": "Qwen3 0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "adapter": "torch_spyre.adapters.hf_qwen3",
    },
    "granite": {
        "name": "Granite 3.3 2B",
        "path": "ibm-granite/granite-3.3-2b-instruct",
        "adapter": "torch_spyre.adapters.hf_granite",
    },
}


# ---------------------------------------------------------------------------
# HF reference: stock forward on CPU with DynamicCache
# ---------------------------------------------------------------------------

def hf_greedy_steps(model, input_ids, num_decode=4):
    """Run stock HF model for prefill + N decode steps on CPU."""
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
                input_ids=ids, position_ids=position_ids,
                past_key_values=past, use_cache=True,
            )

        logits = out.logits[0, -1, :].float()
        token = logits.argmax().item()
        results.append({"logits": logits, "token": token, "step": step})
        past = out.past_key_values
        ids = torch.tensor([[token]])

    return results


# ---------------------------------------------------------------------------
# Adapter forward on Spyre
# ---------------------------------------------------------------------------

def adapter_greedy_steps(run_forward_fn, model, input_ids, num_decode=4):
    """Run adapter forward on Spyre for prefill + N decode steps."""
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    vocab_size = model.config.vocab_size

    # Pad to BLOCK_SIZE
    padded_len = math.ceil(seq_len / BLOCK_SIZE) * BLOCK_SIZE
    prompt_offset = padded_len - seq_len
    if prompt_offset > 0:
        pad = input_ids.new_zeros((batch_size, prompt_offset))
        padded_ids = torch.cat([pad, input_ids], dim=1)
    else:
        padded_ids = input_ids

    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    position_ids[:, prompt_offset:] = torch.arange(seq_len)

    key_caches = [
        torch.empty(batch_size, num_kv_heads, 0, head_dim,
                     dtype=torch.float16, device=DEVICE)
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.empty(batch_size, num_kv_heads, 0, head_dim,
                     dtype=torch.float16, device=DEVICE)
        for _ in range(num_layers)
    ]

    results = []

    # --- Prefill ---
    from torch_spyre.adapters.hf_common import build_prefill_mask

    prefill_mask = build_prefill_mask(batch_size, padded_len, prompt_offset)

    with torch.no_grad():
        logits = run_forward_fn(
            model, padded_ids.to(DEVICE), position_ids.to(DEVICE),
            prefill_mask.to(DEVICE), key_caches, value_caches,
            is_filling=False, token_index=0, cache_position=0,
        )
    logits_cpu = logits.to("cpu")[0, -1, :].float()[:vocab_size]
    token = logits_cpu.argmax().item()
    results.append({"logits": logits_cpu, "token": token, "step": 0})

    # --- Decode steps (expansion mode for simplicity) ---
    cache_len = padded_len
    # Build decode position tracking
    decode_pos = torch.zeros((batch_size, BLOCK_SIZE), dtype=torch.long)
    for j in range(BLOCK_SIZE):
        decode_pos[:, j] = seq_len + j - BLOCK_SIZE

    # We generate one token at a time using expansion blocks
    result_buf = padded_ids.clone()
    result_buf = F.pad(result_buf, (0, BLOCK_SIZE))
    result_buf[:, padded_len] = token

    tokens_in_block = 1

    for step in range(1, num_decode + 1):
        # Always use expansion mode (simpler for the comparison test)
        cache_len += BLOCK_SIZE
        decode_pos = decode_pos + BLOCK_SIZE

        from torch_spyre.adapters.hf_common import build_expansion_mask
        exp_mask = build_expansion_mask(
            batch_size, BLOCK_SIZE, cache_len, prompt_offset,
        )

        next_input = result_buf[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits = run_forward_fn(
                model, next_input.to(DEVICE), decode_pos.to(DEVICE),
                exp_mask.to(DEVICE), key_caches, value_caches,
                is_filling=False, token_index=0, cache_position=0,
            )

        logits_cpu = logits.to("cpu")
        # The relevant logit is at the first position of the block
        # (previous token was placed at result_buf[:, padded_len + ...])
        last_logits = logits_cpu[0, -BLOCK_SIZE, :].float()[:vocab_size]
        token = last_logits.argmax().item()
        results.append({"logits": last_logits, "token": token, "step": step})

        # Place token and prepare next block
        result_buf = F.pad(result_buf, (0, BLOCK_SIZE))
        result_buf[:, -BLOCK_SIZE] = token

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(hf_results, adapter_results, tokenizer, model_name):
    """Compare HF vs adapter results, return comparison rows."""
    rows = []
    for hf_r, ad_r in zip(hf_results, adapter_results):
        step = hf_r["step"]
        h_logits = hf_r["logits"]
        a_logits = ad_r["logits"]

        # Trim to common vocab
        min_vocab = min(h_logits.shape[0], a_logits.shape[0])
        h = h_logits[:min_vocab]
        a = a_logits[:min_vocab]

        diff = (h - a).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        h_top1 = h.argmax().item()
        a_top1 = a.argmax().item()
        match = h_top1 == a_top1

        step_label = "prefill" if step == 0 else f"decode-{step}"
        h_str = tokenizer.decode([hf_r["token"]])
        a_str = tokenizer.decode([ad_r["token"]])

        rows.append({
            "model": model_name,
            "step": step_label,
            "hf_token": hf_r["token"],
            "hf_str": h_str,
            "spyre_token": ad_r["token"],
            "spyre_str": a_str,
            "top1_match": match,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "hf_nan": h_logits.isnan().any().item(),
            "spyre_nan": a_logits.isnan().any().item(),
        })
    return rows


def run_model_test(model_key, num_decode=4):
    """Full comparison for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = MODEL_REGISTRY[model_key]
    adapter = importlib.import_module(info["adapter"])

    print(f"\n{'='*70}")
    print(f"  {info['name']}: {info['path']}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    model = AutoModelForCausalLM.from_pretrained(
        info["path"], torch_dtype=torch.float16, device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)

    prompt = "The capital of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"  Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")

    # --- HF reference on CPU (BEFORE patching) ---
    print("  Running HF reference on CPU ...")
    hf_results = hf_greedy_steps(model, input_ids, num_decode=num_decode)

    # --- Adapter on Spyre ---
    print("  Preparing adapter ...")
    adapter.prepare_for_spyre(model)
    print("  Moving model to Spyre ...")
    model.to(DEVICE)
    print("  Running adapter on Spyre ...")
    adapter_results = adapter_greedy_steps(
        adapter._run_forward, model, input_ids, num_decode=num_decode,
    )

    rows = compare_results(hf_results, adapter_results, tokenizer, info["name"])
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(all_rows):
    """Print markdown comparison table."""
    print(f"\n## E2E Token Comparison: HF (CPU) vs Adapter (Spyre)\n")
    print(f"| Model | Step | HF Token | Spyre Token | Match "
          f"| Max Diff | Mean Diff | HF NaN | Spyre NaN |")
    print(f"|-------|------|----------|-------------|-------"
          f"|----------|-----------|--------|-----------|")
    for r in all_rows:
        match = "OK" if r["top1_match"] else "FAIL"
        hf_col = f"{r['hf_token']:>5} {r['hf_str']!r}"
        sp_col = f"{r['spyre_token']:>5} {r['spyre_str']!r}"
        hn = "Yes" if r["hf_nan"] else "No"
        sn = "Yes" if r["spyre_nan"] else "No"
        print(f"| {r['model']} | {r['step']} | {hf_col} | {sp_col} "
              f"| {match} | {r['max_diff']:.4f} | {r['mean_diff']:.6f} "
              f"| {hn} | {sn} |")


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else ["qwen3"]

    all_rows = []
    for key in which:
        if key not in MODEL_REGISTRY:
            print(f"Unknown: {key}. Options: {list(MODEL_REGISTRY.keys())}")
            continue
        try:
            rows = run_model_test(key)
            all_rows.extend(rows)
        except Exception:
            print(f"\n!!! {MODEL_REGISTRY[key]['name']} FAILED:")
            traceback.print_exc()

    if all_rows:
        print_table(all_rows)
        n_match = sum(1 for r in all_rows if r["top1_match"])
        print(f"\nTop-1 agreement: {n_match}/{len(all_rows)} steps")
        if n_match < len(all_rows):
            print("NOTE: Spyre has known correctness issues being fixed.")
