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
Shared utilities for HuggingFace Transformers adapters on Spyre.

Provides RoPE precomputation, RMSNorm patching, LM head padding, mask
construction, KV cache update helpers, and a model-agnostic generate loop.
Per-model adapters import from this module and provide only model-specific
compiled block functions.
"""

import math
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "spyre"
BLOCK_SIZE = 64  # Spyre stick size at fp16 (128 bytes / 2 bytes per element)


# ---------------------------------------------------------------------------
# RoPE: precompute rotation matrices on CPU (FMS approach)
# ---------------------------------------------------------------------------


class PrecomputedRotaryEmbedding(nn.Module):
    """Builds [S, 2, 2, D/2] rotation matrix cache on CPU.

    Returns ``selected_freqs`` [B, L, 2, 2, D/2] on Spyre, indexed by
    position_ids.  The companion ``apply_rope_matmul`` applies the rotation
    without any tensor slicing.
    """

    def __init__(self, original_rope: nn.Module):
        super().__init__()
        self.original = original_rope
        self._freq_cache: Optional[torch.Tensor] = None
        self._cached_len = 0

    def _extend_cache(self, max_len: int):
        if max_len <= self._cached_len:
            return
        target_len = max(max_len, self._cached_len * 2, 2048)
        inv_freq = self.original.inv_freq.to("cpu").float()
        t = torch.arange(target_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq).float()  # [S, D/2]
        scaling = getattr(self.original, "attention_scaling", 1.0)
        self._freq_cache = torch.stack(
            [
                torch.cos(freqs) * scaling,
                -torch.sin(freqs) * scaling,
                torch.sin(freqs) * scaling,
                torch.cos(freqs) * scaling,
            ],
            dim=1,
        ).view(target_len, 2, 2, freqs.shape[1]).contiguous().to(torch.float16)
        self._cached_len = target_len

    def forward(self, hidden_states, position_ids):
        pos_cpu = position_ids.to("cpu")
        max_pos = int(pos_cpu.max().item()) + 1
        self._extend_cache(max_pos)
        selected = self._freq_cache[pos_cpu]  # [B, L, 2, 2, D/2]
        return selected.to(DEVICE)


def apply_rope_matmul(x, selected_freqs):
    """Apply RoPE via matmul with [2,2,D/2] rotation matrix. No slicing.

    Args:
        x: [B, H, L, D] — query or key tensor
        selected_freqs: [B, L, 2, 2, D/2] — rotation matrices

    Returns: [B, H, L, D]
    """
    B, H, L, D = x.shape
    half = D // 2
    x_ = x.transpose(1, 2).reshape(B, L, H, 2, half)
    sf = selected_freqs[:, :, None, :, :, :]
    out = sf.mul(x_.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# KV cache update (inside compiled graph)
# ---------------------------------------------------------------------------


def kv_cache_update(k, v, key_cache, value_cache,
                    is_filling, token_index, cache_position):
    """Update KV cache: cat for expansion, spyre.overwrite for fill.

    All args are tensors; is_filling/token_index/cache_position are Python
    scalars that torch.compile specializes on.
    """
    if is_filling:
        k_slice = k[:, :, token_index:token_index + 1, :]
        v_slice = v[:, :, token_index:token_index + 1, :]
        if key_cache.device.type == "spyre":
            key_cache = torch.ops.spyre.overwrite(
                input=k_slice,
                output=key_cache, dims=[2], offsets=[cache_position],
            )
            value_cache = torch.ops.spyre.overwrite(
                input=v_slice,
                output=value_cache, dims=[2], offsets=[cache_position],
            )
        else:
            # CPU fallback: direct index assignment
            key_cache = key_cache.clone()
            key_cache[:, :, cache_position:cache_position + 1, :] = k_slice
            value_cache = value_cache.clone()
            value_cache[:, :, cache_position:cache_position + 1, :] = v_slice
    else:
        key_cache = torch.cat((key_cache, k), dim=2)
        value_cache = torch.cat((value_cache, v), dim=2)
    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


def patch_rmsnorm(rmsnorm_cls):
    """Patch any RMSNorm class to stay in fp16 (Spyre has no dtype conversion).

    Args:
        rmsnorm_cls: The RMSNorm class to patch (e.g. GraniteRMSNorm, Qwen3RMSNorm).
    """
    def _forward_fp16(self, hidden_states):
        if hidden_states.device.type == "spyre":
            # Spyre path: no dtype conversion, stay in fp16
            variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
            eps = torch.ops.spyre.full(
                (1,), self.variance_epsilon,
                hidden_states.device, torch.float16,
            )
            return self.weight * (hidden_states * torch.rsqrt(variance + eps))
        else:
            # CPU path: use float32 for numerical stability (matches stock HF)
            xf = hidden_states.float()
            variance = (xf * xf).mean(-1, keepdim=True)
            xf = xf * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * xf.to(hidden_states.dtype)

    rmsnorm_cls.forward = _forward_fp16


def pad_lm_head(model):
    """Pad LM head vocab dim to stick-aligned size (+64 for work division)."""
    w = model.lm_head.weight
    vocab = w.shape[0]
    padded = ((vocab + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) + BLOCK_SIZE
    if padded != vocab:
        model.lm_head.weight = nn.Parameter(
            F.pad(w, (0, 0, 0, padded - vocab)), requires_grad=False
        )


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------


def build_prefill_mask(batch_size, padded_len, prompt_offset):
    """Causal mask for prefill, masking left-padding columns."""
    mask = torch.zeros((batch_size, 1, padded_len, padded_len), dtype=torch.float16)
    mask[:, :, :, :prompt_offset] = -torch.inf
    for i in range(padded_len):
        mask[:, :, i, i + 1:] = -torch.inf
    return mask


def build_expansion_mask(batch_size, block_size, total_cache_len, prompt_offset):
    """Causal mask for an expansion decode step."""
    mask = torch.zeros(
        (batch_size, 1, block_size, total_cache_len), dtype=torch.float16
    )
    mask[:, :, :, :prompt_offset] = -torch.inf
    for j in range(block_size):
        attend_up_to = total_cache_len - block_size + j + 1
        mask[:, :, j, attend_up_to:] = -torch.inf
    return mask


# ---------------------------------------------------------------------------
# Model-agnostic load + generate
# ---------------------------------------------------------------------------


def load_model_common(model_path, prepare_fn, dtype=torch.float16):
    """Load an HF model, apply Spyre adaptations, move to device.

    Args:
        model_path: HF model path or local directory.
        prepare_fn: Model-specific ``prepare_for_spyre(model)`` callable.
        dtype: Weight dtype (default fp16).
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map="cpu",
    )
    model.eval()
    model.requires_grad_(False)
    prepare_fn(model)
    print("Moving model to Spyre ...")
    model.to(DEVICE)
    print("Model ready.")
    return model


def generate(run_forward_fn: Callable, model, tokenizer, prompts,
             max_new_tokens=128, do_sample=False, temperature=1.0,
             top_k=50, timing=False):
    """Model-agnostic generation with padded 64-block decode.

    Args:
        run_forward_fn: ``fn(model, input_ids, position_ids, attn_mask,
            key_caches, value_caches, is_filling, token_index,
            cache_position) -> logits``
        model: Prepared HF model on Spyre.
        tokenizer: HF tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Max tokens to generate.
        do_sample: Sampling vs greedy.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        timing: Print per-token latency.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    batch_size = input_ids.shape[0]
    prompt_length = input_ids.shape[1]

    # Left-pad to BLOCK_SIZE multiple
    padded_len = math.ceil(prompt_length / BLOCK_SIZE) * BLOCK_SIZE
    prompt_offset = padded_len - prompt_length
    if prompt_offset > 0:
        pad = input_ids.new_zeros((batch_size, prompt_offset))
        input_ids = torch.cat([pad, input_ids], dim=1)

    position_ids = torch.zeros((batch_size, padded_len), dtype=torch.long)
    position_ids[:, prompt_offset:] = torch.arange(prompt_length)

    # Initialize empty KV caches
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
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

    # Decode state
    result = input_ids.clone()
    current_cache_len = padded_len
    tokens_in_block = BLOCK_SIZE - 1
    decode_pos = None
    fill_mask_device = None

    times_list = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    num_generated = 0

    for i in range(max_new_tokens):
        t0 = time.time()

        if i == 0:
            # --- PREFILL ---
            prefill_mask = build_prefill_mask(
                batch_size, padded_len, prompt_offset
            )
            logits = run_forward_fn(
                model, input_ids.to(DEVICE), position_ids.to(DEVICE),
                prefill_mask.to(DEVICE), key_caches, value_caches,
                is_filling=False, token_index=0, cache_position=0,
            )
            logits_cpu = logits.to("cpu")
            next_logits = logits_cpu[0, -1, :]
            current_cache_len = padded_len
            # Initialize decode_pos so that after the first +BLOCK_SIZE
            # increment (which happens BEFORE the first expansion forward
            # call), position 0 of the block gets position_id =
            # prompt_length.  Since the expansion code does
            #   decode_pos = decode_pos + BLOCK_SIZE
            # we need: initial[j] + BLOCK_SIZE = prompt_length + j
            # i.e.  initial[j] = prompt_length + j - BLOCK_SIZE
            decode_pos = torch.zeros(
                (batch_size, BLOCK_SIZE), dtype=torch.long
            )
            for j in range(BLOCK_SIZE):
                decode_pos[:, j] = prompt_length + j - BLOCK_SIZE

        else:
            is_filling = tokens_in_block > 0
            next_input = result[:, -BLOCK_SIZE:].to(DEVICE)

            if is_filling:
                fill_pos = (
                    current_cache_len - BLOCK_SIZE + tokens_in_block
                )
                logits = run_forward_fn(
                    model, next_input, decode_pos.to(DEVICE),
                    fill_mask_device, key_caches, value_caches,
                    is_filling=True,
                    token_index=tokens_in_block,
                    cache_position=fill_pos,
                )
                logits_cpu = logits.to("cpu")
                grab_idx = BLOCK_SIZE - tokens_in_block
                next_logits = logits_cpu[0, -grab_idx, :]

            else:
                current_cache_len += BLOCK_SIZE
                decode_pos = decode_pos + BLOCK_SIZE
                exp_mask = build_expansion_mask(
                    batch_size, BLOCK_SIZE, current_cache_len, prompt_offset
                )
                logits = run_forward_fn(
                    model, next_input, decode_pos.to(DEVICE),
                    exp_mask.to(DEVICE), key_caches, value_caches,
                    is_filling=False, token_index=0, cache_position=0,
                )
                logits_cpu = logits.to("cpu")
                next_logits = logits_cpu[0, -BLOCK_SIZE, :]
                fill_mask_device = exp_mask.to(DEVICE)

        # Token selection (CPU)
        if do_sample:
            scaled = next_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < v[-1]] = -torch.inf
            probs = F.softmax(scaled, dim=-1)
            next_val = torch.multinomial(probs.unsqueeze(0), num_samples=1)
        else:
            next_val = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)

        if timing:
            times_list.append(time.time() - t0)

        # Place token in result (FMS logic)
        if tokens_in_block == BLOCK_SIZE - 1:
            result = F.pad(result, (0, BLOCK_SIZE))
        tokens_in_block = (tokens_in_block + 1) % BLOCK_SIZE
        grab_idx = (
            (BLOCK_SIZE - tokens_in_block) if tokens_in_block > 0 else BLOCK_SIZE
        )
        result[:, -grab_idx] = next_val.squeeze()
        num_generated += 1

        if eos_token_id is not None and next_val.item() == eos_token_id:
            break

    # Timing
    if timing and times_list:
        print(f"\nFirst-token latency: {times_list[0]*1000:.3f} ms")
        if len(times_list) > 1:
            avg = sum(times_list[1:]) / len(times_list[1:])
            print(f"Avg next-token latency: {avg*1000:.3f} ms")
        print(
            "Per-token: "
            + ", ".join(f"{t*1000:.1f}" for t in times_list)
            + " ms"
        )

    # Decode text — use num_generated to extract exactly the tokens we placed,
    # avoiding the old nonzero heuristic which fails when token_id 0 is valid.
    # Generated tokens are scattered across blocks in the result buffer.
    # Collect them by walking the block structure.
    all_gen_ids = []
    block_start = padded_len
    remaining = num_generated
    while remaining > 0:
        # First token in a block is at block_start, then block_start+1, etc.
        take = min(remaining, BLOCK_SIZE)
        for j in range(take):
            all_gen_ids.append(result[0, block_start + j].item())
        remaining -= take
        block_start += BLOCK_SIZE

    results = []
    for b in range(batch_size):
        gen_ids = torch.tensor(all_gen_ids)
        if eos_token_id is not None:
            eos_pos = (gen_ids == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0].item()]
        results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    return results
