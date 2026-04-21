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
HuggingFace Transformers adapter for Phi-3/Phi-4 models on Spyre.

Phi-3 differences from Granite:
- Combined QKV projection (split at prepare time)
- Combined gate+up MLP (split at prepare time)
- Partial RoPE: only ``partial_rotary_factor`` of head_dim is rotated.
  Handled by padding the rotation matrix with identity entries so that
  ``apply_rope_matmul`` on full head_dim passes through non-rotated dims.
- No embedding/residual/logits multipliers

Usage::

    from torch_spyre.adapters.hf_phi3 import load_model, generate
    from transformers import AutoTokenizer

    model = load_model("microsoft/Phi-4-mini-instruct")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
    outputs = generate(model, tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_spyre.adapters.hf_common import (
    apply_rope_matmul,
    generate as _generate,
    kv_cache_update,
    load_model_common,
    pad_lm_head,
    patch_rmsnorm,
    DEVICE,
)


# ---------------------------------------------------------------------------
# RoPE with identity padding for partial rotary
# ---------------------------------------------------------------------------


class PartialPrecomputedRotaryEmbedding(nn.Module):
    """Like PrecomputedRotaryEmbedding, but pads with identity for non-rotated dims.

    For partial_rotary_factor < 1.0, the inv_freq covers only ``rope_dim/2``
    frequencies. We pad the ``[S, 2, 2, rope_dim/2]`` rotation matrix to
    ``[S, 2, 2, head_dim/2]`` using identity entries ``[[1,0],[0,1]]`` so that
    ``apply_rope_matmul`` on the full head_dim passes through non-rotated dims.
    """

    def __init__(self, original_rope, head_dim):
        super().__init__()
        self.original = original_rope
        self.head_dim = head_dim
        self._freq_cache = None
        self._cached_len = 0

    def _extend_cache(self, max_len):
        if max_len <= self._cached_len:
            return
        target_len = max(max_len, self._cached_len * 2, 2048)
        inv_freq = self.original.inv_freq.to("cpu").float()
        rope_half = inv_freq.shape[0]  # rope_dim / 2
        full_half = self.head_dim // 2  # head_dim / 2
        pad_half = full_half - rope_half

        t = torch.arange(target_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq).float()  # [S, rope_half]
        scaling = getattr(self.original, "attention_scaling", 1.0)

        # Build rotation matrix [S, 2, 2, rope_half]
        rot = torch.stack([
            torch.cos(freqs) * scaling,
            -torch.sin(freqs) * scaling,
            torch.sin(freqs) * scaling,
            torch.cos(freqs) * scaling,
        ], dim=1).view(target_len, 2, 2, rope_half)

        if pad_half > 0:
            # Identity matrix [[1,0],[0,1]] for non-rotated dims
            ident = torch.zeros(target_len, 2, 2, pad_half)
            ident[:, 0, 0, :] = 1.0  # cos = 1
            ident[:, 1, 1, :] = 1.0  # cos = 1
            # -sin = 0, sin = 0 already
            rot = torch.cat([rot, ident], dim=-1)  # [S, 2, 2, full_half]

        self._freq_cache = rot.contiguous().to(torch.float16)
        self._cached_len = target_len

    def forward(self, hidden_states, position_ids):
        pos_cpu = position_ids.to("cpu")
        max_pos = int(pos_cpu.max().item()) + 1
        self._extend_cache(max_pos)
        selected = self._freq_cache[pos_cpu]
        return selected.to(DEVICE)


# ---------------------------------------------------------------------------
# Weight splitting
# ---------------------------------------------------------------------------


def _chunk_lm_head(model, num_chunks=8):
    """Split LM head weight into N chunks along vocab dim.

    Large vocab (200K+) exceeds Spyre's per-core 256 MB EAR limit.
    We replace the single lm_head with N smaller nn.Linear modules.
    Each chunk processes vocab_size/N output dims.
    """
    w = model.lm_head.weight  # [vocab, hidden]
    vocab, hidden = w.shape
    chunk_size = (vocab + num_chunks - 1) // num_chunks

    STICK = 64
    chunks = nn.ModuleList()
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, vocab)
        w_chunk = w[start:end].clone()
        # Pad to stick-aligned size
        sz = w_chunk.shape[0]
        padded_sz = ((sz + STICK - 1) // STICK) * STICK
        if padded_sz != sz:
            w_chunk = F.pad(w_chunk, (0, 0, 0, padded_sz - sz))
        chunk = nn.Linear(hidden, padded_sz, bias=False)
        chunk.weight = nn.Parameter(w_chunk, requires_grad=False)
        chunks.append(chunk)

    model._spyre_lm_head_chunks = chunks


def _split_fused_qkv(attn, num_q, num_kv, head_dim):
    """Split fused qkv_proj into separate q/k/v projections."""
    w = attn.qkv_proj.weight
    q_dim = num_q * head_dim
    k_dim = num_kv * head_dim
    hidden = w.shape[1]

    def _mk(w_data, out_dim):
        p = nn.Linear(hidden, out_dim, bias=False)
        p.weight = nn.Parameter(w_data.clone(), requires_grad=False)
        return p

    return (
        _mk(w[:q_dim], q_dim),
        _mk(w[q_dim:q_dim + k_dim], k_dim),
        _mk(w[q_dim + k_dim:], k_dim),
    )


def _split_fused_mlp(mlp):
    """Split fused gate_up_proj into separate gate/up projections."""
    w = mlp.gate_up_proj.weight
    half = w.shape[0] // 2
    hidden = w.shape[1]

    def _mk(w_data, out_dim):
        p = nn.Linear(hidden, out_dim, bias=False)
        p.weight = nn.Parameter(w_data.clone(), requires_grad=False)
        return p

    return _mk(w[:half], half), _mk(w[half:], half)


# ---------------------------------------------------------------------------
# Compiled block
# ---------------------------------------------------------------------------


def _make_compiled_block(layer, q_proj, k_proj, v_proj, gate_proj, up_proj,
                         head_dim):
    """Compiled block for Phi-3. Full head_dim RoPE via identity-padded freqs."""
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    down_proj = layer.mlp.down_proj
    act_fn = layer.mlp.activation_fn
    o_proj = layer.self_attn.o_proj
    scaling = layer.self_attn.scaling

    def block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache,
                      is_filling, token_index, cache_position):
        residual = hidden_states
        h = input_ln(hidden_states)
        bsz, seq_len, _ = h.shape

        # Separate Q/K/V projections (split at prepare time)
        q = q_proj(h).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        k = k_proj(h).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        v = v_proj(h).view(bsz, seq_len, -1, head_dim).transpose(1, 2)

        # RoPE on full head_dim — identity-padded freqs pass through
        # non-rotated dims unchanged
        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k, v, key_cache, value_cache,
            is_filling, token_index, cache_position,
        )

        attn_out = F.scaled_dot_product_attention(
            q, key_cache, value_cache,
            attn_mask=attn_mask, dropout_p=0.0, scale=scaling, enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = o_proj(attn_out)

        h = residual + attn_out

        # MLP with pre-split gate/up
        residual = h
        h = post_attn_ln(h)
        h = down_proj(act_fn(gate_proj(h)) * up_proj(h))
        h = residual + h

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


# ---------------------------------------------------------------------------
# Forward / prepare / load / generate
# ---------------------------------------------------------------------------


def _run_forward(model, input_ids, position_ids, attn_mask,
                 key_caches, value_caches,
                 is_filling, token_index, cache_position):
    h = model.model.embed_tokens(input_ids)
    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h, selected_freqs, attn_mask,
            key_caches[i], value_caches[i],
            is_filling, token_index, cache_position,
        )

    h = model.model.norm(h)

    # Chunked LM head: large vocab (200K+) exceeds Spyre's per-core EAR limit.
    # Split into N chunks, run each on Spyre, cat on CPU.
    logits_parts = []
    for lm_chunk in model._spyre_lm_head_chunks:
        logits_parts.append(lm_chunk(h).to("cpu"))
    logits = torch.cat(logits_parts, dim=-1)
    return logits


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Phi-3 model in-place."""
    from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm

    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads

    # RoPE with identity padding for partial rotary
    model._spyre_rope = PartialPrecomputedRotaryEmbedding(
        model.model.rotary_emb, hd
    )
    patch_rmsnorm(Phi3RMSNorm)

    # Chunk LM head for large vocab models (200K+ vocab exceeds EAR limit)
    # Each chunk is stick-padded internally. Don't call pad_lm_head.
    _chunk_lm_head(model, num_chunks=8)

    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads

    # Split fused weights, register as submodules
    model._spyre_q_projs = nn.ModuleList()
    model._spyre_k_projs = nn.ModuleList()
    model._spyre_v_projs = nn.ModuleList()
    model._spyre_gate_projs = nn.ModuleList()
    model._spyre_up_projs = nn.ModuleList()

    for layer in model.model.layers:
        q, k, v = _split_fused_qkv(layer.self_attn, num_q, num_kv, hd)
        model._spyre_q_projs.append(q)
        model._spyre_k_projs.append(k)
        model._spyre_v_projs.append(v)

        gate, up = _split_fused_mlp(layer.mlp)
        model._spyre_gate_projs.append(gate)
        model._spyre_up_projs.append(up)

    model._spyre_compiled_blocks = [
        _make_compiled_block(layer, qp, kp, vp, gp, up, hd)
        for layer, qp, kp, vp, gp, up in zip(
            model.model.layers,
            model._spyre_q_projs,
            model._spyre_k_projs,
            model._spyre_v_projs,
            model._spyre_gate_projs,
            model._spyre_up_projs,
        )
    ]


def load_model(model_path, dtype=torch.float16):
    """Load Phi-3/Phi-4 model for Spyre."""
    return load_model_common(model_path, prepare_for_spyre, dtype)


def generate(model, tokenizer, prompts, **kwargs):
    """Generate text with Phi-3/Phi-4 on Spyre."""
    return _generate(_run_forward, model, tokenizer, prompts, **kwargs)
