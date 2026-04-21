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
HuggingFace Transformers adapter for Qwen3 models on Spyre.

Qwen3 differences from Granite:
- Q/K RMSNorm applied before RoPE
- No embedding/residual/logits multipliers
- Standard head_dim**-0.5 scaling

Usage::

    from torch_spyre.adapters.hf_qwen3 import load_model, generate
    from transformers import AutoTokenizer

    model = load_model("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    outputs = generate(model, tokenizer, ["Hello!"], max_new_tokens=32)
"""

import torch
import torch.nn.functional as F

from torch_spyre.adapters.hf_common import (
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    generate as _generate,
    kv_cache_update,
    load_model_common,
    pad_lm_head,
    patch_rmsnorm,
)


def _make_compiled_block(layer):
    """Compiled block for Qwen3: Q/K norm before RoPE, no multipliers."""
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm

    # Qwen3 has per-head RMSNorm on Q and K
    q_norm = attn.q_norm
    k_norm = attn.k_norm

    def block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache,
                      is_filling, token_index, cache_position):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

        # Qwen3-specific: RMSNorm on Q and K before RoPE
        q = q_norm(q)
        k = k_norm(k)

        q = apply_rope_matmul(q, selected_freqs)
        k = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k, v, key_cache, value_cache,
            is_filling, token_index, cache_position,
        )

        attn_out = F.scaled_dot_product_attention(
            q, key_cache, value_cache,
            attn_mask=attn_mask, dropout_p=0.0, scale=attn.scaling, enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out

        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def _run_forward(model, input_ids, position_ids, attn_mask,
                 key_caches, value_caches,
                 is_filling, token_index, cache_position):
    """Qwen3 forward: no embedding multiplier, no logits scaling."""
    h = model.model.embed_tokens(input_ids)

    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h, selected_freqs, attn_mask,
            key_caches[i], value_caches[i],
            is_filling, token_index, cache_position,
        )

    h = model.model.norm(h)
    logits = model.lm_head(h)
    return logits


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Qwen3 model in-place."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    model._spyre_rope = PrecomputedRotaryEmbedding(model.model.rotary_emb)
    patch_rmsnorm(Qwen3RMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer) for layer in model.model.layers
    ]


def load_model(model_path, dtype=torch.float16):
    """Load Qwen3 model for Spyre."""
    return load_model_common(model_path, prepare_for_spyre, dtype)


def generate(model, tokenizer, prompts, **kwargs):
    """Generate text with Qwen3 on Spyre."""
    return _generate(_run_forward, model, tokenizer, prompts, **kwargs)
