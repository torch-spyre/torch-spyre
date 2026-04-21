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
HuggingFace Transformers adapter for SmolLM3 models on Spyre.

SmolLM3 differences from Granite:
- NoPE layers: some layers skip RoPE entirely (controlled by config.no_rope_layers)
- No embedding/residual/logits multipliers

Usage::

    from torch_spyre.adapters.hf_smollm3 import load_model, generate
    from transformers import AutoTokenizer

    model = load_model("HuggingFaceTB/SmolLM3-3B-Base")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
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


def _make_compiled_block(layer, use_rope):
    """Compiled block for SmolLM3: conditional RoPE per layer."""
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm

    def block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache,
                      is_filling, token_index, cache_position):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

        # SmolLM3: RoPE is conditional per layer (NoPE layers skip it)
        if use_rope:
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
    """SmolLM3 forward: no multipliers."""
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
    """Apply Spyre adaptations to SmolLM3 model in-place."""
    from transformers.models.smollm3.modeling_smollm3 import SmolLM3RMSNorm

    model._spyre_rope = PrecomputedRotaryEmbedding(model.model.rotary_emb)
    patch_rmsnorm(SmolLM3RMSNorm)
    pad_lm_head(model)

    # Determine which layers use RoPE vs NoPE
    # SmolLM3 config has no_rope_layers: list of bools (True = skip RoPE)
    no_rope = getattr(model.config, "no_rope_layers", None)

    model._spyre_compiled_blocks = []
    for idx, layer in enumerate(model.model.layers):
        use_rope = True
        if no_rope is not None and idx < len(no_rope):
            use_rope = bool(no_rope[idx])
        model._spyre_compiled_blocks.append(
            _make_compiled_block(layer, use_rope)
        )


def load_model(model_path, dtype=torch.float16):
    """Load SmolLM3 model for Spyre."""
    return load_model_common(model_path, prepare_for_spyre, dtype)


def generate(model, tokenizer, prompts, **kwargs):
    """Generate text with SmolLM3 on Spyre."""
    return _generate(_run_forward, model, tokenizer, prompts, **kwargs)
