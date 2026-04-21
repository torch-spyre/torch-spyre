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
HuggingFace Transformers adapter for Granite 3.3 models on Spyre.

Usage::

    from torch_spyre.adapters.hf_granite import load_model, generate
    from transformers import AutoTokenizer

    model = load_model("/path/to/granite-3.3-8b-instruct")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/granite-3.3-8b-instruct")
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
    """Compiled block for Granite 3.3: separate QKV, residual multiplier."""
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_attn_ln = layer.post_attention_layernorm
    res_mult = layer.residual_multiplier

    def block_forward(hidden_states, selected_freqs, attn_mask,
                      key_cache, value_cache,
                      is_filling, token_index, cache_position):
        residual = hidden_states
        h = input_ln(hidden_states)

        bsz, seq_len, _ = h.shape
        q = attn.q_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)

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

        h = residual + attn_out * res_mult

        residual = h
        h = post_attn_ln(h)
        h = mlp(h)
        h = residual + h * res_mult

        return h, key_cache, value_cache

    return torch.compile(block_forward, dynamic=False)


def _run_forward(model, input_ids, position_ids, attn_mask,
                 key_caches, value_caches,
                 is_filling, token_index, cache_position):
    """Granite 3.3 forward: embedding * multiplier, blocks, norm, head / scaling."""
    h = model.model.embed_tokens(input_ids)
    h = h * model.model.embedding_multiplier

    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h, selected_freqs, attn_mask,
            key_caches[i], value_caches[i],
            is_filling, token_index, cache_position,
        )

    h = model.model.norm(h)
    logits = model.lm_head(h)
    logits = logits / model.config.logits_scaling
    return logits


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Granite 3.3 model in-place."""
    from transformers.models.granite.modeling_granite import GraniteRMSNorm

    model._spyre_rope = PrecomputedRotaryEmbedding(model.model.rotary_emb)
    patch_rmsnorm(GraniteRMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = [
        _make_compiled_block(layer) for layer in model.model.layers
    ]


def load_model(model_path, dtype=torch.float16):
    """Load Granite 3.3 model for Spyre."""
    return load_model_common(model_path, prepare_for_spyre, dtype)


def generate(model, tokenizer, prompts, **kwargs):
    """Generate text with Granite 3.3 on Spyre."""
    return _generate(_run_forward, model, tokenizer, prompts, **kwargs)
