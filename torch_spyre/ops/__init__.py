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

from typing import Optional
import torch


def linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    feature_map: str = "elu",
    scale: Optional[float] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear attention operation.

    Computes linear attention using feature map representations phi(Q) and phi(K).
    Supports both non-causal (O(N) global associative product) and causal
    (autoregressive prefix scan) modes, with optional recurrent initial state.

    Args:
        query: Tensor of shape [B, Hq, Lq, Dk]
        key: Tensor of shape [B, Hkv, Lkv, Dk]
        value: Tensor of shape [B, Hkv, Lkv, Dv]
        initial_state: Optional recurrent KV state [B, Hq, Dk, Dv]
        is_causal: Whether to apply causal autoregressive masking
        feature_map: Feature map function ("elu", "relu", "silu", "identity")
        scale: Scaling factor for query (defaults to 1 / sqrt(Dk))
        eps: Epsilon constant for denominator normalization

    Returns:
        output: Output tensor of shape [B, Hq, Lq, Dv]
        final_state: Final recurrent KV state of shape [B, Hq, Dk, Dv]
    """
    return torch.ops.spyre.linear_attention(
        query,
        key,
        value,
        initial_state=initial_state,
        is_causal=is_causal,
        feature_map=feature_map,
        scale=scale,
        eps=eps,
    )


__all__ = ["linear_attention"]
