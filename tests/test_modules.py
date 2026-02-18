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

import unittest
import copy

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from transformers.models.roberta.modeling_roberta import (
    RobertaSelfAttention,
    RobertaLayer,
    RobertaEncoder,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaAttention,
)
from transformers.models.llama.configuration_llama import LlamaConfig


class TestOps(TestCase):
    def __init__(self, method_name="runTest", methodName="runTest"):
        super().__init__(method_name, methodName)
        self.rtol = 1e-2
        self.atol = 1e-3
        self.dtype = torch.float16

    def test_linear(self):
        with torch.no_grad():
            m = nn.Linear(64, 128, bias=False, dtype=self.dtype)
            m.weight.normal_(0, 0.01)
            m_spyre = copy.deepcopy(m)
            m_spyre.to("spyre")

            x = torch.randn((2, 64), dtype=self.dtype)
            x_spyre = x.to("spyre")

            y = m_spyre(x_spyre).to("cpu")

        torch.testing.assert_close(y, m(x), rtol=self.rtol, atol=self.atol)

    def test_softmax(self):
        m = nn.Softmax(dim=1)
        m_spyre = m.to("spyre")

        x = torch.randn(1024, 256, dtype=self.dtype)
        x_spyre = x.to("spyre")

        with torch.no_grad():
            y = m_spyre(x_spyre).to("cpu")

        torch.testing.assert_close(y, m(x), rtol=self.rtol, atol=self.atol)

    @unittest.skip("Roberta still has issues on Eager path")
    def test_roberta_self_attention(self):
        config = RobertaConfig()
        config.hidden_size = 144
        m = RobertaSelfAttention(config)
        m.eval()
        m.to(self.dtype)
        m_spyre = copy.deepcopy(m)
        m_spyre.to("spyre")

        x = torch.rand(2, 10, config.hidden_size, dtype=self.dtype)
        x_spyre = x.to("spyre")

        with torch.no_grad():
            y = m(x)
            y_spyre = m_spyre(x_spyre)

        torch.testing.assert_close(
            y_spyre[0].cpu(), y[0], rtol=self.rtol, atol=self.atol
        )

    @unittest.skip("Roberta still has issues on Eager path")
    def test_roberta_layer(self):
        config = RobertaConfig()
        config.hidden_size = 144
        m = RobertaLayer(config)
        m.eval()
        # m.dropout = nn.Sequential()
        m.to(self.dtype)
        m_spyre = copy.deepcopy(m)
        m_spyre.to("spyre")

        x = torch.rand(2, 10, config.hidden_size, dtype=self.dtype)
        x_spyre = x.to("spyre")

        with torch.no_grad():
            y = m(x)
            y_spyre = m_spyre(x_spyre)

        torch.testing.assert_close(
            y_spyre[0].cpu(), y[0], rtol=self.rtol, atol=self.atol
        )

    @unittest.skip("Roberta HF has issues")
    def test_roberta_encoder(self):
        config = RobertaConfig()
        config.hidden_size = 144
        m = RobertaEncoder(config)
        m.eval()
        m.to(self.dtype)
        m_spyre = copy.deepcopy(m)
        m_spyre.to("spyre")

        x = torch.randn(10, 384, config.hidden_size, dtype=self.dtype)
        x_spyre = x.to("spyre")

        with torch.no_grad():
            y = m(x)
            y_spyre = m_spyre(x_spyre)

        torch.testing.assert_close(
            y_spyre[0].cpu(), y[0], rtol=self.rtol, atol=self.atol
        )

    @unittest.skip("Llama still has issues on Eager path")
    def test_llama_attention(self):
        config = LlamaConfig()
        T = 1000
        B = 1
        config.hidden_size = 1024

        config.head_dim = config.hidden_size // 32

        llama_model = LlamaModel(config)
        m = LlamaAttention(config, 0)
        m.eval()
        m.to(self.dtype)
        m_spyre = copy.deepcopy(m)
        m_spyre.to("spyre")

        x = torch.rand(B, T, config.hidden_size, dtype=self.dtype)
        x_spyre = x.to("spyre")

        position_ids = torch.arange(T).unsqueeze(0)
        position_embeddings = llama_model.rotary_emb(x, position_ids)
        position_embeddings_spyre = (
            position_embeddings[0].to(self.dtype).to("spyre"),
            position_embeddings[1].to(self.dtype).to("spyre"),
        )

        attention_mask = torch.ones([T, T], dtype=self.dtype)
        attention_mask_spyre = attention_mask.to("spyre")

        with torch.no_grad():
            y = m(
                hidden_states=x,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            y_spyre = m_spyre(
                hidden_states=x_spyre,
                position_embeddings=position_embeddings_spyre,
                attention_mask=attention_mask_spyre,
            )

        torch.testing.assert_close(
            y_spyre[0].cpu(), y[0], rtol=self.rtol, atol=self.atol
        )

    @unittest.skip("Llama still has issues on Eager path")
    def test_llama_layer(self):
        config = LlamaConfig()
        config.hidden_size = 1024
        T = 1000
        B = 1

        llama_model = LlamaModel(config)
        m = LlamaDecoderLayer(config, 0)
        m.eval()
        m.to(self.dtype)
        m_spyre = copy.deepcopy(m)
        m_spyre.to("spyre")

        x = torch.rand(B, T, config.hidden_size, dtype=self.dtype)
        x_spyre = x.to("spyre")

        position_ids = torch.arange(4096).unsqueeze(0)
        position_embeddings = llama_model.rotary_emb(x, position_ids)
        position_embeddings_spyre = (
            position_embeddings[0].to(self.dtype).to("spyre"),
            position_embeddings[1].to(self.dtype).to("spyre"),
        )

        with torch.no_grad():
            y = m(hidden_states=x, position_embeddings=position_embeddings)
            y_spyre = m_spyre(
                hidden_states=x_spyre,
                position_embeddings=position_embeddings_spyre,
            )

        torch.testing.assert_close(
            y_spyre[0].cpu(), y[0], rtol=self.rtol, atol=self.atol
        )


if __name__ == "__main__":
    run_tests()
