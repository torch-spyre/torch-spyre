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

from typing import List, Tuple
import regex as re


def test_schema_argument_extraction():
    search_pattern = r"\((.*?)\)\s*->"

    test_cases: List[Tuple[str, str]] = [
        (
            "aten::_reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)",
            "Tensor(a) self, SymInt[] size, SymInt[] stride",
        ),
        (
            "aten::_reshape_copy(Tensor self, SymInt[] size) -> Tensor",
            "Tensor self, SymInt[] size",
        ),
        (
            "aten::_backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()",
            "Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False",
        ),
        (
            "aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
            "Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state",
        ),
        (
            "aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
            "Tensor self, *, Tensor(a!) out",
        ),
        ("aten::align_tensors(Tensor[] tensors) -> Tensor[]", "Tensor[] tensors"),
        (
            "aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)",
            "Tensor(a) self, SymInt[] size, *, bool implicit=False",
        ),
    ]
    for schema_string, expected in test_cases:
        extracted = re.search(search_pattern, schema_string).group(1)

        assert extracted == expected
