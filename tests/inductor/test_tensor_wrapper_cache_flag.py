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

"""TORCH_SPYRE_CACHE_TENSOR_WRAPPERS lets AOTAutogradCache keep graphs that call
torch-spyre's ``Tensor.to`` or ``torch.empty`` wrapper."""

import pickle
from unittest import mock

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.utils import fresh_cache

import torch_spyre  # noqa: F401  (installs the wrappers)
from torch_spyre._inductor import (
    _TENSOR_WRAPPER_NAMES,
    _allow_tensor_wrapper_caching,
)


def _wrappers():
    return (torch.Tensor.to, torch.empty)


def test_names_match_the_installed_wrappers():
    names = {f"{fn.__module__}.{fn.__name__}" for fn in _wrappers()}
    assert names == set(_TENSOR_WRAPPER_NAMES)


def test_wrappers_pickle_by_import_path():
    for fn in _wrappers():
        assert pickle.loads(pickle.dumps(fn)) is fn


def test_flag_off_leaves_the_wrappers_unmarked():
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_TENSOR_WRAPPERS": "0"}),
        inductor_config.patch(unsafe_marked_cacheable_functions={}),
    ):
        _allow_tensor_wrapper_caching()
        assert inductor_config.unsafe_marked_cacheable_functions == {}


def test_flag_on_marks_both_wrappers():
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_TENSOR_WRAPPERS": "1"}),
        inductor_config.patch(unsafe_marked_cacheable_functions={}),
    ):
        _allow_tensor_wrapper_caching()
        marked = inductor_config.unsafe_marked_cacheable_functions
        assert set(_TENSOR_WRAPPER_NAMES) <= set(marked)


def _cast(x):
    # Calling ``Tensor.to`` through the class, as model code sometimes does, puts the
    # wrapper itself into the graph as a call_function node.
    return torch.Tensor.to(x * 2, torch.float16)


def _aot_counts():
    c = counters["aot_autograd"]
    return c["autograd_cache_hit"], c["autograd_cache_bypass"]


def _compile_twice(flag):
    x = torch.randn(4, 8)
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_TENSOR_WRAPPERS": flag}),
        inductor_config.patch(unsafe_marked_cacheable_functions={}),
        fresh_cache(),
    ):
        _allow_tensor_wrapper_caching()
        counters.clear()
        for _ in range(2):
            torch._dynamo.reset()
            out = torch.compile(_cast, backend="inductor", fullgraph=True)(x)
        hits, bypasses = _aot_counts()
    torch.testing.assert_close(out, _cast(x))
    return hits, bypasses


def test_graph_with_to_is_cached_by_aot_autograd():
    """A graph that calls the ``.to`` wrapper hits AOTAutogradCache the second time."""
    assert _compile_twice("1") == (1, 0)


def test_flag_off_bypasses_graph_with_to():
    assert _compile_twice("0") == (0, 2)
