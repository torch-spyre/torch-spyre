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

"""TORCH_SPYRE_CACHE_SCAN lets graphs that use the ``scan`` op be cached."""

from unittest import mock

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._higher_order_ops.scan import scan, scan_op
from torch._inductor.utils import fresh_cache

from torch_spyre._inductor import _SCAN_HOP_NAME, _allow_scan_graph_caching


def test_scan_hop_name_matches_torch():
    assert _SCAN_HOP_NAME == f"{scan_op.__module__}.{scan_op.__name__}"


def test_flag_off_leaves_scan_uncacheable():
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_SCAN": "0"}),
        inductor_config.patch(unsafe_marked_cacheable_functions={}),
    ):
        _allow_scan_graph_caching()
        assert not scan_op.cacheable()


def test_flag_on_makes_scan_cacheable():
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_SCAN": "1"}),
        inductor_config.patch(unsafe_marked_cacheable_functions={}),
    ):
        _allow_scan_graph_caching()
        assert scan_op.cacheable()
        assert _SCAN_HOP_NAME in inductor_config.unsafe_marked_cacheable_functions


def _scan_sum(xs):
    def body(carry, x):
        new = carry + x
        return new, new.clone()

    return scan(body, torch.zeros_like(xs[0]), xs)


def _scan_max(xs):
    def body(carry, x):
        new = torch.maximum(carry, x)
        return new, new.clone()

    return scan(body, torch.zeros_like(xs[0]), xs)


def _compile_once(fn, xs):
    torch._dynamo.reset()
    return torch.compile(fn, backend="inductor", fullgraph=True)(xs)


def _fx_counts():
    c = counters["inductor"]
    return c["fxgraph_cache_hit"], c["fxgraph_cache_miss"], c["fxgraph_cache_bypass"]


def test_flag_on_caches_scan_graph_and_misses_on_new_body():
    """A cached scan graph is reused, and a new scan body is a new entry."""
    xs = torch.randn(4, 8)
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_SCAN": "1"}),
        inductor_config.patch(
            unsafe_marked_cacheable_functions={},
            fx_graph_cache=True,
            fx_graph_remote_cache=False,
        ),
        fresh_cache(),
    ):
        _allow_scan_graph_caching()
        counters.clear()

        _compile_once(_scan_sum, xs)
        assert _fx_counts() == (0, 1, 0)

        out = _compile_once(_scan_sum, xs)
        assert _fx_counts() == (1, 1, 0)
        torch.testing.assert_close(out, _scan_sum(xs))

        out = _compile_once(_scan_max, xs)
        assert _fx_counts() == (1, 2, 0)
        torch.testing.assert_close(out, _scan_max(xs))


def test_flag_off_bypasses_scan_graph():
    xs = torch.randn(4, 8)
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_SCAN": "0"}),
        inductor_config.patch(
            unsafe_marked_cacheable_functions={},
            fx_graph_cache=True,
            fx_graph_remote_cache=False,
        ),
        fresh_cache(),
    ):
        _allow_scan_graph_caching()
        counters.clear()
        _compile_once(_scan_sum, xs)
        _compile_once(_scan_sum, xs)
        hit, _, bypass = _fx_counts()
        assert hit == 0
        assert bypass >= 2


def _make_scaled_scan(scale):
    c = torch.full((8,), scale)

    def fn(xs):
        def body(carry, x):
            new = carry + x * c
            return new, new.clone()

        return scan(body, torch.zeros_like(xs[0]), xs)

    return fn


def test_flag_on_new_captured_constant_gives_right_result():
    """A scan body that differs only in a captured tensor is not served stale."""
    xs = torch.randn(4, 8)
    with (
        mock.patch.dict("os.environ", {"TORCH_SPYRE_CACHE_SCAN": "1"}),
        inductor_config.patch(
            unsafe_marked_cacheable_functions={},
            fx_graph_cache=True,
            fx_graph_remote_cache=False,
        ),
        fresh_cache(),
    ):
        _allow_scan_graph_caching()
        for scale in (2.0, 3.0):
            fn = _make_scaled_scan(scale)
            torch.testing.assert_close(_compile_once(fn, xs), fn(xs))
