# Copyright 2026 The Torch-Spyre Authors.
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

"""Helpers in ``scratchpad/utils.py`` that decide LX residency.

``_would_produce_lx_back_gap`` asks whether a device dimension is *fully
covered* by the iteration symbols that walk it: a backGap fires when
``device_size[d]`` exceeds the extent the coordinate actually reaches, and the
backend supports that for HBM but not for LX.

The covered extent is a property of the whole coordinate expression, not of any
one symbol in it. These tests pin that, because the two obvious cheaper
approximations are both wrong and were both live:

* Substituting *one* symbol's iteration range (what this did before) breaks as
  soon as a coordinate folds two symbols into one axis, in both directions --
  see ``test_multi_symbol_*`` for a spurious gap and ``test_stickified_*`` for a
  missed one. It was also read out of the unordered ``free_symbols`` set, so
  which symbol you got moved with ``PYTHONHASHSEED``; that made LX plans, and
  hence core-division plans, differ between processes on identical input.
* Substituting *every* symbol's maximum is right only for a monotone
  coordinate. ``test_mod_coordinate_uses_a_real_bound`` is the counter-example.
"""

import unittest
from types import SimpleNamespace
from unittest import TestCase, mock

import pytest
import sympy
import torch
import torch._dynamo
from torch.testing import FileCheck
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ExternKernel, FallbackKernel

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.utils import (
    _would_produce_lx_back_gap,
    get_ncores_for_buffers,
)
from utils_inductor import compare_with_cpu, cached_randn

_COORDS = "torch_spyre._inductor.scratchpad.utils.device_coordinates"

_BUF = "buf0"

# Iteration symbols, named as the pre-scheduler names them.
d0, d1, d2, d3 = sympy.symbols("d0 d1 d2 d3", integer=True, nonnegative=True)

# The trailing device coordinate is the within-stick lane, which the check skips
# (a stick is atomic, so it cannot carry a gap). Every fixture here has exactly
# one non-stick device dim, so ``device_size[0]`` is the dim under test.
_STICK_COORD = sympy.Mod(d0, 64)
_STICK_SIZE = 64


def _graph(extent, ranges):
    """A one-op graph whose only read of ``_BUF`` carries ``ranges``.

    ``extent`` is ``device_size[0]``, the non-stick device dim under test.
    ``ranges`` maps each iteration symbol to its size, which is what
    ``MemoryDep`` exposes as ``dep.ranges``.
    """
    symbols = tuple(ranges)
    dep = MemoryDep(
        _BUF,
        # The index is unused here: the coordinate under test is supplied by the
        # patched ``device_coordinates``. Kept plausible rather than empty.
        sum(symbols, sympy.Integer(0)),
        symbols,
        tuple(ranges[sym] for sym in symbols),
    )
    op = SimpleNamespace(
        get_read_writes=lambda: SimpleNamespace(reads={dep}, writes=set())
    )
    buf = SimpleNamespace(
        layout=SimpleNamespace(
            device_layout=SimpleNamespace(device_size=[extent, _STICK_SIZE])
        )
    )
    return SimpleNamespace(operations=[op], get_buffer=lambda _name: buf)


def _back_gap(coord, extent, ranges):
    graph = _graph(extent, ranges)
    with mock.patch(_COORDS, return_value=[coord, _STICK_COORD]):
        return _would_produce_lx_back_gap(graph, _BUF, [0])


class BackGapTest(TestCase):
    def test_multi_symbol_coordinate_that_covers_its_dim_has_no_gap(self):
        """``2*d1 + d3`` over ``d1 in [0,40)``, ``d3 in [0,2)`` reaches 0..79.

        The dim is 80 wide, so it is exactly covered and there is no gap. Note
        that *neither* single symbol's range answers this: 80 > 40 and 80 > 2 are
        both true, so picking either one reports a gap that is not there. This
        case fails on any hash seed, which is what makes it a regression test.
        """
        self.assertFalse(_back_gap(2 * d1 + d3, 80, {d1: 40, d3: 2}))

    def test_multi_symbol_stickified_coordinate_has_no_gap(self):
        """``2*d0 + floor(d2/64)`` over ``d0 in [0,8)``, ``d2 in [0,128)``.

        The shape observed on a granite-4.0-micro decoder block, where this
        decided one buffer's LX residency and, through it, the whole
        core-division plan. Reaches 0..15 on a dim 16 wide: no gap. Here the two
        symbols disagree (``d0`` says gap, ``d2`` says none), which is how the
        verdict came to depend on the hash seed.
        """
        self.assertFalse(_back_gap(2 * d0 + sympy.floor(d2 / 64), 16, {d0: 8, d2: 128}))

    def test_uncovered_dim_has_a_gap(self):
        """``d0`` over ``[0,8)`` reaches 0..7, which leaves a dim 16 wide short.

        The positive case: without it the tests above would pass against a
        function that always returned False.
        """
        self.assertTrue(_back_gap(d0, 16, {d0: 8}))

    def test_stickified_coordinate_gap_is_not_hidden_by_the_element_range(self):
        """``floor(d1/64)`` over ``d1 in [0,1024)`` reaches 0..15, not 0..1023.

        A stick-count coordinate covers ``range/64``, so a dim 32 wide really
        does have a gap. Reading ``d1``'s raw *element* range instead (1024)
        overstates coverage by 64x and hides it -- the dangerous direction, since
        LX has no backGap support, so a missed gap is a codegen hazard rather
        than a lost optimization.
        """
        self.assertTrue(_back_gap(sympy.floor(d1 / 64), 32, {d1: 1024}))

    def test_mod_coordinate_uses_a_real_bound(self):
        """``Mod(d0, 5)`` over ``d0 in [0,8)`` reaches 0..4, so a dim 4 wide is
        covered.

        Non-monotone, so substituting the symbol's maximum is not its bound:
        ``Mod(7, 5)`` is 2, which would understate the extent as 3 and report a
        gap. This is why the check needs real range analysis and not ``subs``.
        """
        self.assertFalse(_back_gap(sympy.Mod(d0, 5), 4, {d0: 8}))


class ExternalOperandTest(TestCase):
    def test_external_operands_never_publish_physical_ownership(self):
        graph = SimpleNamespace(try_get_buffer=lambda name: None)
        for op_type in (ExternKernel, FallbackKernel):
            for cores in (1, 32):
                with self.subTest(op=op_type.__name__, cores=cores):
                    op = mock.Mock(spec=op_type)
                    op.get_name.return_value = "opaque"
                    with (
                        config.patch({"sencores": cores}),
                        mock.patch(
                            "torch_spyre._inductor.scratchpad.utils._get_buffer_user_deps",
                            return_value={_BUF: [(op, None)]},
                        ),
                        mock.patch(
                            "torch_spyre._inductor.scratchpad.utils._per_core_view_on_buf"
                        ) as project,
                    ):
                        counts, reasons, views = get_ncores_for_buffers(graph)
                    self.assertEqual(counts, {_BUF: -1})
                    self.assertIn("FallbackKernel/ExternKernel", reasons[_BUF])
                    self.assertEqual(views, {})
                    project.assert_not_called()


def _lx_residency_check(source: str) -> None:
    FileCheck().check("{lx:").run(source)


class TestBackGapProducingBufferExclusion(TestCase):
    """A non-stick-aligned cat buffer must be excluded from LX for some
    real residency reason, not silently placed."""

    def test_gap_producing_buffer_is_excluded_from_lx(self):
        from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator

        rows, c1, c2 = 32, 8, 120

        def cat_then_use_twice(a, b):
            cat = torch.cat([a, b], dim=-1)
            return cat * 2 + cat

        a = cached_randn((rows, c1))
        b = cached_randn((rows, c2), differentiation=1)

        captured: dict[str, "str | None"] = {}
        orig = ScratchpadAllocator._buffer_residency_reason

        def wrapper(self, graph, name, uses, op, **kwargs):
            reason = orig(self, graph, name, uses, op, **kwargs)
            captured[name] = reason
            return reason

        with (
            config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}),
            mock.patch.object(ScratchpadAllocator, "_buffer_residency_reason", wrapper),
        ):
            torch._dynamo.reset()
            compare_with_cpu(cat_then_use_twice, a, b)

        self.assertNotIn(
            None,
            captured.values(),
            "Expected every buffer in this non-stick-aligned cat scenario "
            "to be excluded from LX for some real reason (none cleanly "
            f"placed); captured reasons: {captured}",
        )


def _layernorm_chain(x, weight, bias, normalized_shape):
    # A second op after layer_norm so the normalized output is a real LX
    # candidate, not just an unread graph output.
    y = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps=1e-5)
    return y * 2 + y


class TestLayerNormAddressing:
    """LayerNorm correctness and LX residency under default lx_planning,
    including across core counts (#2533)."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.mark.parametrize(
        "rows,hidden,dtype",
        [
            (2048, 4096, torch.float16),  # Granite/Llama prefill hidden
            (1, 4096, torch.float16),  # decode (M=1) -- boundary shape
            (49152, 768, torch.bfloat16),  # BERT-style hidden, bf16
            (2048, 12800, torch.float16),  # Granite intermediate-sized hidden
            (2049, 4096, torch.float16),  # non-power-of-2 row count
            (32, 256, torch.bfloat16),  # small shape, bf16
        ],
    )
    def test_layernorm_matches_cpu_under_default_lx_planning(self, rows, hidden, dtype):
        x = cached_randn((rows, hidden), dtype=dtype)
        weight = cached_randn((hidden,), differentiation=1, dtype=dtype)
        bias = cached_randn((hidden,), differentiation=2, dtype=dtype)

        with config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}):
            compare_with_cpu(
                lambda x, w, b: _layernorm_chain(x, w, b, (hidden,)),
                x,
                weight,
                bias,
                atol=0.15,
                rtol=0.15,
                source_check=_lx_residency_check,
            )

    @pytest.mark.parametrize("sencores", [1, 4, 8, 32])
    def test_layernorm_matches_cpu_across_sencores(self, sencores):
        """LayerNorm's split dim is the outer (row) dim; sweep core counts
        to exercise the per-core LX addressing path."""
        rows, hidden = 2048, 4096
        x = cached_randn((rows, hidden))
        weight = cached_randn((hidden,), differentiation=1)
        bias = cached_randn((hidden,), differentiation=2)

        with config.patch(
            {
                "lx_planning": True,
                "allow_all_ops_in_lx_planning": True,
                "sencores": sencores,
            }
        ):
            compare_with_cpu(
                lambda x, w, b: _layernorm_chain(x, w, b, (hidden,)),
                x,
                weight,
                bias,
                atol=0.15,
                rtol=0.15,
                source_check=_lx_residency_check,
            )


if __name__ == "__main__":
    unittest.main()
