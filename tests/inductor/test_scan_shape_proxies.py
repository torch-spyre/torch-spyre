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

"""The scoped patch that lets a scan output shape hold an expression.

``resolve_shape_to_proxy`` is upstream's, and it looks its environment up only
for a bare symbol leaf, so an expression key such as ``FloorDiv(s, 64)`` is
never found and it raises ``KeyError(s)``. That is exactly the shape a symbolic
trip count produces. `_resolve_scan_shapes_by_expression` matches the whole
expression first and leaves every other path alone.

These tests drive the wrapper against a recording stub rather than against
upstream's real implementation, because the real one builds FX proxies and
needs a live tracer, so calling it standalone would fail for reasons that have
nothing to do with the claim. One test uses a real ``SymInt`` to pin the
accessor the wrapper reads, which is what would catch an upstream rename.

Two tests call upstream's real function, because they are about upstream
itself: its signature is still the one the wrapper is installed over, and it
still raises ``KeyError`` on an expression key, which it does before it builds
any proxy. When upstream fixes the gap the second one fails, and the patch can
be deleted.

The end-to-end claim, that a symbolic trip count now survives the scan
decomposition at all, is asserted on device in the trip-count recovery tests.
It cannot be asserted here: the patch is scoped to a Spyre compile and a CPU
compile never enters it.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import sympy
import torch._inductor.fx_passes.post_grad as post_grad
from torch.utils._sympy.functions import FloorDiv

from symbolic_shape_fixtures import TILE, fake_symbolic_rows  # noqa: E402
from torch_spyre._inductor.patches import (
    RESOLVE_SHAPE_ISSUE,
    RESOLVE_SHAPE_PARAMS,
    _resolve_scan_shapes_by_expression,
    logger as patches_logger,
)

ATTR = "resolve_shape_to_proxy"
SENTINEL = "the-proxy-for-the-whole-expression"


def _fake_size(expr):
    """A stand-in for a SymInt: the wrapper reads only ``.node.expr``."""
    return SimpleNamespace(node=SimpleNamespace(expr=expr))


def _recording_stub():
    """A stub standing in for upstream, plus the list of shapes it was given."""
    seen = []

    def resolve_shape_to_proxy(shape, bound_symbols):
        seen.append(list(shape))
        return [("fell-back", size) for size in shape]

    return resolve_shape_to_proxy, seen


class TestWholeExpressionIsMatched(unittest.TestCase):
    def test_an_expression_key_resolves_without_reaching_upstream(self):
        stub, seen = _recording_stub()
        # torch's FloorDiv, not sympy's floor: this is the type the trip count
        # actually arrives as before anything round-trips it.
        expr = FloorDiv(sympy.Symbol("s97", integer=True, positive=True), TILE)

        with mock.patch.object(post_grad, ATTR, stub):
            with _resolve_scan_shapes_by_expression():
                resolved = getattr(post_grad, ATTR)(
                    [_fake_size(expr)], {expr: SENTINEL}
                )

        self.assertEqual(resolved, [SENTINEL])
        self.assertEqual(seen, [], "upstream was called for a key we already had")

    def test_anything_else_still_goes_to_upstream_one_element_at_a_time(self):
        stub, seen = _recording_stub()
        s = sympy.Symbol("s97", integer=True, positive=True)
        unknown = _fake_size(s * 3)
        plain = 64  # a concrete size has no .node at all

        with mock.patch.object(post_grad, ATTR, stub):
            with _resolve_scan_shapes_by_expression():
                getattr(post_grad, ATTR)([unknown, plain], {s // TILE: SENTINEL})

        # One call per element, each a single-element list, so each keeps
        # upstream's own type checking and error message for that element.
        self.assertEqual(seen, [[unknown], [plain]])

    def test_a_mixed_shape_resolves_each_element_on_its_own_path(self):
        stub, seen = _recording_stub()
        s = sympy.Symbol("s97", integer=True, positive=True)
        known = _fake_size(s // TILE)
        unknown = _fake_size(s * 3)

        with mock.patch.object(post_grad, ATTR, stub):
            with _resolve_scan_shapes_by_expression():
                resolved = getattr(post_grad, ATTR)(
                    [known, 7, unknown], {s // TILE: SENTINEL}
                )

        self.assertEqual(resolved[0], SENTINEL)
        self.assertEqual(resolved[1], ("fell-back", 7))
        self.assertEqual(resolved[2], ("fell-back", unknown))
        self.assertEqual(seen, [[7], [unknown]])


class TestTheAccessorIsRight(unittest.TestCase):
    """A real SymInt, because that is what would catch an upstream rename.

    Every other test here uses a stand-in with a hand-built ``.node.expr``, so
    none of them would notice if torch stopped exposing the expression there.
    """

    def test_a_real_symint_exposes_its_expression_where_we_read_it(self):
        shape_env, mode, x = fake_symbolic_rows()
        with mode:
            tiles = x.shape[0] // TILE

        expr = getattr(getattr(tiles, "node", None), "expr", None)
        self.assertIsNotNone(
            expr,
            "a SymInt no longer exposes .node.expr, so the wrapper reads the "
            "wrong attribute and silently falls back for every element",
        )

        stub, seen = _recording_stub()
        with mock.patch.object(post_grad, ATTR, stub):
            with _resolve_scan_shapes_by_expression():
                resolved = getattr(post_grad, ATTR)([tiles], {expr: SENTINEL})

        self.assertEqual(resolved, [SENTINEL])
        self.assertEqual(seen, [])


class TestThePatchIsScoped(unittest.TestCase):
    def test_the_original_is_restored_on_exit(self):
        stub, _ = _recording_stub()

        with mock.patch.object(post_grad, ATTR, stub):
            with _resolve_scan_shapes_by_expression():
                self.assertIsNot(getattr(post_grad, ATTR), stub)
            self.assertIs(getattr(post_grad, ATTR), stub)

    def test_the_original_is_restored_even_if_the_body_raises(self):
        stub, _ = _recording_stub()

        with mock.patch.object(post_grad, ATTR, stub):
            with self.assertRaises(RuntimeError):
                with _resolve_scan_shapes_by_expression():
                    raise RuntimeError("boom")
            self.assertIs(getattr(post_grad, ATTR), stub)

    def test_an_absent_upstream_function_is_a_loud_no_op(self):
        """If upstream renames or fixes it, say so rather than failing."""
        saved = getattr(post_grad, ATTR, None)
        self.assertIsNotNone(
            saved,
            f"post_grad.{ATTR} is gone. Either upstream fixed the expression-key "
            f"gap, in which case this patch can be deleted, or it moved and the "
            f"patch is now silently doing nothing. Find out which",
        )
        try:
            delattr(post_grad, ATTR)
            with self.assertLogs(patches_logger.name, level="WARNING") as captured:
                with _resolve_scan_shapes_by_expression():
                    self.assertFalse(hasattr(post_grad, ATTR))
        finally:
            setattr(post_grad, ATTR, saved)

        # Assert on the record, not its wording: one warning that carries the
        # issue link as an argument.
        (record,) = captured.records
        self.assertEqual(record.levelname, "WARNING")
        self.assertIn(RESOLVE_SHAPE_ISSUE, record.args)


class TestTheSignatureGuard(unittest.TestCase):
    def test_a_changed_signature_is_left_alone_and_warned_about(self):
        def resolve_shape_to_proxy(shape, bound_symbols, extra=None):
            raise AssertionError("the guard should not wrap or call this")

        with mock.patch.object(post_grad, ATTR, resolve_shape_to_proxy):
            with self.assertLogs(patches_logger.name, level="WARNING") as captured:
                with _resolve_scan_shapes_by_expression():
                    self.assertIs(getattr(post_grad, ATTR), resolve_shape_to_proxy)
            self.assertIs(getattr(post_grad, ATTR), resolve_shape_to_proxy)

        (record,) = captured.records
        self.assertEqual(record.levelname, "WARNING")
        self.assertIn(("shape", "bound_symbols", "extra"), record.args)
        self.assertIn(RESOLVE_SHAPE_ISSUE, record.args)

    def test_the_expected_signature_is_wrapped_without_a_warning(self):
        stub, _ = _recording_stub()

        with mock.patch.object(post_grad, ATTR, stub):
            with self.assertNoLogs(patches_logger.name, level="WARNING"):
                with _resolve_scan_shapes_by_expression():
                    self.assertIsNot(getattr(post_grad, ATTR), stub)


class TestTheUpstreamGapIsStillThere(unittest.TestCase):
    """Upstream's real function, so a PyTorch upgrade cannot pass this blind."""

    def test_upstream_still_has_the_signature_the_wrapper_expects(self):
        self.assertEqual(
            tuple(inspect.signature(getattr(post_grad, ATTR)).parameters),
            RESOLVE_SHAPE_PARAMS,
            f"post_grad.{ATTR} changed its signature, so the patch is no longer "
            f"installed. Check whether {RESOLVE_SHAPE_ISSUE} is fixed, then "
            f"update or delete _resolve_scan_shapes_by_expression",
        )

    def test_upstream_still_misses_an_expression_key(self):
        shape_env, mode, x = fake_symbolic_rows()
        with mode:
            tiles = x.shape[0] // TILE
        expr = tiles.node.expr

        try:
            getattr(post_grad, ATTR)([tiles], {expr: SENTINEL})
        except KeyError:
            return
        self.fail(
            f"upstream {ATTR} now resolves an expression key, so "
            f"{RESOLVE_SHAPE_ISSUE} looks fixed. Delete "
            f"_resolve_scan_shapes_by_expression and this test"
        )


if __name__ == "__main__":
    unittest.main()
