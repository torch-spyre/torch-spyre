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

"""Indirect access (gather) with a symbolic index count and a symbolic source.

The baseline is a paged gather: a block table (the INDEX tensor) is tiled one
row per trip by ``for_each_tile``, each row names a page, and the page is
gathered out of a page pool (the SOURCE tensor) inside the loop body. Both
batch dimensions are symbolic:

* the block count is the loop's trip count -- the combined branch's symbolic
  loop route, a ``loop_dimension`` bundle parameter bound at launch;
* the pool length is a TABLE LENGTH -- it only bounds the index values, so it
  becomes no loop and no bundle parameter. It is resolved to its declared
  maximum, which is addressable because ``.to("spyre", dynamic=...)`` reserved
  the pool at that maximum.

Every on-device scenario runs at least three sizes with at least one ABOVE the
first call's, counts compiles of the user function only, and checks numbers
against a CPU reference. A single size, or only sizes at or below the warm-up,
would pass even if the geometry were sized from the warm-up call.
"""

import os
import tempfile
import unittest
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace
from unittest.mock import patch

import regex as re
import sympy
import torch
from torch._inductor import ir
from torch._inductor.compile_fx import compile_fx
from torch._inductor.utils import run_and_get_code

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._C import SymbolicArgKind, _resolve_symbolic_args
from torch_spyre._inductor import insert_restickify, pass_utils
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.wsr import coarse_tile
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.execution import kernel_runner

PAGE_SIZE, PAGE_HS, Q_ROWS = 32, 64, 32

POOL_MIN, POOL_MAX, POOL_G = 8, 16, 8  # source tensor, dim 0
BLOCKS_MIN, BLOCKS_MAX, BLOCKS_G = 8, 16, 1  # index tensor, dim 0 (G == tile_size)
POOL_SPEC = {0: {"min": POOL_MIN, "max": POOL_MAX, "granularity": POOL_G}}
BLOCKS_SPEC = {0: {"min": BLOCKS_MIN, "max": BLOCKS_MAX, "granularity": BLOCKS_G}}

# Max-normalised relative error (rel_err below), set at about 2x the worst
# observed on device over BOTH_SIZES. Reduction mode chains two fp16 matmuls and
# accumulates over up to 16 blocks: observed 3.4e-3 .. 5.1e-3. Map mode is a
# gathered copy times 2.0, so only the fp16 <-> device-format rounding remains:
# observed 4.5e-4 .. 9.3e-4.
REL_TOL = 1e-2
MAP_REL_TOL = 2e-3

# Shared by both refusals (pass_utils.undeclared_maximum_message): names the
# missing maximum and the supported way to declare it.
UNDECLARED_MAX_REFUSAL = r'no declared maximum.*\.to\("spyre", dynamic='


def _check_blocks(block_table):
    n = block_table.shape[0]
    torch._check(n >= BLOCKS_MIN)
    torch._check(n <= BLOCKS_MAX)
    torch._check(n % BLOCKS_G == 0)


def _check_pool(pages_pool):
    p = pages_pool.shape[0]
    torch._check(p >= POOL_MIN)
    torch._check(p <= POOL_MAX)
    torch._check(p % POOL_G == 0)


def paged_gather(pages_pool, block_table, q_in):
    """The baseline: reduction-mode paged gather, both batch dims symbolic."""
    _check_blocks(block_table)
    _check_pool(pages_pool)

    def body(acc, tiles):
        table_row, pool, q_whole = tiles
        page_idx = table_row[0, 0:1]
        page = pool.index_select(0, page_idx).squeeze(0)
        scores = q_whole @ page.transpose(0, 1)
        return acc + scores @ page, None

    acc0 = torch.zeros(Q_ROWS, PAGE_HS, device=q_in.device, dtype=q_in.dtype)
    out, _ = for_each_tile(
        body,
        (block_table, pages_pool, q_in),
        dims=(0, None, None),
        tile_size=BLOCKS_G,
        init=acc0,
    )
    return out


def paged_gather_index_only(pages_pool, block_table, q_in):
    """Same body; only the index tensor is declared symbolic."""
    _check_blocks(block_table)

    def body(acc, tiles):
        table_row, pool, q_whole = tiles
        page = pool.index_select(0, table_row[0, 0:1]).squeeze(0)
        return acc + (q_whole @ page.transpose(0, 1)) @ page, None

    acc0 = torch.zeros(Q_ROWS, PAGE_HS, device=q_in.device, dtype=q_in.dtype)
    out, _ = for_each_tile(
        body,
        (block_table, pages_pool, q_in),
        dims=(0, None, None),
        tile_size=BLOCKS_G,
        init=acc0,
    )
    return out


def paged_gather_pool_only(pages_pool, block_table, q_in):
    """Same body; only the source tensor is declared symbolic."""
    _check_pool(pages_pool)

    def body(acc, tiles):
        table_row, pool, q_whole = tiles
        page = pool.index_select(0, table_row[0, 0:1]).squeeze(0)
        return acc + (q_whole @ page.transpose(0, 1)) @ page, None

    acc0 = torch.zeros(Q_ROWS, PAGE_HS, device=q_in.device, dtype=q_in.dtype)
    out, _ = for_each_tile(
        body,
        (block_table, pages_pool, q_in),
        dims=(0, None, None),
        tile_size=BLOCKS_G,
        init=acc0,
    )
    return out


def gather_pages_map(pages_pool, block_table):
    """Map mode: each trip gathers one page and writes it at its block's slot."""
    _check_blocks(block_table)
    _check_pool(pages_pool)

    def body(_carry, tiles):
        table_row, pool = tiles
        page = pool.index_select(0, table_row[0, 0:1])  # [1, PAGE_SIZE, PAGE_HS]
        return None, page * 2.0

    _, out = for_each_tile(
        body,
        (block_table, pages_pool),
        dims=(0, None),
        tile_size=BLOCKS_G,
        out_dim=0,
    )
    return out


def paged_gather_no_pool_range(pages_pool, block_table, q_in):
    """Pool marked dynamic, but its range is never declared in the trace."""
    _check_blocks(block_table)

    def body(acc, tiles):
        table_row, pool, q_whole = tiles
        page = pool.index_select(0, table_row[0, 0:1]).squeeze(0)
        return acc + (q_whole @ page.transpose(0, 1)) @ page, None

    acc0 = torch.zeros(Q_ROWS, PAGE_HS, device=q_in.device, dtype=q_in.dtype)
    out, _ = for_each_tile(
        body,
        (block_table, pages_pool, q_in),
        dims=(0, None, None),
        tile_size=BLOCKS_G,
        init=acc0,
    )
    return out


# ------------------------------------------------------------------ references


def ref_paged_gather(pages, table, q):
    acc = torch.zeros(Q_ROWS, PAGE_HS)
    for b in range(table.shape[0]):
        page = pages[int(table[b, 0])].float()
        acc += (q.float() @ page.T) @ page
    return acc


def ref_gather_pages_map(pages, table):
    return torch.stack(
        [pages[int(table[b, 0])].float() * 2.0 for b in range(table.shape[0])]
    )


def rel_err(out, ref):
    return (
        (out.cpu().float() - ref).abs().max() / ref.abs().max().clamp(min=1e-6)
    ).item()


# ------------------------------------------------------------------ harness


class _Counted:
    """torch.compile(fn) that counts only fn's own compiles."""

    def __init__(self, fn):
        self.compiles = 0

        def backend(gm, example_inputs):
            self.compiles += 1
            return compile_fx(gm, example_inputs)

        self.fn = torch.compile(fn, backend=backend, fullgraph=True)


def _pages(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, PAGE_SIZE, PAGE_HS, dtype=torch.float16, generator=g) * 0.25


def _table(n_blocks, n_pages, seed, dtype=torch.int32):
    g = torch.Generator().manual_seed(seed + 1000)
    return torch.randint(0, n_pages, (n_blocks, 32), dtype=dtype, generator=g)


def _q():
    g = torch.Generator().manual_seed(7)
    return torch.randn(Q_ROWS, PAGE_HS, dtype=torch.float16, generator=g) * 0.25


@contextmanager
def _fresh_inductor_cache():
    with tempfile.TemporaryDirectory(prefix="symbolic_gather_") as d:
        with patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": d}):
            yield d


def _bundles_under(root):
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "bundle.mlir" in files:
            with open(os.path.join(dirpath, "bundle.mlir")) as f:
                found.append(f.read())
    return found


def _with_dynamo_reset(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        torch._dynamo.reset()
        return fn(*args, **kwargs)

    return wrapper


class _ResetEachTest(unittest.TestCase):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if name.startswith("test") and callable(value):
                setattr(cls, name, _with_dynamo_reset(value))


# ------------------------------------------------------------------ device tests

# (blocks, pages): first call 12 blocks / 8 pages, then sizes below and ABOVE it.
BOTH_SIZES = [(12, 8), (8, 16), (16, 8), (10, 16)]


class TestPagedGatherBothSymbolic(_ResetEachTest):
    """The baseline: index tensor AND source tensor batch dims symbolic."""

    def test_one_binary_correct_at_every_size(self):
        c = _Counted(paged_gather)
        q = _q()
        for i, (nb, np_) in enumerate(BOTH_SIZES):
            pages, table = _pages(np_, i), _table(nb, np_, i)
            out = c.fn(
                pages.to(DEVICE_NAME, dynamic=POOL_SPEC),
                table.to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
                q.to(DEVICE_NAME),
            )
            self.assertEqual(tuple(out.shape), (Q_ROWS, PAGE_HS))
            err = rel_err(out, ref_paged_gather(pages, table, q))
            self.assertLess(err, REL_TOL, f"blocks={nb} pages={np_}")
        self.assertEqual(c.compiles, 1, "recompiled within the declared ranges")

    def test_pages_beyond_the_first_pool_are_reachable(self):
        """Every page id lands in the UPPER half of a 16-page pool, after a
        first call with an 8-page pool: a gather sized from the warm-up pool
        could not address them."""
        c = _Counted(paged_gather)
        q = _q()
        pages8, table8 = _pages(8, 1), _table(8, 8, 1)
        c.fn(
            pages8.to(DEVICE_NAME, dynamic=POOL_SPEC),
            table8.to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
            q.to(DEVICE_NAME),
        )
        pages16 = _pages(16, 2)
        table_hi = torch.randint(8, 16, (16, 32), dtype=torch.int32)
        out = c.fn(
            pages16.to(DEVICE_NAME, dynamic=POOL_SPEC),
            table_hi.to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
            q.to(DEVICE_NAME),
        )
        self.assertLess(rel_err(out, ref_paged_gather(pages16, table_hi, q)), REL_TOL)
        self.assertEqual(c.compiles, 1)

    def test_bundle_and_kernel_structure(self):
        """One dimension parameter (the block count), none for the pool, and
        every pool geometry sized at the pool's declared maximum."""
        with _fresh_inductor_cache() as cache:
            c = _Counted(paged_gather)
            q = _q()
            pages, table = _pages(8, 0), _table(12, 8, 0)
            _, code = run_and_get_code(
                c.fn,
                pages.to(DEVICE_NAME, dynamic=POOL_SPEC),
                table.to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
                q.to(DEVICE_NAME),
            )
            bundles = _bundles_under(cache)
        src = "\n".join(code)

        bounds = re.findall(r"count_symbol_bounds=\{([^}]*)\}", src)
        self.assertEqual(len(bounds), 1, f"expected one symbolic loop, got {bounds}")
        self.assertRegex(bounds[0], r"^'s\d+': \(16, 1\)$")
        self.assertRegex(src, r"count_symbol_sources=\{'s\d+': \(1, 0\)\}")
        # The pool copy inside the body is sized at the declared max (16 pages).
        self.assertRegex(src, r"device_size=\[1, (16, 32|32, 16), 64\]")

        loop_bundle = next(b for b in bundles if "scf.for" in b)
        dim_params = re.findall(
            r"%dim_\w+_base: !sdscbundle\.input_arg<[^>]*>", loop_bundle
        )
        self.assertEqual(
            len(dim_params), 1, f"pool must not add a parameter: {dim_params}"
        )
        self.assertIn("granularity=1, max_value=16", dim_params[0])
        self.assertRegex(loop_bundle, r"scf\.for %\w+ = %c0 to %dim_\w+ step %c1")
        self.assertNotRegex(loop_bundle, r"divsi|divui|ceildiv|floordiv")

    def test_launch_binds_the_logical_block_count(self):
        c = _Counted(paged_gather)
        q = _q()
        launches = []
        original = kernel_runner.launch_jobplan

        def record(job_plan, args, symbolic_args=()):
            launches.append((list(args), list(symbolic_args)))
            if symbolic_args:
                return original(job_plan, args, symbolic_args)
            return original(job_plan, args)

        with patch.object(kernel_runner, "launch_jobplan", side_effect=record):
            for i, (nb, np_) in enumerate(BOTH_SIZES):
                launches.clear()
                c.fn(
                    _pages(np_, i).to(DEVICE_NAME, dynamic=POOL_SPEC),
                    _table(nb, np_, i).to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
                    q.to(DEVICE_NAME),
                )
                bound = [
                    _resolve_symbolic_args(args, [sa])[0]
                    for args, payload in launches
                    for sa in payload
                    if sa.kind == SymbolicArgKind.kDimension
                ]
                self.assertEqual(
                    bound, [nb], "one dimension slot, the real block count"
                )
        self.assertEqual(c.compiles, 1)

    def test_reserved_buffers_resized_in_place(self):
        """Serving pattern: both tensors reserved once at max, resized per call."""
        c = _Counted(paged_gather)
        q = _q()
        pool_buf = torch.zeros(POOL_MAX, PAGE_SIZE, PAGE_HS, dtype=torch.float16).to(
            DEVICE_NAME, dynamic=POOL_SPEC
        )
        table_buf = torch.zeros(BLOCKS_MAX, 32, dtype=torch.int32).to(
            DEVICE_NAME, dynamic=BLOCKS_SPEC
        )
        ptrs = (
            pool_buf.untyped_storage().data_ptr(),
            table_buf.untyped_storage().data_ptr(),
        )
        for i, (nb, np_) in enumerate(BOTH_SIZES):
            pages, table = _pages(np_, i), _table(nb, np_, i)
            pool_buf.resize_(np_, PAGE_SIZE, PAGE_HS)
            table_buf.resize_(nb, 32)
            pool_buf.copy_(pages)
            table_buf.copy_(table)
            out = c.fn(pool_buf, table_buf, q.to(DEVICE_NAME))
            self.assertLess(rel_err(out, ref_paged_gather(pages, table, q)), REL_TOL)
            self.assertEqual(
                (
                    pool_buf.untyped_storage().data_ptr(),
                    table_buf.untyped_storage().data_ptr(),
                ),
                ptrs,
            )
        self.assertEqual(c.compiles, 1)


class TestPagedGatherOneSideSymbolic(_ResetEachTest):
    def test_index_only(self):
        c = _Counted(paged_gather_index_only)
        q = _q()
        pages = _pages(8, 0)
        for i, nb in enumerate([12, 8, 16]):
            table = _table(nb, 8, i)
            out = c.fn(
                pages.to(DEVICE_NAME),
                table.to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
                q.to(DEVICE_NAME),
            )
            self.assertLess(rel_err(out, ref_paged_gather(pages, table, q)), REL_TOL)
        self.assertEqual(c.compiles, 1)

    def test_pool_only(self):
        """The index count is static, so there is no symbolic loop at all: the
        pool's symbol must still never reach the body or a bundle parameter."""
        c = _Counted(paged_gather_pool_only)
        q = _q()
        for i, np_ in enumerate([8, 16, 8]):
            pages, table = _pages(np_, i), _table(8, np_, i)
            out = c.fn(
                pages.to(DEVICE_NAME, dynamic=POOL_SPEC),
                table.to(DEVICE_NAME),
                q.to(DEVICE_NAME),
            )
            self.assertLess(rel_err(out, ref_paged_gather(pages, table, q)), REL_TOL)
        self.assertEqual(c.compiles, 1)


class TestGatherMapMode(_ResetEachTest):
    def test_gather_pages_into_blocks(self):
        """Map mode: the gathered page is the output tile, one per block."""
        c = _Counted(gather_pages_map)
        for i, (nb, np_) in enumerate(BOTH_SIZES):
            pages, table = _pages(np_, i), _table(nb, np_, i)
            out = c.fn(
                pages.to(DEVICE_NAME, dynamic=POOL_SPEC),
                table.to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
            )
            self.assertEqual(tuple(out.shape), (nb, PAGE_SIZE, PAGE_HS))
            self.assertLess(
                rel_err(out, ref_gather_pages_map(pages, table)), MAP_REL_TOL
            )
        self.assertEqual(c.compiles, 1)


class TestGatherIndexDtype(_ResetEachTest):
    def test_int64_block_table(self):
        c = _Counted(paged_gather)
        q = _q()
        for i, (nb, np_) in enumerate(BOTH_SIZES[:3]):
            pages, table = _pages(np_, i), _table(nb, np_, i, dtype=torch.int64)
            out = c.fn(
                pages.to(DEVICE_NAME, dynamic=POOL_SPEC),
                table.to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
                q.to(DEVICE_NAME),
            )
            self.assertLess(rel_err(out, ref_paged_gather(pages, table, q)), REL_TOL)
        self.assertEqual(c.compiles, 1)


class TestGatherKnownLimits(_ResetEachTest):
    def test_index_past_current_pool_reads_stale_rows(self):
        """KNOWN LIMITATION, pinned so a change to it is noticed (indirect access
        track, https://github.com/torch-spyre/torch-spyre/issues/4382).

        The index range is the pool's declared maximum, so a page id at or
        above the CURRENT pool size but below the maximum is not refused: the
        gather reads whatever the reservation still holds there. Here the pool
        is filled at 16 pages and shrunk to 8 without a refill, so ids 8..15
        return the old pages. If a bounds check is added this test must fail
        and be replaced by one that expects the refusal.
        """
        c = _Counted(paged_gather)
        q = _q()
        pool_buf = torch.zeros(POOL_MAX, PAGE_SIZE, PAGE_HS, dtype=torch.float16).to(
            DEVICE_NAME, dynamic=POOL_SPEC
        )
        old_pages = _pages(POOL_MAX, 3)
        pool_buf.copy_(old_pages)
        pool_buf.resize_(POOL_MIN, PAGE_SIZE, PAGE_HS)
        table = torch.randint(POOL_MIN, POOL_MAX, (BLOCKS_MIN, 32), dtype=torch.int32)
        out = c.fn(
            pool_buf, table.to(DEVICE_NAME, dynamic=BLOCKS_SPEC), q.to(DEVICE_NAME)
        )
        self.assertLess(rel_err(out, ref_paged_gather(old_pages, table, q)), REL_TOL)
        self.assertEqual(c.compiles, 1)


class TestGatherRefusals(_ResetEachTest):
    def test_undeclared_pool_range_is_refused_not_guessed(self):
        """A symbolic pool with no declared maximum has no static range, so the
        compile refuses it by name instead of sizing the gather from the
        warm-up pool."""
        c = _Counted(paged_gather_no_pool_range)
        q = _q()
        pages, table = _pages(8, 0), _table(8, 8, 0)
        with self.assertRaisesRegex(Exception, UNDECLARED_MAX_REFUSAL):
            c.fn(
                pages.to(DEVICE_NAME, dynamic=POOL_SPEC),
                table.to(DEVICE_NAME, dynamic=BLOCKS_SPEC),
                q.to(DEVICE_NAME),
            )

    def test_pool_size_off_granularity_rejected_at_transfer(self):
        pages = _pages(12, 0)  # 12 % 8 != 0
        with self.assertRaisesRegex(ValueError, "not a multiple of granularity"):
            pages.to(DEVICE_NAME, dynamic=POOL_SPEC)


# ------------------------------------------------------------------ unit tests (no device compile)


class TestIndirectIndexRange(unittest.TestCase):
    def test_concrete_size_passes_through(self):
        self.assertEqual(pass_utils.indirect_index_range(sympy.Integer(8)), 8)
        self.assertEqual(pass_utils.indirect_index_range(8), 8)

    def test_symbolic_size_resolves_to_declared_maximum(self):
        s = sympy.Symbol("s7", integer=True, positive=True)
        with patch.object(pass_utils, "finite_upper_or_none", return_value=16):
            self.assertEqual(pass_utils.indirect_index_range(s), 16)

    def test_symbolic_size_without_maximum_raises_naming_the_symbol(self):
        s = sympy.Symbol("s7", integer=True, positive=True)
        with patch.object(pass_utils, "finite_upper_or_none", return_value=None):
            with self.assertRaisesRegex(Unsupported, "s7"):
                pass_utils.indirect_index_range(s)
            with self.assertRaisesRegex(Unsupported, UNDECLARED_MAX_REFUSAL):
                pass_utils.indirect_index_range(s)


class TestPinFixedCopyExtent(unittest.TestCase):
    def test_symbolic_extent_without_maximum_is_refused(self):
        """The fixed in-loop copy refuses an extent it cannot size, with the
        same advice as the index-range refusal."""
        s = sympy.Symbol("s9", integer=True, positive=True)
        restick_buff = SimpleNamespace(
            data=SimpleNamespace(ranges=[s, PAGE_SIZE, PAGE_HS]),
            get_name=lambda: "buf_restick",
        )
        with patch.object(insert_restickify, "finite_upper_or_none", return_value=None):
            with self.assertRaisesRegex(Unsupported, UNDECLARED_MAX_REFUSAL):
                insert_restickify._pin_fixed_copy_extent_to_declared_max(restick_buff)
            with self.assertRaisesRegex(Unsupported, r"buf_restick.*s9"):
                insert_restickify._pin_fixed_copy_extent_to_declared_max(restick_buff)

    def test_static_extent_is_left_alone(self):
        restick_buff = SimpleNamespace(
            data=SimpleNamespace(ranges=[sympy.Integer(16), PAGE_SIZE, PAGE_HS]),
            get_name=lambda: "buf_restick",
        )
        insert_restickify._pin_fixed_copy_extent_to_declared_max(restick_buff)
        self.assertEqual(list(restick_buff.data.ranges), [16, PAGE_SIZE, PAGE_HS])

    def test_clears_every_cached_method(self):
        """The pin rewrites ``ranges``, so it must drop every cached method of
        the restickify's Pointwise data and ComputedBuffer. The expected set is
        derived from the IR classes, not copied from the pin, so a cached method
        added upstream fails here instead of surviving as a stale extent."""

        def cache_keys(cls):
            return {
                coarse_tile._cache_key(getattr(cls, name))
                for name in dir(cls)
                if hasattr(getattr(cls, name, None), "clear_cache")
            }

        s = sympy.Symbol("s9", integer=True, positive=True)
        restick_buff = SimpleNamespace(
            data=SimpleNamespace(ranges=[s, PAGE_SIZE, PAGE_HS]),
            layout=None,
            get_name=lambda: "buf_restick",
        )
        cleared = []
        with (
            patch.object(insert_restickify, "finite_upper_or_none", return_value=16),
            patch.object(
                coarse_tile,
                "_clear_cache",
                side_effect=lambda obj, key: cleared.append((obj, key)),
            ),
        ):
            insert_restickify._pin_fixed_copy_extent_to_declared_max(restick_buff)

        self.assertEqual(list(restick_buff.data.ranges), [16, PAGE_SIZE, PAGE_HS])
        self.assertEqual(
            {key for obj, key in cleared if obj is restick_buff.data},
            cache_keys(ir.Pointwise),
        )
        self.assertEqual(
            {key for obj, key in cleared if obj is restick_buff},
            cache_keys(ir.ComputedBuffer),
        )


if __name__ == "__main__":
    unittest.main()
