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

"""Lowering tests for `for_each_tile`: the loop must reach a `while_loop`, and copy nothing.

Nine cases:

  A     tile M as a map: step `i` takes a row band of X, sees Y whole, and its result
        tile is laid along dim 0 of the output. The map level in its smallest form.
  B     the same loop with the tiles landing along dim 1 instead (N tiled, X invariant),
        which is what makes `scan`'s dim-0 stack cost a second full-size copy.
  C     split-K matmul: co-indexed `dims=(-1, 0)` on the shared axis, carry accumulates.
        A plain-`scan` formulation has to copy the whole operand to manufacture a
        leading tile axis; this one does not, which is exactly the HBM traffic working
        set reduction exists to remove.
  D     the same matmul with several axes tiled at once -- (M,K), (K,N), (M,K,N) at equal
        and at mixed tile sizes -- one while_loop level per axis, nested map around
        reduction. Variants through one nest builder.
  E     case D's (M,K) nest again, as the readable reference for what map mode costs next
        to a reduction: the tile counter carry, the tile write, and the fold that is free
        at `out_dim=0`.
  F     split-K whose carry is `(counter, acc)`, with a closed-over `Z` indexed by the
        counter each step: how a body learns its own tile index without a tile spec.
  PA_B  a miniature paged attention with both of the body's matmuls tiled along their
        contraction axis, i.e. case C nested inside a page loop: three while_loop levels
        in one kernel.
  PA_C  PA_B stripped to a single page, so the page loop is gone and only the two matmul
        levels remain: no Gather, no mask stack, no online-softmax carry (all three are
        ACTIVE > 1 machinery). Read PA_C before PA_B to see the matmul levels alone.
  PA_D  the complement of PA_C: the page loop and nothing else. Gather K and V for the
        pages `page_index` names, add a constant, accumulate. Read it for the Gather
        alone -- it copies nothing at all.

Each case asserts three things about the post-grad graphs, `while_loop` subgraphs
included:

  * one `while_loop` node per nest level -- the loop was lifted, not unrolled -- and no
    surviving `scan` node;
  * NO materialization node (`copy`, `clone`, `contiguous`, `cat`, `stack`) beyond the
    ones listed per case, each of which is something the caller asked for;
  * the numbers, against eager and against a dense reference.

The gathered cases take their page index already contiguous, so nothing the caller does
to a stick-wide page table lands in the graph. `for_each_tile` raises on a strided index
instead of copying it silently, because the index table becomes a while_loop carry whose
strides would be re-established every step.

The graph is the right instrument for the copy count rather than the generated wrapper:
inductor names a fused kernel after every op it fused (`cpp_fused_add_copy_...`), and
the while_loop prologue clones every carried operand under `if not should_loop:`, so
grepping the wrapper text counts copies that either do not exist or are not ours.

Every tensor stays on CPU and the compiled backend is stock inductor: what is under test
is the graph `for_each_tile` traces to, which is device-independent, so no Spyre device
and no Spyre backend compiler is required.

Run:

    python tests/inductor/test_for_each_tile.py
    python tests/inductor/test_for_each_tile.py TestForEachTileLowering.test_split_k
    SPYRE_FOR_EACH_TILE_DUMP=/tmp/fet python tests/inductor/test_for_each_tile.py
"""

import contextlib
import os
import unittest

import torch
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor.wsr import for_each_tile, Gather


DUMP_DIR = os.environ.get("SPYRE_FOR_EACH_TILE_DUMP")

# Matched as substrings of the node target. `cat`/`stack` are in here beside the copies:
# assembling one tensor out of several is the other way a tiling loop can end up paying
# for data movement it did not need (the mask list PA_B builds).
MATERIALIZE = ("copy", "copies", "clone", "contiguous", "cat", "stack")


@contextlib.contextmanager
def post_grad_graphs():
    """Capture each post-grad graph right after `decompose_scan_to_while_loop` ran.

    `post_grad_custom_post_pass` fires BEFORE that decomposition, so a custom pass
    cannot see the while_loop; wrapping the decomposition itself can.
    """
    import torch._inductor.fx_passes.post_grad as pg

    seen: list[torch.fx.GraphModule] = []
    original = pg.decompose_scan_to_while_loop

    def wrapper(gm):
        out = original(gm)
        seen.append(gm)
        return out

    pg.decompose_scan_to_while_loop = wrapper
    try:
        yield seen
    finally:
        pg.decompose_scan_to_while_loop = original


def live_nodes(gm, depth=0):
    """(depth, target) for every call_function `gm` runs, subgraphs of its loops included.

    Depth is the nesting level, so a nested loop's body shows up below its parent's.
    Subgraphs are reached through the HOP's own arguments rather than through every
    `get_attr`: after the decomposition the module still carries the original
    `scan_combine_graph_*` attribute, a dead copy of what became the loop body, and
    walking that would count every node in the nest twice.
    """
    found: list[tuple[int, str]] = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        found.append((depth, str(node.target)))
        for arg in node.args:
            sub = arg
            if isinstance(sub, torch.fx.Node) and sub.op == "get_attr":
                sub = getattr(gm, str(sub.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                found.extend(live_nodes(sub, depth + 1))
    return found


def write_dump(name, header, graphs, code):
    """Write every captured graph and the COMPLETE generated wrapper to one file.

    Nothing is filtered: this is the whole compiled artifact, to be diffed across cases
    or handed to a backend author.
    """
    bar = "=" * 78
    parts = ["\n".join(header) + "\n"]
    parts += [
        f"\n{bar}\npost-grad graph {n} (after decompose_scan_to_while_loop)\n{bar}\n"
        + gm.print_readable(print_output=False, include_stride=True, include_device=False)
        + "\n"
        for n, gm in enumerate(graphs)
    ]
    parts += [
        f"\n{bar}\ngenerated wrapper {n} (complete)\n{bar}\n{c}\n"
        for n, c in enumerate(code)
    ]
    os.makedirs(DUMP_DIR, exist_ok=True)
    path = os.path.join(DUMP_DIR, f"{name}.txt")
    with open(path, "w") as f:
        f.write("".join(parts))
    return path


# ======================================================================
# Problem setups.
# ======================================================================

# Matmul settings:
M, K, N = 8, 12, 6

# Paged attention settings: a 6-page pool with 3 pages active. Shared by PA_B, PA_C, PA_D.
# Names and roles follow vLLM's; the sizes are shrunk so codegen stays readable.
NKV = 2  # KV heads, i.e. GQA groups. q, k, v and the carry all keep this leading.
NQ = 2  # query heads per KV head, so HEADS = NKV * NQ overall. k/v broadcast over it.
PQL = 3  # query tokens in this block (1 for decode, the prefill tile size otherwise).
HS = 4  # head size: the contraction axis of both matmuls. The page loop never tiles it.
BLOCK = 4  # KV tokens per page: a score tile's key axis, and the loop's tile width.
POOL_PAGES = 6  # pages in the whole cache. Only Gather sees this; the loop does not.
ACTIVE = 3  # pages this sequence occupies = the trip count = len(mask_tiles).
HEADS = NKV * NQ
SCALE = 1.0 / (HS**0.5)

# The operands those give, and how each is tiled by the outer page loop:
#
#   q            (NKV, NQ, PQL, HS)          None    invariant, read whole every step
#   k_pool       (POOL_PAGES, BLOCK, NKV, HS) Gather(0, page_index)  one page per step
#   v_pool       (POOL_PAGES, BLOCK, NKV, HS) Gather(0, page_index)  one page per step
#   mask         (ACTIVE, PQL, BLOCK)        0       one [PQL, BLOCK] tile per step
#   carry        (NKV, NQ, PQL, 1) x2 + (NKV, NQ, PQL, HS)   online-softmax (m, l, o)
#   result       (PQL, HEADS, HS)            the carry's o, reheaded
#
# k_pool/v_pool are one tensor each, as in the real kernel, so Gather is faithful: the
# per-step page is an index_select of ACTIVE-many rows out of POOL_PAGES. Only `mask`
# is stacked from a list here, because a per-step operand has to be one tensor.


def matmul_inputs():
    torch.manual_seed(0)
    X = torch.randn(M, K)
    Y = torch.randn(K, N)
    return (X, Y), X @ Y


def nest(axes, tM, tK, tN):
    """A matmul with any subset of {M, K, N} tiled, one `for_each_tile` level each.

    Level order is fixed M outside N outside K, so the two map levels place their tiles
    around a reduction that accumulates into the block they place. Each level sees the
    tiles of the one above it, which is what makes the nest compose: tiles are
    rank-preserving, so `dims` means the same thing at every depth and a level cannot
    tell whether its operands are the originals or somebody else's tiles.
    """

    def k_level(x, y):
        if "K" not in axes:
            return x @ y

        def body(acc, ops):
            x_tile, y_tile = ops
            return acc + x_tile @ y_tile, None

        final, _ = for_each_tile(
            body,
            (x, y),
            dims=(-1, 0),
            tile_size=tK,
            init=torch.zeros(x.shape[0], y.shape[1]),
        )
        return final

    def n_level(x, y):
        if "N" not in axes:
            return k_level(x, y)

        def body(_, ops):
            x_whole, y_tile = ops
            return None, k_level(x_whole, y_tile)

        _, out = for_each_tile(body, (x, y), dims=(None, 1), tile_size=tN, out_dim=1)
        return out

    def fn(X, Y):
        if "M" not in axes:
            return n_level(X, Y)

        def body(_, ops):
            x_tile, y_whole = ops
            return None, n_level(x_tile, y_whole)

        _, out = for_each_tile(body, (X, Y), dims=(0, None), tile_size=tM, out_dim=0)
        return out

    return fn


def paged_attn_inputs():
    """(q, k_pool, v_pool, page_index, mask_tiles) and the dense-softmax reference.

    Shaped the way the real kernel is called: the page table is stick-wide int32 with
    the page index in column 0, and the mask tiles carry `finfo.min` (not -inf) on a
    padded tail, as vLLM's do.

    The page index is handed over already contiguous, so the column extraction is the
    caller's business and stays out of the compiled region: `for_each_tile` requires a
    contiguous `Gather` index and raises otherwise (the table becomes a while_loop
    carry whose strides are re-established every step, so the frontend makes the cost
    visible where it is paid rather than hiding a copy). What the graph then shows is
    the tiling alone.
    """
    torch.manual_seed(0)
    q = torch.randn(NKV, NQ, PQL, HS)
    k_pool = torch.randn(POOL_PAGES, BLOCK, NKV, HS)
    v_pool = torch.randn(POOL_PAGES, BLOCK, NKV, HS)
    page_table = torch.zeros(ACTIVE, 32, dtype=torch.int32)
    page_table[:, 0] = torch.tensor([4, 1, 5], dtype=torch.int32)
    mask_tiles = [torch.zeros(PQL, BLOCK) for _ in range(ACTIVE)]
    mask_tiles[-1][:, -2:] = torch.finfo(q.dtype).min

    page_index = page_table[:ACTIVE, 0].contiguous()
    kc = k_pool.index_select(0, page_index).reshape(ACTIVE * BLOCK, NKV, HS)
    vc = v_pool.index_select(0, page_index).reshape(ACTIVE * BLOCK, NKV, HS)
    k4, v4 = kc.permute(1, 0, 2).unsqueeze(1), vc.permute(1, 0, 2).unsqueeze(1)
    scores = torch.matmul(q, k4.transpose(-2, -1)) * SCALE + torch.cat(mask_tiles, -1)
    ref = torch.matmul(torch.softmax(scores, -1), v4)
    ref = ref.reshape(1, HEADS, PQL, HS).transpose(1, 2).reshape(PQL, HEADS, HS)
    return (q, k_pool, v_pool, page_index, mask_tiles), ref


def single_page_inputs():
    """One page's worth of the above: `(q, k_page, v_page, mask_tile)` and the reference.

    No pool and no page table, because there is nothing to select: the page is handed
    over as the `(BLOCK, NKV, HS)` tile the page loop would have produced. The mask is
    one `(PQL, BLOCK)` tile, so a list never arises either (G4 is an `ACTIVE > 1`
    problem).
    """
    torch.manual_seed(0)
    q = torch.randn(NKV, NQ, PQL, HS)
    k_page = torch.randn(BLOCK, NKV, HS)
    v_page = torch.randn(BLOCK, NKV, HS)
    mask_tile = torch.zeros(PQL, BLOCK)
    mask_tile[:, -2:] = torch.finfo(q.dtype).min

    k4 = k_page.permute(1, 0, 2).unsqueeze(1)
    v4 = v_page.permute(1, 0, 2).unsqueeze(1)
    scores = torch.matmul(q, k4.transpose(-2, -1)) * SCALE + mask_tile
    ref = torch.matmul(torch.softmax(scores, -1), v4)
    ref = ref.reshape(1, HEADS, PQL, HS).transpose(1, 2).reshape(PQL, HEADS, HS)
    return (q, k_page, v_page, mask_tile), ref


# ======================================================================
# The tests.
# ======================================================================

class TestForEachTileLowering(unittest.TestCase):
    def _assert_lowers(self, fn, args, ref, *, name, loops, materializations=0):
        """Compile `fn` and check the loop count, the copy count and the numbers.

        `loops` is one `while_loop` node per nest level; a level that unrolled instead
        of lowering shows up as a smaller count. `materializations` is how many
        copy/clone/contiguous nodes the case is allowed, each explained in its test.
        """
        torch._dynamo.reset()
        eager = fn(*args)
        with post_grad_graphs() as graphs:
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            out, code = run_and_get_code(compiled, *args)

        # The decomposition runs bottom-up, one module at a time, so the graph captured
        # last is the top-level one and the rest are the subgraphs it now contains.
        nodes = live_nodes(graphs[-1])
        found_loops = [t for _, t in nodes if "while_loop" in t]
        scans = sorted({t for _, t in nodes if t.endswith("scan")})
        copies = [(d, t) for d, t in nodes if any(m in t for m in MATERIALIZE)]
        joined = "\n".join(code)

        if DUMP_DIR is not None:
            header = [
                name,
                f"while_loop nodes: {len(found_loops)}/{loops}",
                f"surviving scan nodes: {scans or 'none'}",
                f"materializations: {copies or 'none'}",
            ]
            print(f"wrote {write_dump(name, header, graphs, code)}")

        torch.testing.assert_close(out, eager)
        torch.testing.assert_close(out, ref)
        self.assertEqual(len(found_loops), loops, f"while_loop nodes: {found_loops}")
        self.assertEqual(scans, [], f"a scan survived the decomposition: {scans}")
        self.assertEqual(len(copies), materializations, f"materializations: {copies}")
        # Every level is a real sequential driver in the wrapper, not a rolled-out body.
        # Counted on the driver rather than on `def while_loop_body_graph_*`, whose names
        # nest (`..._body_graph_0_0_while_loop_cond_graph_0`) and would over-count.
        self.assertIn("while_loop_body_graph", joined)
        self.assertEqual(joined.count("while should_loop:"), loops)

    def test_split_m(self):
        """A: tile M as a map -- independent output tiles, Y invariant, nothing contracted.

        The map level at its smallest: step `i` takes X's `(2, K)` row band, sees Y whole,
        and its `(2, N)` result is laid at `narrow(0, 2*i, 2)` of the output. No step needs
        anything from the one before it, so the carry the frontend threads is only the tile
        counter `scan` insists on (case E spells that out).

        One materialization, and it IS the output: scan pre-allocates the stacked `ys` and
        each step writes its tile in with `select` + `copy_`. The fold afterwards is free
        -- `out_dim=0` merges scan's step axis into an axis that is already outermost, so
        the `(4, 2, 6)` stack and the `(8, 6)` result share storage.
        """
        args, ref = matmul_inputs()

        def fn(X, Y):
            def body(_, ops):
                x_tile, y_whole = ops
                return None, x_tile @ y_whole

            _, out = for_each_tile(body, (X, Y), dims=(0, None), tile_size=2, out_dim=0)
            return out

        self._assert_lowers(
            fn, args, ref, name="A_split_m", loops=1, materializations=1
        )

    def test_split_n(self):
        """B: tile N as a map -- case A with the output axis moved, X invariant.

        Step `i` takes Y's `(K, 2)` column band and writes its `(M, 2)` result at
        `narrow(1, 2*i, 2)`, so which axis the tiles land along is the only difference from
        A. It costs one materialization more, and that one is nobody's output: `scan`
        always stacks on dim 0, so the `(3, 8, 2)` stack has to be folded to `(8, 6)`, and
        that flatten crosses the moved axis, which strides cannot express -- a full-size
        copy of every output byte on top of the write. Only a lowered `output=` on `scan`
        removes it; the frontend cannot (torch-spyre#3965, 5.6).
        """
        args, ref = matmul_inputs()

        def fn(X, Y):
            def body(_, ops):
                x_whole, y_tile = ops
                return None, x_whole @ y_tile

            _, out = for_each_tile(body, (X, Y), dims=(None, 1), tile_size=2, out_dim=1)
            return out

        self._assert_lowers(
            fn, args, ref, name="B_split_n", loops=1, materializations=2
        )

    def test_split_k(self):
        """C: co-indexed `dims=(-1, 0)` on the shared axis, carry accumulates.

        No reshape, no `.contiguous()`, no copy of any kind -- compare a plain `scan`,
        which stacks X to get a tile axis. Both tiles are
        `reinterpret_tensor(arg, ..., <stride>*u0)` off the original operand, and X's
        keeps X's own strides.
        """
        args, ref = matmul_inputs()

        def fn(X, Y):
            def body(acc, ops):
                x_tile, y_tile = ops
                return acc + x_tile @ y_tile, None

            final, _ = for_each_tile(
                body, (X, Y), dims=(-1, 0), tile_size=3, init=torch.zeros(M, N)
            )
            return final

        self._assert_lowers(fn, args, ref, name="C_split_k", loops=1)

    NEST_AXES = ("MK", "KN", "MKN")
    NEST_SIZES = ("same", "mixed")

    def test_nested_matmul_tilings(self):
        """D: (M,K), (K,N) or (M,K,N) tiled at once, one while_loop level per axis.

        The copy inventory is a property of which levels a nest HAS, not of how deep it
        is or what tile sizes it uses:

          * a K reduction level costs nothing -- pure addressing, as in case C;
          * each map level costs one `copy_`, the write of the step's tile into scan's
            stacked output. That IS the output, not overhead;
          * the N level costs one more, because `out_dim=1` folds scan's step axis into
            an axis that is not outermost, which is not expressible in strides (case B).
            `out_dim=0` is a pure view, so the M level pays nothing extra.

        Hence `("M" in axes) + 2 * ("N" in axes)`, independent of `sizes`.

        `sizes="mixed"` exists to check that nothing is quietly shared between levels:
        with tM=4, tK=3, tN=2 no two axes agree, and neither do the trip counts (2, 4,
        3). Every variant must agree with `X @ Y` to the same tolerance.
        """
        args, ref = matmul_inputs()
        for axes in self.NEST_AXES:
            for sizes in self.NEST_SIZES:
                with self.subTest(axes=axes, sizes=sizes):
                    tM, tK, tN = (2, 2, 2) if sizes == "same" else (4, 3, 2)
                    self._assert_lowers(
                        nest(axes, tM, tK, tN),
                        args,
                        ref,
                        name=f"D_nest_{axes}_{sizes}",
                        loops=len(axes),
                        materializations=("M" in axes) + 2 * ("N" in axes),
                    )

    def test_map_outside_reduction(self):
        """E: map M outside a split-K reduction -- the two level kinds in one nest.

        The same nest as `test_nested_matmul_tilings(axes="MK")`, deliberately: that one
        sweeps, this one is the one to read. What map mode adds over case C:

          * a carry the caller never asked for. `scan` requires one, so the frontend
            carries an int64 tile counter and advances it, which is the cheapest thing
            that is not an alias: passing an unchanged carry through would alias input to
            output, which `scan` rejects, and a `clone` to dodge that would be a copy for
            nothing.
          * one `copy_` per map level, the write of the step's `(tM, N)` tile into scan's
            stacked output -- the single materialization this case allows.
          * no copy for the fold afterwards. `out_dim=0` merges scan's step axis into an
            axis that is already outermost, so `_stacked_to_full` is a pure view: the
            `(4, 2, 6)` stack and the `(8, 6)` result share storage.

        The inner level is untouched by any of it: a reduction nested inside a map body
        lowers exactly as it does at top level, affine tiles and no copies.
        """
        args, ref = matmul_inputs()
        self._assert_lowers(
            nest("MK", 2, 3, 2),
            args,
            ref,
            name="E_map_outside_reduction",
            loops=2,
            materializations=1,
        )

    def test_counter_carry_indexes_lifted_operand(self):
        """F: an explicit loop counter as a carry, slicing a lifted parameter per tile.

        Split-K again, but the carry is `(i, acc)` with `i` an int64 scalar tensor the
        body advances itself, and `Z[i]` is added to the accumulator each step. That is
        the pattern for anything the body needs to know its own tile index for -- a
        per-tile bias, scale or mask table -- without a tile spec for it: `Z` is not an
        operand at all, it is closed over, so it reaches the loop whole as a lifted
        additional input and only the row the step needs is `index_select`ed inside.

        Zero materializations: the counter is a scalar the body computes, `Z` is passed
        by reference rather than sliced by the frontend, and the reduction carry needs no
        buffer of its own. Note the counter counts STEPS, not tile indices, so under
        `reverse=True` `Z[i]` would pair with the tiles in the opposite order.
        """
        (X, Y), _ = matmul_inputs()
        TK = 3
        nK = K // TK
        Z = torch.randn(nK, M, N)
        ref = X @ Y + Z.sum(0)

        def loop(X, Y, Z):
            def body(carry, ops):
                i, acc = carry
                x_tile, y_tile = ops
                z = Z.index_select(0, i.reshape(1)).squeeze(0)
                return (i + 1, acc + x_tile @ y_tile + z), None

            return for_each_tile(
                body,
                (X, Y),
                dims=(-1, 0),
                tile_size=TK,
                init=(torch.zeros((), dtype=torch.int64), torch.zeros(M, N)),
            )

        # The counter is a real output, not a placeholder, so check it in its own right:
        # `_assert_lowers` compares one tensor against the reference.
        (count, _), _ = loop(X, Y, Z)
        self.assertEqual(int(count), nK)

        def fn(X, Y, Z):
            (_, final), _ = loop(X, Y, Z)
            return final

        self._assert_lowers(fn, (X, Y, Z), ref, name="F_counter_carry", loops=1)

    def test_paged_attention_nested_matmuls_inner2(self):
        """PA_B at 2 tiles per inner level."""
        self._paged_attention_nested_matmuls(2)

    def test_paged_attention_nested_matmuls_inner4(self):
        """PA_B at 4 tiles per inner level."""
        self._paged_attention_nested_matmuls(4)

    def _paged_attention_nested_matmuls(self, inner):
        """PA_B: both of the body's matmuls tiled inside the page loop, three levels.

        Both inner levels tile the matmul's CONTRACTION axis and accumulate in a carry,
        i.e. case C one level down, inside the page body:

            Q@K^T  over head_size            dims=(-1, -1) (the axis is last on both)
            P@V    over the page token axis  dims=(-1, -2)

        The `P@V` level is seeded with the rescaled accumulator rather than a zero
        buffer, so its tiles land straight in the page loop's carry and no full-size
        temporary is allocated. `inner` is the trip count of both levels and must divide
        both tiled axes.

        Three materializations, none of them the tiling's. One sits outside every loop
        and is the caller's own: the `torch.stack` of the mask list -- a per-step operand
        has to be one tensor, and at real prefill shapes that copy is score-matrix
        sized. The page index arrives already contiguous, so extracting
        the stick-wide table's column costs nothing here; `for_each_tile` requires that
        and raises on a strided index rather than copying behind the caller's back. The
        other two are one per inner loop body, the same GQA broadcast clone as PA_C.
        """
        args, ref = paged_attn_inputs()

        def fn(q, k_pool, v_pool, page_index, mask_tiles):
            def body(carry, ops):
                tile_max, tile_sum, tile_output = carry
                k_page, v_page, mask_tile, q_whole = ops
                k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
                v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

                def qk_body(acc, inner_ops):
                    q_tile, k_tile = inner_ops
                    return acc + torch.matmul(q_tile, k_tile.transpose(-2, -1)), None

                scores, _ = for_each_tile(
                    qk_body,
                    (q_whole, k_page_4d),
                    dims=(-1, -1),
                    tile_size=HS // inner,
                    init=torch.zeros(NKV, NQ, PQL, BLOCK),
                )
                scores = scores * SCALE + mask_tile.squeeze(0)
                new_max = torch.maximum(
                    tile_max, torch.amax(scores, dim=-1, keepdim=True)
                )
                rescale = torch.exp(tile_max - new_max)
                tile_probs = torch.exp(scores - new_max)
                new_sum = tile_sum * rescale + tile_probs.sum(-1, keepdim=True)

                def pv_body(acc, inner_ops):
                    p_tile, v_tile = inner_ops
                    return acc + torch.matmul(p_tile, v_tile), None

                out, _ = for_each_tile(
                    pv_body,
                    (tile_probs, v_page_4d),
                    dims=(-1, -2),
                    tile_size=BLOCK // inner,
                    init=tile_output * rescale,
                )
                return (new_max, new_sum, out), None

            page = Gather(0, page_index)
            # (-inf, 0, 0) makes the general update reproduce a peeled i == 0 exactly.
            init = (
                torch.full((NKV, NQ, PQL, 1), float("-inf")),
                torch.zeros((NKV, NQ, PQL, 1)),
                torch.zeros((NKV, NQ, PQL, HS)),
            )
            (_, tile_sum, tile_output), _ = for_each_tile(
                body,
                (k_pool, v_pool, torch.stack(mask_tiles), q),
                dims=(page, page, 0, None),
                tile_size=1,
                init=init,
            )
            attn = tile_output / tile_sum
            attn = attn.reshape(1, HEADS, PQL, HS).transpose(1, 2)
            return attn.reshape(PQL, HEADS, HS)

        self._assert_lowers(
            fn,
            args,
            ref,
            name=f"PA_B_nested_inner{inner}",
            loops=3,
            materializations=3,
        )

    def test_single_page_matmuls_inner2(self):
        """PA_C at 2 tiles per level."""
        self._single_page_matmuls(2)

    def test_single_page_matmuls_inner4(self):
        """PA_C at 4 tiles per level."""
        self._single_page_matmuls(4)

    def _single_page_matmuls(self, inner):
        """PA_C: PA_B stripped to one page -- the two matmul levels alone, no page loop.

        With a single active page there is nothing for the outer level to iterate, so it
        goes away and takes three things with it: the `Gather` (one page is not selected
        from a pool), the mask `torch.stack` (one mask tile is not a list), and the
        online-softmax carry. That last one is exact, not an approximation: at ACTIVE=1
        the carry enters as `(-inf, 0, 0)`, so `rescale = exp(-inf - m) = 0` zeroes both
        accumulators and the update collapses to `softmax(scores) @ V`.

        What is left is only the inner pair, each tiling its matmul's contraction axis --
        case C twice, at attention shapes -- so both levels sit at depth 0 and both get
        an affine tile. This is the case to read when the question is what a tiled
        matmul costs in codegen and not how a page reaches the body.

        `q`/`probs` reach `bmm` as `reinterpret_tensor(arg, ..., 2*u)`, a pure offset,
        while the K/V tile is copied into a contiguous buffer each step -- the one
        materialization per level, `clone(expand(...))`. That copy is not the tiling's
        doing: it is the GQA head broadcast, `unsqueeze(1)` against NQ, which `bmm`
        needs as a real batch. It is one tile's worth of bytes, i.e. the working set the
        loop exists to bound, and it stays that size however many tiles there are.
        """
        args, ref = single_page_inputs()

        def fn(q, k_page, v_page, mask_tile):
            k_page_4d = k_page.permute(1, 0, 2).unsqueeze(1)
            v_page_4d = v_page.permute(1, 0, 2).unsqueeze(1)

            def qk_body(acc, inner_ops):
                q_tile, k_tile = inner_ops
                return acc + torch.matmul(q_tile, k_tile.transpose(-2, -1)), None

            scores, _ = for_each_tile(
                qk_body,
                (q, k_page_4d),
                dims=(-1, -1),
                tile_size=HS // inner,
                init=torch.zeros(NKV, NQ, PQL, BLOCK),
            )
            probs = torch.softmax(scores * SCALE + mask_tile, -1)

            def pv_body(acc, inner_ops):
                p_tile, v_tile = inner_ops
                return acc + torch.matmul(p_tile, v_tile), None

            out, _ = for_each_tile(
                pv_body,
                (probs, v_page_4d),
                dims=(-1, -2),
                tile_size=BLOCK // inner,
                init=torch.zeros(NKV, NQ, PQL, HS),
            )
            out = out.reshape(1, HEADS, PQL, HS).transpose(1, 2)
            return out.reshape(PQL, HEADS, HS)

        self._assert_lowers(
            fn,
            args,
            ref,
            name=f"PA_C_single_page_inner{inner}",
            loops=2,
            materializations=2,
        )

    def test_page_loop_only(self):
        """PA_D: the page loop with no attention in it -- gather K and V, add, accumulate.

        PA_B's operand structure with the arithmetic removed, so what is left is exactly
        the `Gather` mechanism. Step `i` reads `page_index[i]` out of both pools and the
        body's only op is `+ BIAS`, which makes the codegen readable as a single
        question: how does one page reach the body? The answer is the two things to look
        for -- each pool enters the loop WHOLE, `(6, 4, 2, 4)`, beside a `(3,)` index
        table, and there is no affine tile at all, because a gathered index is data
        rather than an offset the loop can compute from its counter.

        The carry sums the gathered pages only because the loop needs SOME sink that
        touches every gathered element; nothing about the gather is hidden by it. The
        two pools share one index but get an index leaf each, `(3,) int32` twice, which
        is torch-spyre#3965, 5.7's un-deduplicated `Gather` index in its smallest form --
        and
        it costs no data movement, hence zero materializations: a gather plus a
        reduction is the whole loop.
        """
        BIAS = 1.0
        torch.manual_seed(0)
        k_pool = torch.randn(POOL_PAGES, BLOCK, NKV, HS)
        v_pool = torch.randn(POOL_PAGES, BLOCK, NKV, HS)
        page_table = torch.zeros(ACTIVE, 32, dtype=torch.int32)
        page_table[:, 0] = torch.tensor([4, 1, 5], dtype=torch.int32)
        page_index = page_table[:ACTIVE, 0].contiguous()

        pages = k_pool.index_select(0, page_index) + v_pool.index_select(0, page_index)
        ref = (pages + BIAS).sum(0)

        def fn(k_pool, v_pool, page_index):
            def body(acc, ops):
                k_tile, v_tile = ops
                # Gathered tiles are rank-preserving: (1, BLOCK, NKV, HS), as in PA_B.
                return acc + k_tile.squeeze(0) + v_tile.squeeze(0) + BIAS, None

            page = Gather(0, page_index)
            total, _ = for_each_tile(
                body,
                (k_pool, v_pool),
                dims=(page, page),
                tile_size=1,
                init=torch.zeros(BLOCK, NKV, HS),
            )
            return total

        self._assert_lowers(
            fn,
            (k_pool, v_pool, page_index),
            ref,
            name="PA_D_page_loop",
            loops=1,
            materializations=0,
        )


if __name__ == "__main__":
    unittest.main()
