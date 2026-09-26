# Spyre Tensor Layout Assignment and Optimization

This document describes how the Spyre compiler assigns on-device memory layouts
to tensors and ensures hardware stick-compatibility constraints are met across
the op graph.

---

## Terminology

- **Stick** — a 128-byte aligned chunk of contiguous elements; the stick dimension is always the innermost device dimension.
- **Stick variable** — an iteration variable (from an op's `var_ranges`) that indexes the stick dimension — i.e. the variable that appears in the last dimension of `device_coordinates` for an op's memory access.
- **SpyreTensorLayout** (STL) — fully describes a tensor's on-device storage, including its stick.
- **Restickify** — a data-movement op that copies a tensor to achieve a  different stick arrangement.
- **Stick compatibility** — each op imposes stick constraints on its inputs and output. When a constraint is not met, a restickify is required to resolve it.
- **Layout optimization** — selecting an output STL for each op that satisfies its stick constraints while minimizing total restickify cost.

---

## Passes

The following passes implement the layout assignment pipeline, in order:

| Pass | File | Purpose |
|------|------|---------|
| `propagate_spyre_tensor_layouts` | `propagate_layouts.py` | Forward propagation: assign *sets* of candidate STLs to each op's output |
| `optimize_restickify_locations` | `optimize_restickify.py` | Layout selection: reduce each candidate set to one committed STL, minimizing total restickify cost |
| `finalize_layouts` | `insert_restickify.py` | Convert committed STLs to `FixedTiledLayout` and build the restickify insertion plan |
| `insert_restickify` | `insert_restickify.py` | Insert restickify ops into the graph |
| `reorder_nonstick_dims` | `nonstick_dim_order.py` | Reorder non-stick device dimensions for performance (independent of stick compatibility) |

---

## Pass 1 — Layout Propagation (`propagate_spyre_tensor_layouts`)

Layout propagation is a forward data-flow pass over the op graph. At each node
it takes candidate STL sets of its inputs and
derives the set of candidate STLs for that node's output. Passing sets
forward — rather than committing to one layout immediately — is what gives the
downstream optimizer room to make a globally-better choice.

`propagate_layouts` does not enumerate all possible layouts. It intentionally excludes layouts unlikely to be helpful,
keeping the solution space manageable for the downstream optimizer. The optimizer can only select from the propagated
layouts, so excluding a layout during propagation means it cannot be chosen later — even if it would yield the globally
optimal solution. Additional candidate STLs should be added if evidence shows they are needed, or if the optimizer
improves to the point where the larger state space is manageable.

### Stick Compatibility and Restickify Cost

Three node types encode the different classes stick-compatibility constraints and the cost
to bring incompatibilities into compliance via restickify.

**`AllSameNode`** — used for pointwise ops and most reductions. All inputs
and the output must share a stick. The cost is the sum of per-input
restickify costs for each input/output pair that is not stick compatible.

**`FixedInOutNode`** — used for `matmul`, `conv`, `exx2`, `layernormnorm`,
`keep_by_index`, and `aten.clone` with a stick offset. Each input and the
output have a fixed required stick; any operand that arrives with the wrong
stick triggers a restickify. `matmul`'s two inputs, for example, require
different stick variables.

**`AnyInNode`** — used for `clone`, no-ops, constants, and similar. Accepts
any input at zero cost and introduces no optimization constraints.

**Mutation ops.** Mutation ops create additional stick constraints: they have
two output dependencies — their logical output and the target buffer they write
into — and both must be committed to the same STL. The optimizer enforces this
via a co-output edge: INF cost for any candidate where the two disagree.

### Candidate generation

`propagate_layouts` walks the graph in topological order, seeding graph inputs
from their actual on-device tensor layout. Each subsequent op is assigned a set
of candidate output STLs (`op.layouts`) and a `restick_cost_fn` from one of
the three types above. The sections below describe how candidates are derived
for each op type.

#### Single-arg ops

The default strategy is to preserve the input stick as the output stick —
threading the same loop variable through the op costs nothing. If the input
stick expression maps cleanly to an output dimension and produces an
offset-free output stick, that single candidate is returned. If it doesn't
(e.g. the dimension disappears, or the mapped output stick acquires a constant
offset from slicing), the pass scans all output dimensions as alternatives.

**Reductions** have one important exception: some reduction ops
(`REDUCTIONS_NON_STICK_DIM_ONLY`) cannot reduce along the stick dimension.
When the input stick carries the reduction variable, the input layout is not
preserved — only surviving (non-reduced) dimensions are offered as output
stick candidates.

**Type conversions** that change `elems_per_stick` (e.g. fp16↔fp32,
fp8→fp16) need special handling because the stick depth changes. Staggered
element arrangements (DL16 ↔ FP32, used for RMSNorm upcasts) propagate by
rescaling the device layout's stick depth in place, preserving any padding in
the input layout. Plain conversions rebuild a fresh dense layout from the
output host size, which avoids propagating degenerate device sizes that can
appear after QFP8 quantization rescaling.

#### Multi-arg pointwise

Multi-arg pointwise ops have multiple inputs that must all be stick-compatible
with the output. The pass collects the stick expressions from all input candidates (excluding
those with a constant offset) and tries each as a candidate output stick. Dimensions whose size is divisible by the stick size are preferred as the
stick; unaligned dimensions are only added when no aligned candidate exists.

**Same-layout pass-through.** When all inputs have the same size, index
expression, and element size as the output, the output device layout is copied
from the input exactly as-is. This preserves the exact device layout, including
`ElementArrangement` and non-stick dimension ordering.

**ElementArrangement propagation.** Most ops use the `STANDARD` arrangement.
When a staggered EA (e.g. `DL16_TO_FP32`) is present on any input, it
propagates to the output and `STANDARD` candidates are suppressed.

#### Fixed-requirement ops (`matmul`, `conv`, `exx2`, `layernormnorm`)

These ops have hardware-mandated stick requirements that leave no room for the
optimizer to choose:

- **`matmul`**: input1 must stick on the reduction variable (K); input2 and the
  output must stick on the generated variable (N).
- **`conv2d`**: same structure — activation sticks on the input-channel
  contraction variable, weight and output stick on the out-channel variable.
- **`exx2` / `layernormnorm`**: input must stick on the last (reduction) dimension.

---

## Pass 2 — Restickify Optimization (`optimize_restickify_locations`)

Given the candidate STL sets and cost nodes established during propagation, the beam
optimizer commits one STL per op (written to `op.committed_stl`) with the goal of minimizing
total restickify cost.

### STL candidates as stick-dimension proxies

An important subtlety: the candidate STLs produced by propagation based only
on **which device dimension is mapped to the stick**. The non-stick dimension
ordering in a candidate STL is not something the
optimizer reasons about or optimizes over. Each STL candidate is a
proxy for the set of all valid layouts that place the stick on a particular
loop variable, and the non-stick dimensions are just along for the ride.

The optimizer works correctly under this interpretation because the cost
function it calls — `compute_restickify_needed` / `stick_compatible` in
`pass_utils.py` — only checks **stick compatibility**, not full STL equality.
`stick_compatible()` returns true when all tensors' stick expressions share at
most one iteration variable and that variable doesn't appear in any non-stick
coordinate. Two STLs with different non-stick orderings but the same stick
variable are therefore judged compatible and assigned cost 0, even though they
are not identical layouts.

### Cost model

Restickify runtime is broadly linear in tensor size, so device element count
is used as the cost metric. Each input edge carries an `EdgeCostMap` that
lazily computes the cost of pairing a producer's candidate STL with a
consumer's candidate STL. The result is one of three values:

- **0** — sticks are already compatible; no restickify needed
- **element count** — a restickify is needed; cost is the total device element count of the input buffer
- **`INF`** — no restickify can make them compatible

Costs are cached so the optimizer can query the same pair repeatedly without
recomputation.

The optimizer operates over the same three cost node types (`AllSameNode`,
`FixedInOutNode`, `AnyInNode`) defined above. See
[Stick Compatibility and Restickify Cost](#stick-compatibility-and-restickify-cost) above.

---

### Global beam optimizer

The optimizer must assign one STL to every op in the graph such that the
total restickify cost is minimized. The choices interact: picking a particular
stick at one op affects what sticks are compatible at downstream join points.
The beam optimizer navigates the exponential space by maintaining a frontier
of up to `BEAM_WIDTH` states — each an assignment of STLs covering all ops
seen so far — advancing in topological order. At each op, every state branches
into one new state per candidate STL. At the end, the lowest-cost state wins
and its assignments are committed.

If no feasible output layout exists for an op (all candidates have `INF` cost), `_no_feasible_layout_error` builds a detailed error message for debugging.

### The beam width tradeoff

Beam width is the fundamental accuracy/tractability tradeoff. After each op,
the frontier is sorted by lower bound and trimmed to at most `BEAM_WIDTH`
states. A wider beam is more likely to find the global optimum; a narrower
beam finishes faster but may discard the optimal path if it gets trimmed because it looked poor initially.

### Pruning optimizations

The optimizer performs two additional optimizations to reduce the state space (improving both runtime and memory usage) by pruning provably non-optimal
states.  Note this differs from the previously discussed `BEAM_WIDTH` and `propagate_layouts` heuristics, which may prevent the optimal solution from being found.  The optimizations below should never drop an optimal solution.

**Liveness merge.** If two states
differ only in the STLs of dead buffers — buffers no future op will read —
the state with the higher cost can never win, thus can be discarded.  In graphs with many short-lived
intermediates, liveness merge can eliminate a large fraction of states, helping prevent an optimal state from exceeding `BEAM_WIDTH` and being trimmed.

**Backward DP lower bound.** Before the forward pass runs, a backward pass
walks the graph in reverse topological order and computes, for every
`(op, candidate_stl)` pair, the minimum remaining restickify cost achievable
from that choice onward.  This is a form of look-ahead, to avoid preferring choices early that it is easy to see will become poor later.

This is done by adding a "future" component of each
hypothesis's lower bound: `lower_bound = cost_so_far + future_min_cost`. The
beam is sorted and trimmed by lower bound rather than actual cost, so a
hypothesis whose current choices lead toward an expensive or infeasible join
downstream is de-prioritized even if its cost-so-far looks cheap.

The bound is conservative because each downstream
consumer's minimum cost is computed independently, ignoring cross-consumer
constraints. That conservatism is deliberate: an overestimating bound could
prune the optimal path from the beam.

---

## Pass 3 — Finalization and Restickify Insertion

### `finalize_layouts`

Finalization runs in two steps. First, every op's `committed_stl` is frozen
into a `FixedTiledLayout` and assigned as the op's permanent layout —
optimizer-only attributes are cleaned up. Once all layouts are frozen, each
input edge is checked: if the producer's committed STL is incompatible with
the consumer's required STL, a restickify is recorded in
`graph.restickify_plan`.

### `insert_restickify`

Consumes `graph.restickify_plan` and splices the required restickify ops into
`graph.operations` before their consumers. Each restickify is lowered as a
real `spyre.restickify` IR node. The consumer op's `inner_fn` is then patched
via `NameSwapHandler` — a `WrapperHandler` subclass that intercepts buffer
loads and redirects them to the new restickified buffer, without touching any
index expressions. The consumer `ComputedBuffer` is reconstructed as a fresh
object to invalidate any cached size/body derived from the old inputs.

### `insert_post_mutation_restickify`

A mutation op (`MutationLayoutSHOULDREMOVE`) writes its output directly into
an existing target buffer rather than allocating a new one — it is an in-place
write.  These need to be handled separately because `insert_restickify` pass runs before scheduling, but inductor requires an mutation ops to have their `FixedLayout` (rather than `FixedTiledLayout`) during scheduling. The solution is to have a separate pass, `insert_post_mutation_restickify`, handle mutation ops after scheduling has run.  

---

## Pass 4 — Non-stick Dimension Reorder (`reorder_nonstick_dims`)

Layout propagation varies only which dimension is the stick; non-stick
dimension ordering is not optimized. `reorder_nonstick_dims` is a pass that focuses on selecting the order of non-stick dimensions.  It does not impact or change stick compatibility.

The current pass focuses only on improving `matmul` performance by swapping the largest non-stick device dimension into the slot
between the two stick dimensions (`outer_stick+1`).  However this pass will be expanded in future work.

NOTE: this pass is currently executed after propagate_layouts and before `optimize_restickify`; however the code will soon align with the order written in this document.  `reorder_nonstick_dims` does not impact stick decisions or `optimize_restickify` in any way, so executing it between those passes is confusing and unnecessary.

---

## Full pass sequence

```
propagate_spyre_tensor_layouts(graph)
    → op.layouts + op.restick_cost_fn for every op

optimize_restickify_locations(graph)
    → op.committed_stl for every op

finalize_layouts(graph)
    → op.layout = FixedTiledLayout(committed_stl) for every op
    → graph.restickify_plan = {op_name: [{arg_name, target_layout}, ...], ...}

insert_restickify(graph)
    → splices restickify ComputedBuffers before affected consumer ops
    → patches consumer inner_fn via NameSwapHandler

insert_post_mutation_restickify(graph)
    → inserts pre/post ops for offset-mutation edge cases

reorder_nonstick_dims(graph)
    → rewrites non-stick dimensions - no impact on stick compatibility
```

---

## Configuration

| Symbol | Default | Effect |
|--------|---------|--------|
| `BEAM_WIDTH` | `200` | Maximum beam states retained after each op expansion |
| `MAX_BEAM_STATES_LOGGED` | `10` | Number of states logged at DEBUG level per step |
