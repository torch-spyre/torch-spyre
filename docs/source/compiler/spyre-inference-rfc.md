# RFC: Using torch-spyre's dynamic dimensions in spyre-inference

| Field | Value |
|---|---|
| Status | Draft, for discussion under the boot-time epic |
| Created | 2026-11-05 |
| Tracking | [#1432 Reduce spyre-inference boot/compile time](https://github.com/torch-spyre/spyre-inference/issues/1432), [#1515 Investigate how torch-spyre symbolic shape can reduce compile time](https://github.com/torch-spyre/spyre-inference/issues/1515) |
| Related | #1523, #1525, #1524, #1473 (encoders and vision towers), #771, #1060 (a request pays for its bucket), #1364 (compiles after warm-up), #942 (performance above 32 sequences), #713 (long context), #468 (jagged attention), #959, #566 (boot time as a metric), #1242, #1239 (the V2 runner's startup and step inputs); torch-spyre #43 (the epic), #5891, #5912, #5917, #5179, #5186 |

What this RFC is. torch-spyre is adding dynamic dimensions: a tensor dimension that varies over a declared range and is served by one compiled program instead of one program per size. This document describes the feature as it stands, where it pays and where it does not, how spyre-inference can use it, and in what order, so that any part of the plugin can adopt it the same way. It is written for the priorities stated on the epic: compile-time reduction first, which here means fewer programs through fewer buckets, and end-to-end throughput second. It is a guide, not a description of one implementation, and it carries no measurements. The measurements that back it exist and are shared separately.

Terms. A **static-shape program** is what Spyre compiles today: one binary for one exact tensor shape. A **dynamic dimension** is a dimension declared as variable over a range, with a minimum, a maximum and a granularity (the step between supported sizes). A **dynamic-shape program** is one binary compiled over such a range. A **bucket** is a fixed size the plugin rounds a varying quantity up to so that a static-shape program can serve it. A **KV cache page** is a block of tokens of a sequence's history.

## 1. The feature

### 1.1 The declaration

A caller marks one dimension of a host tensor as dynamic when it moves the tensor to the device:

```python
x_dev = x_cpu.to("spyre", dynamic={0: {"min": m, "max": M, "granularity": G}})
```

The device buffer is allocated once at `M` rows with the strides of `M`, the real rows are copied in, and the dimension is marked dynamic for Dynamo. The tensor carries the declaration (`reserved_dims`, readable with `get_reserved_dims`) for its life. Nothing changes in the `torch.compile` call: `dynamic=False` stays, and the compiled function states the range with `torch._check` on the dimension and ties any second dynamic tensor to the first.

### 1.2 What the compiler does with it

Inside a compiled kernel the dynamic size reaches exactly one thing: the iteration count of a `for_each_tile` loop over that dimension. Every address, stride and buffer extent in the program is built at `M`, and the loop's iteration count is set at launch from the tensor's real size. One Inductor compile at the maximum therefore produces one program that serves every size in `[m, M]` that is a multiple of `G`, with no compile for the other sizes. The dynamic dimension must be the loop's tile axis and nothing else: it cannot index a weight, set an output's static extent or decide control flow outside the loop. A reduction along the dimension is fine, as long as it is carried through the loop (an online softmax across pages is the canonical case).

### 1.3 The rules a caller has to respect

- Dimension 0 only, today.
- `M` is a multiple of `G`, the real size at every call is in `[m, M]` and a multiple of `G`.
- `M / G` is at most `MAX_BUCKETS`, 32 by default and an environment variable, because every possible iteration is a slot the host sets at launch. Long contexts therefore take several pages per iteration at the default cap unless the cap is raised, which caps of 128 and 256 have shown to work (section 2.5). A cap set per declaration rather than per process is a torch-spyre item.
- A range of one value is a constant, not a dynamic dimension: one supported size compiles as a static-shape program. Today a range also has to start at two granules; a fix for that is in hand on the torch-spyre side, and until it lands a static companion program covers the one-granule case.
- The declaration is carried by the declaring transfer and by shallow copies of the tensor. A view that changes sizes or strides, or sits at an offset, is a plain tensor again. The output of a compiled graph does not carry its input's declaration on the current pull requests; carrying it across graph outputs is implemented and under test on the torch-spyre side (section 2.2), and is the one item the encoder cases wait on.
- A dynamic dimension on a tensor that is also moved with a custom device layout (`device_layout=`) is refused by the current `.to()`; allowing the two together is a torch-spyre item.

### 1.4 Where the torch-spyre work stands

The compile half (#5186) is being merged as a stack: #5891 merged, #5912 and #5917 open, a fourth part to come. The runtime half, the declaring transfer and the reservation, is #5179. Three more items are in progress or filed on the torch-spyre side: the declaration carried across graph outputs, the layout combination above, and two runtime behaviours found while measuring (a copy of fewer rows than the maximum into a dynamic buffer is far slower than a whole copy, and `resize_` of a dynamic buffer in a custom device layout reads wrong). The plugin can plan around all of them (section 2.6) and none blocks the decoder attention path.

## 2. Where a dynamic dimension pays, and where it does not

### 2.1 A per-step tensor that enters a graph and never leaves it

The simplest and most valuable case. Paged attention takes, every step, a set of host-built tensors whose first dimension is a count that varies per request: the page table and the mask stack of a sequence (one row per KV cache page), and the chunk page index and chunk mask of the batched decode kernel (one row per chunk of pages). Today those counts are rounded up to a bucket, and the plugin compiles one attention program per (query bucket, page bucket) and per (sequence bucket, page bucket), then warms every one of them up. With the page count or the chunk count declared dynamic, the kernel's existing page walk becomes the loop, one program per kernel per query bucket serves every context length, and the programs no longer scale with `max_model_len` or `max_num_seqs`.

What goes away: the KV page bucket list and its presets, the padding of a request to a page bucket, the (sequences, pages) variant grid, the per-variant warm-up loops, and most of the reason for jagged grouping. What a request gains: it pays for its own page count and not for its bucket's, which is what #771, #1060, #942 and #713 ask for in their different ways, and the compile hazards of #1364 shrink to the shapes no range covers. The tensors never cross a graph boundary, so this case needs no propagation of the declaration and works on the current pull requests plus the layout combination.

### 2.2 A batch-like axis that the output keeps

The encoder cases: a vision tower run over any number of fixed-size units (#1523, CLIP images or Granite Vision tiles), an encoder block graph over any batch width at a fixed sequence length (#1525), and the vision budgets more generally (#1473, with #1524 as the interim with fewer budgets). Here the dynamic dimension is on the hidden state itself, `[units * rows_per_unit, width]` or `[B * L, H]`, nothing reduces across units, and every block's output has the same dynamic dimension as its input. Because the plugin compiles one graph per block, the declaration has to survive a graph boundary, which is the question asked on torch-spyre #43 and the propagation item of section 1.3. Until it lands the options are one graph for the whole tower or waiting; once it lands the pattern is the same as 2.1 with the declaration made once at the tower's input. The range workarounds those issues list, declaring the flattened rows with the unit's row count as the granularity and keeping a static graph for a single unit, are the right ones.

### 2.3 What not to make dynamic

The transformer body's token axis. A dynamic dimension becomes a loop, and in the body that loop runs over rows of a matmul against the weights, so every iteration streams the block's weights again; a decode step is a weight pass already, and a prefill chunk would pay one weight pass per tile instead of one. The body wants one flat row count per phase, bucketed, and the lever there is fewer token buckets by default, which is the plugin's own setting and needs nothing from torch-spyre. The same reasoning covers the sequence count of the batched decode kernel: keep it bucketed (it sets how many pages each chunk holds), and let the chunk count be the dynamic axis.

### 2.4 Short ranges and the single-size case

A kernel whose count can be one (a lone short request, a single image) hits the single-size rule: one supported size is a static program, and today a range starts at two granules. Keep a static program for that case beside the dynamic one, chosen by the metadata builder when the real count fits one tile. When the one-granule fix lands the companion goes and the range starts at `G`.

### 2.5 Scenario: a long-context deployment and the iteration cap

A deployment at 32K context has many more pages per sequence than the default cap of 32 iterations allows at one page each, so the plugin has to choose a tile of several pages per iteration. That costs twice: the per-iteration program body unrolls the tile's pages and compiles slower, and a short request walks a whole minimum of two tiles whatever its length.

The guidance: raise the cap (`MAX_BUCKETS`, an environment variable today) so that one page per iteration fits the longest context, and let the plugin's bucketer derive its tile from that same variable rather than carry its own copy of the cap. Caps of 128 and 256 iterations have been exercised with identical tokens and no observed launch cost, so the number of slots is not the constraint it was assumed to be at these sizes; the per-sequence programs compile faster the smaller the tile, and a request walks only its own pages. What a raised cap does not remove is the padding of the per-step mask stack to the maximum before the copy (section 2.6), which a short prompt in a long deployment still pays; the two ways to remove that are granularity bands (a short list of ranges with one program each, so a short request moves a short mask) and a mask built on the device from the sequence lengths, which is the plugin's kernel design to decide. Test at the chosen cap: tokens at the minimum, the maximum and a size just above a former bucket boundary, and the launch cost at the longest context, since the host work per launch does grow with the slot count and larger programs or tensor-parallel deployments may show it.

### 2.6 How to move the per-step tensors

Until the runtime copies fewer rows than the maximum at the speed of a whole copy, pad each per-step tensor to the maximum on the host, move it whole, and `resize_` the device tensor down to the real count, which moves no data and is exact in the default device layout. Do not `resize_` a dynamic buffer that carries a custom device layout; keep the partial copy for that one tensor. Between two compiled graphs, pass outputs as they are or as shallow copies; a reshape that changes strides drops the declaration.

## 3. What would change in spyre-inference

Guidance per area, the same for any kernel that adopts the feature.

- **Bucketer.** A declared range per dynamic axis in place of a bucket list: the minimum, the maximum derived from `max_model_len` (and `max_num_seqs` for the batched kernel) and the granularity derived from the cap. The query buckets and the body's row counts stay. One compile point per kernel per range instead of a grid walk.
- **Metadata builder.** Real counts floored at the range's minimum instead of rounded to a bucket; the per-step tensors built at their real size and moved with the declaration (section 2.6); no host-side slicing of a tensor after it has been declared, since a slice is a view.
- **Kernels.** The operand slice to the bucket (`[:num_blocks]`) replaced by `torch._check` statements on the declared range, and the page body taking a tile of `G` pages per iteration. The kernels already walk pages with `for_each_tile`, so this is a small change in each.
- **Warm-up.** One compile per kernel per range instead of one per variant; the recorder stays, with far fewer entries. A dynamic program that fails to compile at warm-up should fail warm-up rather than be deferred to serving, because the compile guard at `error` would otherwise stop the engine on the first request that needs it.
- **Compile guard.** No change: when the ranges cover every size the engine can see, the guard sees no compile at serving time.
- **Compile cache.** Fewer and larger entries; the populated-cache start shrinks with the program count. Behaviour across tensor-parallel ranks is to be checked on a multi-card host.
- **Tests.** Every kernel that adopts a range is run at several sizes across the range, including the minimum, the maximum and a size just above a former bucket boundary, against the static program or CPU, with tokens compared, before any size is served. The plugin's attention device tests run with the switches on.
- **The V2 runner.** The same recorder and the same seams (#1242 for startup, #1239 for step inputs); nothing in the pattern is specific to V1, and the extract-first rule applies to the shared helpers.
- **Rollout.** Behind per-kernel switches, off by default: the per-sequence kernels' page axis, the batched kernel's chunk axis, the padded transfer, the single-size companion. The bucket path stays until the switches have run the CI configurations, then the buckets go.

## 4. The order of work, and what can start in parallel

On the torch-spyre side: merging the stack (#5912, #5917, the fourth part) and #5179 so the plugin's pin can carry the mechanism; then the layout combination, the declaration across graph outputs, the one-granule minimum, the runtime copy behaviours and the cap. Of these, the decoder attention path needs only the merged stack, #5179 and the layout combination; the encoder path needs the propagation as well.

On the spyre-inference side, in parallel: the decoder attention path of section 2.1 behind switches, with its tests at every size, is the first thing to build and the one that answers #1515 directly; the body's default token buckets (section 2.3) can be revisited independently of any of this; the encoder work of #1523 and #1525 can be designed now and declared once the propagation lands, with #1524 as the interim. Owners and dates are for the epic to assign.

## 5. Risks and limits

- Correctness at every size in a range is a property to test, not to assume, on every kernel that adopts the feature.
- The iteration cap constrains long contexts until it moves; the plugin should read the cap from the same environment variable torch-spyre uses rather than carry its own copy.
- The body is unchanged, so short-context decode and time to first token for short prompts do not change with this work.
- Tensor-parallel sizes above one have not been exercised with a dynamic dimension.
- The encoder and vision paths wait on the propagation item.

## Appendix: the seven questions of #1515

1. **Mechanism.** `tensor.to("spyre", dynamic={dim: {"min", "max", "granularity"}})` on the host tensor, nothing in the `torch.compile` call, `torch._check` statements in the kernel. Dimension 0 only, `max / granularity` at most the cap, the dimension must be a loop's tile axis.
2. **One compile for many buckets.** Yes: one Inductor compile at the maximum per kernel per range; the other sizes are not compiled at all.
3. **Correctness.** A single compile serves every size in the range with the same results as the static programs when the kernel reduces along the dimension through the loop's carry; it is tested at every size before it is served.
4. **Run-time cost.** None in the launch for a dynamic-shape program against a static one at the same page count. The cost that exists today is in the per-step copy of fewer rows than the maximum, which the padded transfer avoids. The gain is fewer pages walked at long context and full batch; short-context decode is a weight pass and does not change.
5. **Changes in spyre-inference.** Section 3: the bucketer, the metadata builder, the kernels' operand checks and tile walk, nothing in the `torch.compile` calls, the recorder and the guard unchanged, behind switches. The body's token buckets keep their own compiles.
6. **Compile cache.** The dynamic-shape programs hit the kernel cache across restarts like any other; fewer programs, a faster populated-cache start. Across TP ranks, to be checked.
7. **Gaps.** The iteration cap at long context, `dynamic=` with `device_layout=`, the slow partial copy, `resize_` under a custom layout, the two-granule minimum, the declaration across graph outputs, and the body's token axis, which is not a loop's business.
