# RFC: Using torch-spyre's dynamic dimensions in spyre-inference

| Field | Value |
|---|---|
| Status | Draft, for discussion under the boot-time epic |
| Created | 2026-11-05 |
| Tracking | [#1432 Reduce spyre-inference boot/compile time](https://github.com/torch-spyre/spyre-inference/issues/1432), [#1515 Investigate how torch-spyre symbolic shape can reduce compile time](https://github.com/torch-spyre/spyre-inference/issues/1515) |
| Related | #1523, #1525, #1524, #1473 (encoders and vision towers), #771, #1060 (a request pays for its bucket), #1364 (compiles after warm-up), #942 (performance above 32 sequences), #713 (long context), #468 (jagged attention), #959, #566 (boot time as a metric), #1242, #1239 (the V2 runner's startup and step inputs); torch-spyre #43 (the epic), #5891, #5912, #5917, #5179, #5186 |

What this RFC is. torch-spyre is adding dynamic shapes support ( https://github.com/torch-spyre/torch-spyre/issues/43 ) : a tensor dimension that varies over a declared range and is served by one compiled program instead of one program per size. This document describes the feature and how spyre-inference can use it, and in what order, so that any part of the plugin can adopt it the same way.

Terms:
- **static-shape program** is what Spyre compiles today: one binary for one exact tensor shape.
- **dynamic dimension** is a dimension declared as variable over a range, with a minimum, a maximum and a granularity (the step between supported sizes).
- **dynamic-shape program** is one binary compiled over such a range.
- **bucket** is a fixed size the plugin rounds a varying quantity up to so that a static-shape program can serve it.
- **KV cache page** is a block of tokens of a sequence's history.

## 1. The feature

### 1.1 The declaration

A caller marks one dimension of a host tensor as dynamic when it moves the tensor to the device:

```python
x_dev = x_cpu.to("spyre", dynamic={0: {"min": m, "max": M, "granularity": G}})
```

The device buffer is allocated once at `M` rows with the strides of `M`, the real rows are copied in, and the dimension is marked dynamic for Dynamo. The tensor carries the declaration (`reserved_dims`, readable with `get_reserved_dims`) for its life. Nothing changes in the `torch.compile` call: `dynamic=False` stays ( dynamic = `false` is later overwritten by `mark_dynamic` as per the PyTorch precedence rule ), and the compiled function states the range with `torch._check` on the dimension and ties any second dynamic tensor to the first.

### 1.2 What the compiler does with it

Inside a compiled kernel, the dynamic size reaches exactly one thing: the iteration count of a `for_each_tile` loop over that dimension. Every address, stride and buffer extent in the program is built at `M`, and the loop's iteration count is set at launch from the tensor's real size. One Inductor compile at the maximum therefore produces one program that serves every size in `[m, M]` that is a multiple of `G`, with no compile for the other sizes. The dynamic dimension must be the loop's tile axis and nothing else: it cannot index a weight, set an output's static extent or decide control flow outside the loop. A reduction along the dimension is fine, as long as it is carried through the loop (an online softmax across pages is the canonical case).

Launching a dynamic-shape program costs the same as launching a static-shape program at that size: the host sets the iteration count, and nothing else changes at dispatch. 

### 1.3 The rules a caller has to respect

- Dimension 0 only, today.
- `M` is a multiple of `G`, the real size at every call is in `[m, M]` and a multiple of `G`.
- `M / G` is at most `MAX_BUCKETS`, 32 by default and an environment variable ( higher caps of 128 and 256 have been shown to work (section 2.5) ). A cap set per declaration rather than per process is a torch-spyre item.
- A range of one value is a constant, not a dynamic dimension: one supported size compiles as a static-shape program. ( Related bug:  https://github.com/torch-spyre/torch-spyre/issues/6018 ,  Status: Resolved, PR to be raised )
- The declaration stays with the tensor through `detach()` and other shallow copies. A view that changes sizes or strides, or sits at an offset, is a plain tensor again. `clone()` of a declared tensor fails today ( Known issue: https://github.com/torch-spyre/torch-spyre/issues/6030 , use `detach()` for a copy that keeps the declaration ).
- The output of a compiled graph does not carry its input's declaration on the current pull requests, so a second compiled graph fed that output sees a plain tensor. Carrying the contract across graph outputs is in the development plan and is implemented, with its pull request to be raised ( https://github.com/torch-spyre/torch-spyre/issues/6029 ). Until it lands, a dynamic dimension is usable on tensors that enter a graph and are consumed inside it (section 2.1), and not yet on a hidden state that flows from one block's graph into the next (section 2.2).
- A dynamic dimension on a tensor that is also moved with a custom device layout (`device_layout=`) is refused by the current `.to()`. Accepting the two together is in progress ( https://github.com/torch-spyre/torch-spyre/issues/6027 ); the batched decode kernel's chunk index needs it (section 2.1).

### 1.4 Known issues and in-progress development

| item | where | status |
|---|---|---|
| the compile half: symbolic loop iteration counts through Inductor to the device program | torch-spyre #5186, split into #5891 (merged), #5912, #5917 and a fourth part | merging as a stack |
| the runtime half: the declaring transfer, the buffer reserved at the maximum, the contract on the tensor | torch-spyre #5179 | approved, merging |
| the contract carried across compiled graph outputs | torch-spyre #6029 | implemented, pull request to be raised and in queue |
| `dynamic=` together with `device_layout=` | torch-spyre #6027 | in progress; the same issue records that `resize_` of a dynamic buffer in a custom device layout reads wrong, so until it is fixed such a buffer is moved at its real size and never resized |
| copying fewer rows than the maximum into a dynamic buffer is several times slower than copying the whole buffer | torch-spyre #6026 | open; the workaround in section 2.6 covers it |
| a range cannot start at one granule | torch-spyre #6018 | resolved, pull request to be raised and in queue |
| `clone()` of a declared tensor raises | torch-spyre #6030 | resolved, pull request to be raised and in queue, workaround use :  `detach()` instead |
| the iteration cap as a per-declaration setting | torch-spyre #43 | planned |

## 2. Cases

### 2.1 A kernel that walks a per-request number of pages

Paged attention takes, at every step, a few host-built tensors whose first dimension is a count that varies with the request: a sequence's page table and mask stack have one row per KV cache page, and the batched decode kernel's chunk page index and chunk mask have one row per chunk of pages. The kernels walk those rows one tile at a time already. Today the count is rounded up to a bucket, so there is one attention program per (query bucket, page bucket) for the per-sequence kernels and one per (sequence bucket, page bucket) for the batched kernel, all compiled at start, and a request walks its bucket's pages and not its own.

Declare the count instead. The page count of the table and the mask stack becomes the dynamic dimension of the per-sequence kernels, the chunk count of the index and the chunk mask becomes the dynamic dimension of the batched kernel, and the existing walk is the loop. One program per kernel per query bucket serves every context length, and the count of programs stops growing with `max_model_len` and with `max_num_seqs`. With that, the KV page bucket list and its presets, the padding of a request to a page bucket, the (sequences, pages) variant grid and the per-variant warm-up loops have no job left, and most of the reason for jagged grouping (#468) goes with them, since a request pays for its own page count. The compile hazards after warm-up (#1364) shrink to the shapes no range covers. These tensors enter a graph and are consumed inside it, so this case needs nothing beyond the declaration, the merged pull requests and #6027 for the chunk index.

### 2.2 A tower that runs over a variable number of equal units

A vision tower over any number of fixed-size images or tiles (#1523), an encoder block graph over any batch width at a fixed sequence length (#1525), and the vision budgets in general (#1473, with #1524 as the interim with fewer budgets) are one shape of problem: the hidden state is `[units * rows_per_unit, width]` or `[B * L, H]`, nothing reduces across the units, and every block's output has the same dynamic dimension as its input. The declaration is made once, at the tower's input, with the unit's row count as the granularity; the first block's graph gets it, and the point of #6029 is that every block after it gets it too, through the previous block's output. Until that pull request lands, the tower is either one compiled graph or stays on budgets. The two range workarounds those issues already note are the right ones: declare the flattened rows with the unit's row count as the granularity, and keep a static graph for a single unit until #6018 lands.

### 2.3 The transformer body and the batch size

The body's token axis is not a case for a dynamic dimension, and it is worth saying why once. A dynamic dimension becomes a loop, and in the body that loop runs over rows of a matmul against the weights, so every iteration streams the block's weights again. A decode step is a weight pass already, and a prefill chunk would pay one weight pass per tile instead of one. The body wants one flat row count per phase, bucketed, and the lever there is fewer token buckets by default, which is the plugin's own setting and needs nothing from torch-spyre. The same reasoning keeps the sequence count of the batched decode kernel bucketed: it sets how many pages each chunk holds, and the chunk count is the axis that varies.

### 2.4 A count that can be one

A lone short request, a single image: the count can be one, and a range of one value is a static program. Keep a static program for that case beside the dynamic one and let the metadata builder pick it when the real count fits one tile. Once #6018's pull request lands the range starts at `G` and the companion goes.

### 2.5 Long contexts and the iteration cap

A 32K deployment has many more pages per sequence than the default cap of 32 iterations allows at one page each, so the tile has to be several pages per iteration. That costs twice: the per-iteration program body unrolls the tile's pages and compiles slower, and a short request walks a whole minimum of two tiles whatever its length. Raise the cap (`MAX_BUCKETS`) so that one page per iteration fits the longest context, and let the bucketer derive its tile from that same variable instead of carrying its own copy of the cap. Caps of 128 and 256 iterations have been exercised with identical tokens and no observed launch cost, so the slot count is not the constraint it was assumed to be at these sizes; the per-sequence programs compile faster the smaller the tile, and a request walks only its own pages. What a raised cap does not remove is the padding of the per-step mask stack to the maximum before the copy (section 2.6), which a short prompt in a long deployment still pays; granularity bands (a short list of ranges, one program each) or a mask built on the device from the sequence lengths would remove that, and the second is the plugin's kernel design to decide. At the chosen cap, test tokens at the minimum, the maximum and a size just above a former bucket boundary, and the launch cost at the longest context, since the host work per launch does grow with the slot count and larger programs or tensor-parallel deployments may show it.

### 2.6 Moving the per-step tensors

Until #6026 is fixed, pad each per-step tensor to the maximum on the host, move it whole, and `resize_` the device tensor down to the real count; the `resize_` moves no data and is exact in the default device layout. Do not `resize_` a dynamic buffer that carries a custom device layout (#6027 records why); move that one at its real size. Between two compiled graphs, pass outputs as they are or through `detach()`, not `clone()` (#6030); a reshape that changes strides drops the declaration.

## 3. What would change in spyre-inference

The same handful of changes for any kernel that adopts the feature. The snippets show the shape of each change, not the plugin's code.

**The bucketer.** A declared range per dynamic axis in place of a bucket list, the maximum from the deployment's limits and the granularity from the cap. The query buckets and the body's row counts stay.

```python
max_pages = ceil(max_model_len / block_size)
pages_per_iteration = max(1, ceil(max_pages / max_buckets))      # 1 page up to 4096 context at the default cap
page_range = {0: {"min": 2 * pages_per_iteration, "max": max_pages, "granularity": pages_per_iteration}}

pages_per_chunk = max(1, cores // num_seqs)                       # the batched kernel keeps its pinning
chunk_range = {0: {"min": 2, "max": ceil(max_pages / pages_per_chunk), "granularity": 1}}
```

**The metadata builder.** Real counts floored at the range's minimum instead of rounded to a bucket, the per-step tensors built at their real size and moved with the declaration (padded to the maximum for now, section 2.6), and no host-side slicing of a tensor after it has been declared, since a slice is a view.

```python
num_pages = max(ceil(seq_len / block_size), page_range[0]["min"])
table = page_index_table_cpu[:num_pages]                          # [num_pages, ...], the real count

padded = torch.zeros((max_pages, *table.shape[1:]), dtype=table.dtype)
padded[:num_pages] = table
table_dev = padded.to("spyre", dynamic=page_range)
table_dev.resize_(table.shape)                                    # no data moved; default layout only
```

**The kernels.** The operand slice to the bucket becomes a range statement on the declared dimension, the two dynamic tensors are tied, and the page body takes a tile of `G` pages per iteration. The kernels walk pages with `for_each_tile` already, so this is a small change in each.

```python
def prefill_attention(query, k_pages, v_pages, page_table, mask_stack):
    n = page_table.shape[0]
    torch._check(n >= page_range[0]["min"])
    torch._check(n <= page_range[0]["max"])
    torch._check(n % pages_per_iteration == 0)
    torch._check(mask_stack.shape[0] == n)                        # tie the two dynamic tensors

    def body(carry, tiles):
        page_ids, mask = tiles                                    # one tile of pages_per_iteration pages
        ...                                                       # gather the pages, online softmax, update the carry
        return carry, out_tile

    carry, out = for_each_tile(body, (page_table, mask_stack), dims=(0, 0),
                               tile_size=pages_per_iteration, out_dim=0)
    return normalize(carry)
```

**The single-size companion.** Until #6018 lands, the builder picks a static one-tile program when the real count fits one tile, and the dynamic program otherwise.

```python
if num_chunks < chunk_range[0]["min"]:       # the step fits one chunk
    run(static_one_chunk_program, ...)
else:
    run(dynamic_program, chunk_index_dev, chunk_mask_dev, ...)
```

**Warm-up.** One compile per kernel per query bucket, at the maximum, instead of one per variant; the recorder stays, with far fewer entries. A dynamic program that fails to compile at warm-up should fail warm-up rather than be deferred to serving, because the compile guard at `error` would otherwise stop the engine on the first request that needs it.

```python
for q in query_buckets:
    record(prefill_attention, query_rows=q, pages=max_pages)      # once per query bucket, not per page bucket
for b in sequence_buckets:
    record(batched_decode, seqs=b, chunks=chunk_range[0]["max"])  # once per sequence bucket
```

**The compile guard.** No change: when the ranges cover every size the engine can see, the guard sees no compile at serving time.

**The compile cache.** Fewer and larger entries, so the populated-cache start shrinks with the program count. Behaviour across tensor-parallel ranks is to be checked on a multi-card host.

**Tests.** Every kernel that adopts a range runs at several sizes across it, including the minimum, the maximum and a size just above a former bucket boundary, against the static program or CPU, with tokens compared, before any size is served. The plugin's attention device tests run with the switches on.

```python
for n in range(page_range[0]["min"], max_pages + 1, pages_per_iteration):
    torch.testing.assert_close(dynamic_kernel(n), static_kernel(n))
```

**The V2 runner.** The same recorder and the same seams (#1242 for startup, #1239 for step inputs); nothing here is specific to V1, and the extract-first rule applies to the shared helpers.

**Rollout.** Behind per-kernel switches, off by default: the per-sequence kernels' page axis, the batched kernel's chunk axis, the padded transfer, the single-size companion. The bucket path stays until the switches have run the CI configurations, then the buckets go.

## 4. The order of work

Compile time comes first, which here means fewer programs through fewer buckets, and throughput after it. The decoder attention path of section 2.1 is the first thing to build: it needs the merged pull requests, #5179 and #6027, nothing else, and behind switches with its tests at every size it can be built now against the combined branch. The body's default token buckets (section 2.3) can be revisited independently of any of this. The encoder work of #1523 and #1525 can be designed now and declared once #6029 lands, with #1524 as the interim. Owners and dates are for the epic to assign.

## 5. Limits

- Correctness at every size in a range is a property to test, not to assume, on every kernel that adopts the feature.
- The iteration cap constrains long contexts until it is a per-declaration setting; the plugin should read the cap from the same environment variable torch-spyre uses rather than carry its own copy.
- The body is unchanged, so short-context decode and time to first token for short prompts do not change with this work.
- Tensor-parallel sizes above one have not been exercised with a dynamic dimension.
- The encoder and vision paths wait on #6029.
