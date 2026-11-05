# RFC: Platform simplification through symbolic shapes, for compile time and runtime

| Field | Value |
|---|---|
| Status | Draft, for discussion under the boot-time epic |
| Created | 2026-11-05 |
| Tracking | [#1432 Reduce spyre-inference boot/compile time](https://github.com/torch-spyre/spyre-inference/issues/1432), [#1515 Investigate how torch-spyre symbolic shape can reduce compile time](https://github.com/torch-spyre/spyre-inference/issues/1515) |
| Related | #959, #566 (boot time as a tracked metric), #771 (denser block ladder), #942 (performance above 32 sequences), #1364 (compiles after warm-up), #1473 (vision encoders once dynamic shapes exist), #713 (long context); torch-spyre #5891, #5912, #5917, #5179, #5186 |

## 1. Motivation

Every graph on Spyre is compiled for one exact shape. The plugin handles a varying size by rounding it to a bucket and compiling one program per bucket, then warming every bucket up at start. For paged attention that is one program per (query bucket, page bucket) for the per-sequence kernels and one per (sequence bucket, page bucket) for the batched decode kernel. The counts grow with the context length and with `max_num_seqs`, and so does the boot time:

| granite-3.3-8b, one card | attention programs at warm-up | body programs | cold boot |
|---|---|---|---|
| 2048 context, 4 sequences | 25 | 4 | 367 s |
| 4096 context, 32 sequences (epic row 1) | 48 | 7 | 961 s |
| 32768 context, 4 sequences | 45 | 4 | 751 s |

The epic's target is a boot under 8 minutes on row 1 and on six more rows at 32K with 32 sequences. The same bucketing also costs at run time (a request pays for the bucket, not for its own length, which is #771's motivation and was #1060's), and it leaves compile hazards after warm-up whenever a shape escapes the ladder (#1364).

torch-spyre is landing symbolic shapes: a tensor dimension declared with a range and a granularity, served by one compiled program whose device loop reads the real size at launch. This RFC proposes to use that to remove the attention ladders from the plugin, states what it measured, what it does not solve, and what it asks of torch-spyre. It answers the seven questions of #1515 in the appendix.

## 2. Goals and non-goals

Goals: one attention program per kernel per query bucket, independent of the context length and of `max_num_seqs`. The attention buckets, their presets, the variant grids and the per-variant recording removed from the warm-up. Boot time on row 1 under the target with no cache. Output tokens identical to today's. No regression at short context, a gain at long context.

Non-goals: the transformer body's token buckets (a decode step is a weight pass, so the body wants one flat row count per phase and a loop does not help it), tensor parallel sizes above 1 (not yet measured), the vision towers (#1473, a later step on the same mechanism), jagged attention, prefix caching changes.

## 3. The mechanism torch-spyre provides

A caller moves a tensor with a declaration, `tensor.to("spyre", dynamic={0: {"min": m, "max": M, "granularity": G}})`. The device buffer is allocated once at `M` rows with the strides of `M`, the real rows are copied in, and dim 0 is marked dynamic for Dynamo (torch-spyre #5179). Inside a compiled kernel the declared size may reach exactly one thing: the trip count of a `for_each_tile` loop over that dimension. Every address, stride and buffer extent is built at `M`, and the loop count is corrected at launch from the tensor's size, so one program serves every size in `[m, M]` that is a multiple of `G` (torch-spyre #5186, split into #5891 merged, #5912 approved, #5917 and a fourth part open). The compile is one Inductor compile at the maximum, not one per size.

Limits as of the build measured (torch-spyre `43c4150a`, main plus #5179 plus #5186 plus one cross-kernel commit): dim 0 only, `m` at least `2G` (a single tile is its own static program), `M / G` at most 32 trips, the dimension must be a loop's tile axis and nothing else, and `resize_` of a reserved tensor is exact in the default device layout only.

## 4. Proposal

### 4.1 What becomes declared

| kernel | declared axis | one program per | pages per trip |
|---|---|---|---|
| per-sequence prefill and decode (head-major) | the page count of `page_index_table` and `mask_stack` | query bucket | `ceil(max_pages / 32)`: 1 up to 4096 tokens, 8 at 32K |
| batched decode (head-major) | the chunk count of `chunk_page_ids` and `mask_by_chunk` | sequence bucket | pages per chunk pinned at `32 // num_seqs` |

Everything else in the kernels keeps a fixed extent, so the declared size reaches only the walk's trip count. The kernels already walk pages with `for_each_tile`, so the kernel change is the operand cut (`[:num_blocks]`) replaced by four `torch._check` statements on the declared range, and the page body taking a tile of `G` rows.

### 4.2 What leaves the plugin

The KV bucket ladder and its presets (`SPYRE_ATTN_KV_LADDER`, `SPYRE_ATTN_KV_BUCKETS`), the padding of a request to a page bucket, the sequence by page variant grid of the batched kernel, the per-variant recording loops over those grids, and the reason for the jagged grouping (it exists to avoid padding every sequence to the longest bucket, which exact chunk counts make small). The bucketer keeps the query buckets and the body row counts. `record_attention_graphs` and the compile guard stay, with far fewer things to record and watch. The V2 runner uses the same recorder, so nothing in this proposal is specific to V1.

### 4.3 How the host moves the per-step tensors

A reserved copy below the maximum is slow on the current runtime (ten times a whole copy, section 6). The plugin therefore pads each per-step tensor to the declared maximum on the host, moves it whole, and resizes the device tensor down to the real count, which moves no data. Measured exact and 2.8 ms for a 512-query mask stack where the partial copy took 19 ms. The batched kernel's chunk index keeps the partial copy, because a `resize_` of a tensor in its custom chunk-major layout reads wrong today (section 6), and it is small.

### 4.4 A static program for one chunk

The batched kernel's declared program needs at least two chunks. A context that fits one chunk (every request under 4096 tokens at one sequence, say) runs a static one-chunk program recorded beside the declared one, one more program per sequence bucket, which removes the two-chunk floor at one sequence.

### 4.5 Rollout

Behind switches, off by default, in one pull request against the plugin once torch-spyre's pins carry the stack and #5179: `SPYRE_ATTN_SYMBOLIC_PAGES`, `SPYRE_ATTN_SYMBOLIC_BATCHED`, `SPYRE_ATTN_SYMBOLIC_PAD_TRANSFER`, `SPYRE_ATTN_SYMBOLIC_ONE_CHUNK`. The plugin's own attention device tests pass with them on. The ladder path stays until the switches have run the CI configurations, then the ladders go.

## 5. Evidence

All on granite-3.3-8b-instruct, one card, one build, the shipped plugin against the same plugin with the switches, cold caches per configuration, output tokens compared before any timing. Records: `spyre_inference_symbolic_matrix_{2k,4096,32k}_2026-11-04.md` and `spyre_inference_symbolic_fixes_2026-11-04.md`.

| configuration | attention programs | cold boot | correctness |
|---|---|---|---|
| 2048, 4 seqs | 25 to 5 | 367 s to 200 s | tokens identical on 4 prompts, batch and single |
| 4096, 32 seqs (epic row 1) | 48 to 8 | 961 s to 370 s, 233 s with one body bucket | identical on 8 prompts, batch and single |
| 32768, 4 seqs | 45 to 5 | 751 s to 335 s | identical on 4 prompts to 17K tokens |

Run time, same build and same day, the declared plugin with the padded transfer against the shipped one:

- 4096 with 32 sequences: a decode step at 32 sequences 298 ms shipped against 258 ms (the morning's shipped figure against the evening's declared figure, two engines; in one run the padded transfer alone took the declared step from 271 to 258 ms). Generation at 8 sequences 16.9 s against 16.6 s. At 1 sequence 9.5 s against 9.95 s, and 9.6 s with the one-chunk program on its own. Prefill of 512 tokens 0.232 s against 0.238 s.
- 32768 with 4 sequences: prefill 10 to 18 percent faster on 9K to 31K prompts, decode 7 to 9 percent faster a step between the ladder's buckets, flat at a bucket boundary.
- Kernel level: the declared program and the static program at the same page count launch in the same time within 1 percent, decode and prefill. The loop costs nothing against the unrolled program.
- Warm start with a populated cache at 2048: 14.5 to 16 s against 25 to 27 s, fewer programs to load.
- Serving under load at 4096 with 32 sequences, `vllm bench serve`, 200 random prompts of 1024 tokens plus or minus 80 percent, 128 output tokens, concurrency 32, both legs in one run: server boot 1176 s shipped against 379 s declared (48 attention programs against 8). Throughput and latency are a wash under this load: 84.3 against 83.5 output tokens a second, median TTFT 2.34 against 2.32 s, median ITL 345 against 355 ms, every p99 within 2 percent either way, all inside the run-to-run band of the shipped plugin itself. Without the padded whole transfer (section 6) the declared axes run 4 to 11 percent behind under the same load (80.4 tokens a second, median ITL 372 ms, a third run the same night), so that transfer is part of the design, not an option.

The run-to-run drift of one configuration on one build is 2 to 6 percent between engine starts, so only same-run differences are quoted above as gains, and the cross-engine ones are labelled.

## 6. What it does not solve, and what it costs

- Short-context decode is a weight pass, 137 ms a step for 16 GB of fp16 weights at one sequence, so attention changes do not move it. Time to first token for short prompts likewise.
- The 32-trip cap forces 8 pages per trip at 32K. The two per-sequence programs then unroll 8 pages in their body and cost about 80 s each to compile, which is why the 32K boot is 335 s and not lower, and a short request in a 32K deployment pads to 16 pages. A cap set per declaration, or granularity bands (a list of ranges, one program each), removes this.
- A reserved copy below the declared maximum is about ten times slower than a whole one on the current runtime (19 ms against 2 ms for a 2 MB mask stack). The padded whole transfer works around it in the plugin, the fix belongs in the runtime.
- `resize_` on a reserved tensor with a caller-supplied device layout reads wrong, silently. The plugin avoids it, the fix belongs in the runtime.
- The one-chunk program's gain at one sequence does not reproduce when the padded transfer is also on, three runs out of three. Open.
- Tensor parallel sizes above 1, the fp8 KV cache, sliding-window models and the official-list models beyond granite-3.3 were being measured on the night of 5 November and are reported in the records as they land, failures included.

## 7. Asks of torch-spyre

1. Merge the stack (#5912, #5917, the fourth part) and #5179, so the plugin's pin can carry the mechanism.
2. `dynamic=` together with `device_layout=`, needed by the batched kernel's chunk index (a Python-only relaxation exists as a draft).
3. The reservation's floor and cap as arguments of the declaration, not process-wide environment variables, and the cap measured rather than assumed.
4. Granularity bands, so a long deployment keeps one page per trip for short requests.
5. The partial reserved copy brought to the speed of a whole copy, and `resize_` made exact for a reserved tensor in a custom device layout.

## 8. Alternatives considered

A denser ladder (#771) adds programs and boot time where this removes them. A populated compile cache (#959) helps restarts, not the first boot of a new configuration, and the declared programs make the cache smaller too. Emitting static loops instead of unrolled programs was tested and found equal, so there is no codegen change to make.

## Appendix: the seven questions of #1515

1. **Mechanism.** `tensor.to("spyre", dynamic={dim: {"min", "max", "granularity"}})` on the host tensor, nothing in the `torch.compile` call (`dynamic=False` stays), four `torch._check` statements in the kernel stating the range and tying the two declared tensors. Dim 0 only, `min` at least twice the granularity, `max / granularity` at most 32, the dimension must be a loop's tile axis.
2. **One compile for many buckets.** Yes. One Inductor compile at the maximum per kernel per query bucket, the other buckets are not compiled at all. 25 attention programs to 5 at 2K, 48 to 8 at 4096 with 32 sequences, 45 to 5 at 32K.
3. **Correctness.** Output tokens identical to the shipped plugin at every size tried, 2K to 32K, batch and single, greedy with logprobs (largest logprob gap 0.005). The kernels are bit-exact against the static ones on the device probe, at 1, 2 and 8 pages per trip.
4. **Run-time cost.** None in the launch: the declared and the static program at the same page count launch in the same time within 1 percent. The cost was in the per-step reserved transfer below the maximum, removed by the padded whole transfer. The gain is fewer pages walked: 7 to 18 percent at long context, a 13 percent decode step at 4096 with 32 sequences across engines, a wash at short context.
5. **Changes in spyre-inference.** The bucketer (the page range, the chunking, one recording point per kernel), the metadata builder (the real count floored at the minimum instead of a bucket, two declared transfers, no host slice of a declared tensor), the head-major backend (the declared chunk index and mask), the three kernels (the operand checks and the tile walk), nothing in the `torch.compile` calls, the recorder and the guard unchanged, about 960 patch lines behind switches. The body buckets keep their own compiles.
6. **Compile cache.** The declared programs hit the kernel cache across restarts (warm start 14.5 to 16 s against 25 to 27 s at 2K). Across TP ranks not measured, one card.
7. **Gaps.** The 32-trip cap at long context, `dynamic=` with `device_layout=`, the slow partial reserved copy, the wrong `resize_` under a custom layout, the one-value range that cannot be declared (handled by a static program), and the body's token axis, which is not a loop's business.
