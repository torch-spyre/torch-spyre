# Symbolic shapes in spyre-inference: the build, the integration, the test setup and the results


## 1. The idea in one paragraph

spyre-inference compiles one static-shape program per bucket of every varying size, and for paged attention that means one program per (query bucket, page-count bucket) for the per-sequence kernels and one per (sequence-count bucket, page-count bucket) for the batched decode kernel, all compiled at start. torch-spyre's dynamic dimension lets one dimension of a tensor vary over a declared range, served by one program whose device loop runs for the real size at launch. The integration declares the two counts that paged attention varies on, the page count a sequence's kernel walks and the chunk count the batched kernel walks, so that one program per kernel per query bucket serves every context length and every sequence count. The transformer body is left bucketed on purpose: a decode step is a weight pass and a loop over its rows would stream the weights once per iteration.

## 2. The base build

| component | version | note |
|---|---|---|
| torch-spyre | `43c4150a`, branch `combined_5179_5186_cross_kernel` | main as of 2 November plus #5179 (the runtime half: the declaring transfer, the buffer reserved at the maximum, the contract on the tensor) plus #5186 (the compile half: symbolic loop iteration counts through Inductor to the device program, being merged as #5891, #5912, #5917 and a fourth part) plus one cross-kernel commit (the contract carried across compiled graph outputs, with tests) |
| torch-spyre, two local relaxations | | `.to()` accepting `dynamic=` together with `device_layout=` (the check of #6027 bypassed, needed by the batched kernel's chunk index), and `MIN_RESERVED_DIM_SIZE=2` in the environment (the default floor of 8 refuses a two-page minimum) |
| torch-spyre wheel | `0.1.0.dev2225+combined51795186cros.g43c4150a.dirty` | the same wheel installed on both pods, md5 `9d5781b409b0` |
| PyTorch | 2.13.0+cpu | main moved to 2.14 on 3 November, the build stayed on 2.13 |
| vLLM | 0.32.0 | |
| spyre-inference | `1881aa45fa` plus the attention patch | the patch is 987 lines across 8 files, behind switches, off by default |
| Spyre stack | DeepTools `0dabe5e1`, Flex `e849de63`, senlib `d381a839` | identical on both pods |
| hardware | one Spyre card per pod, two dev pods on the same shared volume | results from the two pods agree within 1 percent on the shipped plugin's benches (checked three times) |

The plugin's own defaults were kept throughout: block size 128 tokens, the head-major KV layout, batched decode on, prefix caching off, compile sizes 1, 2, 4, 8, 16, 32 and 512 for the body (64 added for the 64-sequence run), the KV bucket ladder's default power-of-two preset for the shipped legs.

## 3. What was changed in spyre-inference

Eight files, 987 patch lines, every change behind a switch that is off by default so the shipped path is untouched.

**Switches (`envs.py`).** `SPYRE_ATTN_SYMBOLIC_PAGES` declares the page axis of the per-sequence prefill and decode kernels. `SPYRE_ATTN_SYMBOLIC_BATCHED` declares the chunk axis of the batched decode kernel. `SPYRE_ATTN_SYMBOLIC_PAGES_TILE` overrides the pages per loop iteration (otherwise derived from the cap). `SPYRE_ATTN_SYMBOLIC_ONE_CHUNK` records a static single-chunk program beside the dynamic one for the case a step fits one chunk. `SPYRE_ATTN_SYMBOLIC_PAD_TRANSFER` pads each per-step tensor to the maximum on the host, moves it whole and resizes it down. `SPYRE_ATTN_SYMBOLIC_PERSISTENT` reserves the buffers once and refills them in place (refused by the result, kept for the record). `SPYRE_ATTN_DEBUG_DISPATCH` logs which batched program a step dispatches, a diagnostic.

**The bucketer (`spyre_attn_bucketer.py`).** A declared page range per deployment instead of the KV bucket ladder: the maximum is the page count of `max_model_len`, the pages per iteration are `ceil(max_pages / MAX_BUCKETS)` (1 page up to 4096 context at the default cap of 32, 8 pages at 32K), and the minimum is two iterations. For the batched kernel the pages per chunk stay pinned at `32 // num_seqs` (so every chunk fills the cores) and the chunk count becomes the declared range, from 2 to the chunks the longest context takes; the sequence buckets stay. `variants()` returns one dynamic variant per query bucket and one per sequence bucket instead of the (query, pages) and (sequences, pages) grids. The bucketer's own cap reads torch-spyre's `MAX_BUCKETS` so the two never disagree.

**The metadata builder and the backend (`spyre_attn.py`, `spyre_head_major_attn.py`).** The real page count floored at the range's minimum instead of rounded to a bucket; the per-sequence page tables and mask stacks moved with `.to("spyre", dynamic=page_range)`; the batched kernel's chunk index moved with its chunk-major device layout and the declaration together, and its chunk mask with the declaration; `_chunk_range` deciding per step whether the dynamic program or the static single-chunk program runs; `_declared_whole` implementing the padded transfer; the batched query staged into a buffer of `max_num_batched_tokens` rows. No host slicing of a declared tensor anywhere.

**The kernels (`page_attn_head_major_prefill.py`, `page_attn_head_major_decode.py`, `batched_decode_head_major.py`, `tile_loop.py`).** The operand slice to the bucket (`[:num_blocks]`) replaced by `torch._check` statements on the declared range and a check tying the two declared tensors; the page body taking a tile of `G` pages per iteration under the existing `for_each_tile` walk; the batched kernel's chunk range passed in and its entry-local path used. The online softmax carry across pages is unchanged.

**What did not change.** The `torch.compile` calls (`dynamic=False` everywhere), the recorder, the compile guard, the KV write, the scheduler, the body's token buckets, the V2 runner.

## 4. What was changed in torch-spyre, and what was found there

Nothing was changed in torch-spyre's code for these measurements beyond the two local relaxations of section 2. What the work found on the torch-spyre side was filed on 5 November under the symbolic shapes epic #43:

| finding | issue | status |
|---|---|---|
| a copy of fewer rows than the maximum into a dynamic buffer is several times slower than a whole copy (13.5 ms for 9 of 16 rows against 2.3 ms whole and 1.2 ms plain, a 2 MB-per-row mask) | #6026 | open; the plugin's padded transfer is the workaround |
| `dynamic=` refused together with `device_layout=`, and once combined, `resize_` of a dynamic buffer in a custom layout reads back wrong (exact in the default layout) | #6027 | in progress |
| a compiled graph's output does not carry the declaration on #5179 alone; the carry is the design's second half, implemented on the cross-kernel branch with tests | #6029 | pull request to follow |
| `clone()` of a declared tensor raises `ConstraintViolationError` through the `dynamic=False` eager-op path; `detach()` carries the declaration, a view drops it | #6030 | open |
| a declared range cannot start at one granule (a contiguity check guard asked "is this size 1" before the stride) | #6018 | resolved, pull request to follow |
| the loop iteration cap of 32 was an assumption about per-slot launch cost; caps of 128 and 256 measured with no observed cost (section 8) | #43 | a per-declaration cap is planned |

Two things were also checked at the kernel level before any end-to-end number: a dynamic program and the static program at the same page count launch in the same time within 1 percent (the loop costs nothing against the unrolled program), and the kernels are bit-exact against the static ones on a device probe at 2 and 8 pages per iteration.

## 5. The test setup

Every experiment is a pair of legs in one run on one pod: the plugin as shipped (the control) and the same plugin with the switches on, each leg from a fresh, empty compile cache, in that order, with the cache removed afterwards. Nothing is quoted across runs or across pods as a gain, because the shipped plugin itself moves 2 to 6 percent between engine starts on the same build. A difference inside that band is reported as a wash.

**Correctness first.** Before any timing, each leg runs the same eight prompts (1200 to 3900 tokens at 2K and 4096; four prompts of 5000 to 17000 tokens at 32K) greedily through the vLLM `LLM` API, as one batch and the longest alone, 64 output tokens, and the output tokens of the switched leg are compared with the shipped leg's token by token, with the logprob gap over the agreeing prefix recorded. A leg whose tokens differ is not timed.

**Boot and latency.** `vllm bench latency`, the plugin's own tool. The first run of a leg from the empty cache gives the engine init time (the cold boot: model load, the body compiles, the attention compiles, warm-up), and the recorder's count of attention programs. The later runs give the latencies: 5 iterations after 2 warm-up iterations for the 1- and 8-sequence benches, 3 after 1 for the 32- and 64-sequence ones. The plugin's compile guard runs at `error`, so a compile leaking into a timed iteration fails the run instead of polluting it, and the iteration spread is recorded (a spread above 1.05 marks a leg as not clean). The benches: a 512-token prefill, 1100 tokens in and 64 out at 1, 8 and 32 (or 64) sequences, and a 32- (or 64-) sequence prefill of 1100 tokens. The decode step at full batch is the generation bench minus the prefill bench, divided by 64.

**Serving under load.** `vllm bench serve` against `vllm serve`, the plugin team's own parity method: 200 random prompts of 1024 tokens plus or minus 80 percent, 128 output tokens, concurrency 32 equal to `max_num_seqs`, prefix caching off, percentiles for TTFT, TPOT, ITL and E2EL. The team's internal traces are not available on the pods, so random prompts stand in.

**Warm start.** Every bench after a leg's first starts the engine from that leg's populated cache, so the `init engine` lines of the later benches give the populated-cache start for free.

## 6. Results on granite-3.3-8b-instruct, bf16, one card

### 6.1 Compile time by context length and sequence count

| configuration | attention programs, shipped | with the switches | cold boot, shipped | with the switches | gain |
|---|---|---|---|---|---|
| 2048 context, 4 sequences | 25 | 5 | 367 s | 200 s | 45 % |
| 4096 context, 32 sequences (the epic's first configuration) | 48 | 8 | 961 s | 370 s | 61 % |
| the same, with the body compiled at two token sizes instead of seven (32 and 512) | 48 | 8 | 961 s | 233 s | 76 % |
| 4096 context, 64 sequences | 54 | 9 | 1106 s | 424 s | 62 % |
| 32768 context, 4 sequences | 45 | 5 | 751 s | 335 s | 55 % |

Tokens identical in every row. The epic's 8-minute target (480 s) is met on its first configuration with the attention switches alone, and with room to spare once the body's buckets are cut, which is the plugin's own setting. The boot gap widens with the sequence count: going from 32 to 64 sequences cost the shipped plugin 145 s and six more programs, and the switched plugin one program and under a minute, because the shipped grid multiplies sequence buckets by page buckets and the dynamic one does not. The 32K boot is 335 s and not lower because the cap of 32 iterations forces 8 pages per iteration there and the two per-sequence programs then unroll 8 pages in their body (section 8 removes that).

### 6.2 Warm start from a populated cache

| configuration | shipped | with the switches |
|---|---|---|
| 2048, 4 sequences | 24.7 to 27.1 s | 15.1 to 16.2 s |
| 4096, 32 sequences | 45.6 to 51.3 s | 23.5 to 25.2 s |
| 4096, 64 sequences | 53 to 55 s | 27 to 30 s |
| 32768, 4 sequences | 40.1 to 42.7 s | 18.5 to 20.3 s |

About halved everywhere, since fewer programs are loaded from the cache. This is the number the plugin's populated-cache boot benchmark (#959) would track.

### 6.3 Run time, `vllm bench latency`, 1100 tokens in and 64 out

| configuration and bench | shipped | with the switches | difference |
|---|---|---|---|
| 4096 with 64 sequences, decode step at 64 sequences | 473 ms | 380 ms | 20 % faster |
| 4096 with 64 sequences, 64-sequence generation | 81.8 s | 74.1 s | 9 % faster |
| 4096 with 64 sequences, 64-sequence prefill | 51.5 s | 49.8 s | 3 % faster |
| 4096 with 64 sequences, 8 sequences | 17.0 s | 16.6 s | 2.5 % faster |
| 4096 with 64 sequences, 1 sequence | 9.96 s | 9.95 s | equal |
| 4096 with 64 sequences, prefill of 512 tokens | 0.231 s | 0.239 s | 3 % slower |
| 4096 with 32 sequences, decode step at 32 sequences | 298 ms | 258 ms | 13 % faster (two engines, the morning's shipped against the evening's switched) |
| 32768 with 4 sequences, decode step between page buckets | | | 7 to 9 % faster, flat at a bucket boundary |
| 32768 with 4 sequences, prefill of 9K to 31K prompts | | | 10 to 18 % faster |

Where the gain comes from: at full batch the attention step reads the real page count instead of the bucket's, and the larger the batch, the larger the share of a step that attention is, which is why the step gain grows from 13 percent at 32 sequences to 20 at 64. Where there is none: a short-context decode step at one sequence is a weight pass and does not move; the 512-token prefill pays about 3 percent for its per-sequence mask stack being padded to the maximum before the copy.

### 6.4 The two fixes of 4 November, measured in one run at 4096 with 32 sequences

The first matrix had two losses at short context: 7 percent slower at one sequence (the batched kernel's minimum of two chunks padding a lone request) and 8 percent slower at 8 sequences (a quarter of it the per-step reserved transfers, the rest unattributed). Two switches were built against them and run in one four-leg run (the control being the plugin with both axes declared and no fix):

| | control, both axes declared | one-chunk program | padded transfer | both |
|---|---|---|---|---|
| attention programs recorded | 2 + 6 | 2 + 11 | 2 + 6 | 2 + 11 |
| engine init, cold | 393 s | 436 s | 369 s | 426 s |
| prefill 512 | 0.256 s | 0.250 s | 0.238 s, 7 % faster | 0.239 s |
| decode, 1 seq | 10.05 s | 9.61 s, 4.4 % faster | 9.95 s | 10.01 s |
| decode, 8 seqs | 17.31 s | 17.59 s | 16.55 s, 4.4 % faster | 17.00 s |
| prefill, 32 seqs | 25.72 s | 25.71 s | 24.67 s, 4.1 % faster | 24.67 s |
| decode, 32 seqs | 43.09 s | 43.29 s | 41.19 s, 4.4 % faster | 41.22 s |
| decode step at 32 seqs | 271 ms | 275 ms | 258 ms, 4.8 % faster | 259 ms |

The padded transfer is a consistent gain on every bench and turns the 8-sequence loss into a 2 percent gain against the shipped plugin. The one-chunk program removes the floor at one sequence (10.05 to 9.61 s, reproduced three times from warm caches and once more in a cold trace). The combined leg did not show the one-chunk gain in that run, four readings at 10.0 s, but a dispatch trace the next night found no interaction (both legs 9.62 s with identical dispatch in all 40 layers), so those readings stand unexplained and both switches are recommended together. A third switch, persistent buffers refilled in place, was refused: `resize_` plus `copy_` on a reserved tensor left the kernel reading stale data (the `resize_` part of #6027).

### 6.5 Serving under load, `vllm bench serve`, 4096 context with 32 sequences

| | shipped | with the switches | with the switches, no padded transfer |
|---|---|---|---|
| server boot to ready | 1176 s | 379 s | 373 s |
| attention programs | 48 | 8 | 8 |
| output throughput | 84.3 tok/s | 83.5 tok/s | 80.4 tok/s |
| TTFT median, p99 | 2337 ms, 26.5 s | 2323 ms, 26.3 s | 2563 ms, 28.7 s |
| TPOT median, p99 | 328 ms, 369 ms | 331 ms, 367 ms | 343 ms, 392 ms |
| ITL median, p99 | 345 ms, 443 ms | 355 ms, 435 ms | 372 ms, 454 ms |
| E2EL median, p99 | 45.6 s, 86.5 s | 45.9 s, 85.2 s | 47.6 s, 88.8 s |

Under their own method the switched plugin is level with the shipped one, every figure inside the run-to-run band, and the server boots in a third of the time. The decode-step gain of 6.3 does not show once prefills of mixed lengths interleave with the decodes. The padded transfer is what keeps it level: without it the switched plugin runs 4 to 11 percent behind on every figure. The shipped leg reproduced within 1 percent on the second pod and across two runs, so the method is stable on this build.

## 7. Results on the other checkpoints, 4096 context with 32 sequences, one run each

| model | programs, shipped to switched | cold boot, shipped | with the switches | boot gain | decode step at 32 sequences | tokens |
|---|---|---|---|---|---|---|
| granite-3.3-8b-instruct-FP8 | 48 to 8 | 1079 s | 494 s | 54 % | 247 to 203 ms, 18 % faster | identical |
| granite-4.1-8b | 48 to 8 | 922 s | 379 s | 59 % | 4 % faster (one bench leaked a compile) | identical |
| granite-4.1-8b-fp8 with the fp8 KV cache (the epic's fourth configuration) | 48 to 8 | 1101 s | 504 s | 54 % | 199 to 182 ms, 9 % faster | identical |
| granite-4.2-8b | 48 to 8 | 971 s | 422 s | 57 % | 316 to 276 ms, 13 % faster | identical |
| granite-4.2-8b-fp8 | 48 to 8 | 1209 s | 525 s | 57 % | not quoted: a card fault during that run left the switched leg's benches erratic | not compared, the fault hit the probe |
| Ministral-3-14B, with its vision tower | 48 to 8 | 1231 s | 428 s | 65 % | 384 to 348 ms, 9 % faster | identical |
| Ministral-3-8B, with its vision tower | 48 to 8 | 1067 s | 409 s | 62 % | 311 to 282 ms, 9 % faster | identical |

The picture is the same on every checkpoint: 48 attention programs down to 8, the boot 2.2 to 2.9 times faster, tokens identical, the step at full batch faster by 4 to 18 percent and the small batches level. The FP8 granite-3.3 runs its benches about a third faster than the bf16 one, so attention is a larger share of its step and its gain at full batch is the largest. On the 1-sequence bench these legs ran without the single-chunk program, so they show the two-chunk floor (3 to 6 percent) that the bf16 run removed with it. Against the 8-minute target: granite-3.3-8b, granite-4.1-8b and granite-4.2-8b are under it with the attention switches alone; the three FP8 checkpoints miss it by 14 to 45 s on the body's buckets, which one body bucket brings under it.

## 8. The loop iteration cap at 32K, two pods

The cap of 32 iterations was an assumption (each slot costs host work at launch). It was measured at 32K with 4 sequences: the shipped plugin against the switches at 8 pages per iteration (today's cap), 2 pages (cap 128) and 1 page (cap 256), each a cold boot, the four long-prompt probe and five benches, on both pods.

| | shipped, pod 1 / pod 2 | 8 pages per iteration, pod 1 | 2 pages, pod 1 / pod 2 | 1 page, pod 1 / pod 2 |
|---|---|---|---|---|
| cold boot | 766 / 767 s | 347 s | 294 / 295 s | 236 / 243 s |
| prefill 512 | 0.232 / 0.229 s | 0.353 s | 0.271 / 0.270 s | 0.267 / 0.270 s |
| prefill 9000, 1 seq | 10.00 / 10.01 s | 9.28 s | 9.13 / 8.99 s | 9.03 / 9.01 s |
| prefill 17000, 1 seq | 28.94 / 28.83 s | 24.69 s | 24.32 / 24.28 s | 24.39 / 24.40 s |
| 17000 in 64 out, 1 seq | 40.36 / 40.36 s | 35.38 s | 35.17 / 34.95 s | 35.03 / 35.10 s |
| 5000 in 64 out, 4 seqs | 28.74 / 28.67 s | 28.17 s | 27.37 / 27.53 s | 27.38 / 27.51 s |
| tokens | the reference | identical | identical | identical |

Caps of 128 and 256 compile, boot and give identical tokens on both pods, with logprob gaps at or under 0.0001. The run time at 1 and 2 pages per iteration is equal within 0.5 percent on every bench, so 256 launch slots cost nothing measurable at this size. The boot falls with the tile because the per-sequence programs compile faster the fewer pages they unroll (249, 196 to 201 and 136 to 144 s of recording): 32K boots in under four minutes at one page per iteration against 12.8 minutes shipped, 69 percent less. The long prompts gain 10 to 16 percent at every tile. The 512-token prefill stays 15 to 18 percent behind at every tile, so that cost is not the pages walked but the mask stack padded to all 256 pages before the copy (33 MB a step), the short-request cost of a long-context deployment that granularity bands or a mask built on the device would remove. The two pods agree within 0.5 percent on every cell.

## 9. What was not measured, and why

- Tensor-parallel sizes above one: one card per pod. The epic's six TP4 rows stay as arithmetic (48 attention programs to 8 at any context).
- 32K context with 32 sequences: the batched kernel's chunk count at one page per chunk exceeds even a cap of 256, and the KV cache for it was not tried on one card.
- Sliding-window models: the Gemma checkpoints were gated or failed at config load in the plugin.
- The encoder and vision paths: they need the carry across compiled graph outputs (#6029) and were not attempted.
- granite-4.2-8b-fp8's run time: a PCIe bus fence on the card during its probe; its boots stand.
- The shared volume's quota killed four stages over the two days (a serve control, two dispatch traces, a download); every one was re-run or recorded as lost, and no number from a run the quota touched is quoted.

## 10. What the numbers say, in order of weight

1. Compile time is the gain, and it is large and uniform: 48 attention programs to 8 on every checkpoint tried, the cold boot 2.2 to 2.9 times faster, the epic's first configuration under its 8-minute target with the attention switches alone, and the gap widening with the sequence count and shrinking with nothing.
2. Correctness held everywhere: tokens identical to the shipped plugin at every size from 2K to 32K, batch and single, on seven checkpoints, and at caps of 32, 128 and 256.
3. Run time gains where attention is a large share of the step: 13 to 20 percent a decode step at full batch, 10 to 18 percent on long prompts, nothing at short context, and level under the serving load, where the padded transfer is the part that keeps it level.
4. The next lever on boot is the body's token buckets, the plugin's own setting: 137 s of the 370 s on the epic's first configuration, and what the FP8 checkpoints need to reach the target.
5. The cap of 32 iterations is not a cost at 128 or 256, so a long-context deployment can run one page per iteration and boot in under four minutes at 32K.
