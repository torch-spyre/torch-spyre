# Symbolic shapes in spyre-inference: the build, the integration, the test setup and the results

These are the initial results of a proof of concept: torch-spyre's dynamic dimensions applied to spyre-inference's paged attention, measured with the plugin's own tools. 

## 1. The idea in one paragraph

spyre-inference compiles one static-shape program per bucket of every varying size, and for paged attention that means one program per (query bucket, page-count bucket) for the per-sequence kernels and one per (sequence-count bucket, page-count bucket) for the batched decode kernel, all compiled at start. torch-spyre's dynamic dimension lets one dimension of a tensor vary over a declared range, served by one program whose device loop runs for the real size at launch. The integration declares the two counts that paged attention varies on, the page count a sequence's kernel walks and the chunk count the batched kernel walks, so that one program per kernel per query bucket serves every context length and every sequence count. The transformer body is left bucketed on purpose: a decode step is a weight pass and a loop over its rows would stream the weights once per iteration.

## 2. The base build

| component | what was used |
|---|---|
| torch-spyre | main as of 2 November, built on top of #5179 (the runtime half: the declaring transfer, the buffer reserved at the maximum, the contract on the tensor) and #5186 (the compile half: symbolic loop iteration counts through Inductor to the device program, being merged as #5891, #5912, #5917 and a fourth part), with cross-kernel support (the contract carried across compiled graph outputs, #6029) |
| two small adjustments to that build | `.to()` accepting `dynamic=` together with `device_layout=` (#6027, needed by the batched kernel's chunk index), and the reservation floor set to 2 rows (`MIN_RESERVED_DIM_SIZE=2`, the default of 8 refuses a two-page minimum) |
| PyTorch | 2.13.0 |
| vLLM | 0.32.0 |
| spyre-inference | `1881aa45fa` plus the attention patch (987 lines across 8 files, behind switches, off by default) |
| hardware | one Spyre card; a second card with the same software stack was used to repeat runs, and the two agree within 1 percent on every static-shape bench |

The plugin's own defaults were kept throughout: block size 128 tokens, the head-major KV layout, batched decode on, prefix caching off, body compile sizes 1, 2, 4, 8, 16, 32 and 512 (64 added for the 64-sequence run), the default power-of-two KV bucket ladder for the static-shape runs.

## 3. What was changed in spyre-inference

Eight files, 987 patch lines, every change behind a switch that is off by default so the existing path is untouched.

**Switches (`envs.py`).** One to declare the page axis of the per-sequence prefill and decode kernels, one to declare the chunk axis of the batched decode kernel, one to override the pages per loop iteration (otherwise derived from the cap), one to record a static single-chunk program beside the dynamic one for a step that fits one chunk, one to pad each per-step tensor to the maximum on the host before the copy, and a diagnostic that logs which batched program a step dispatches.

**The bucketer (`spyre_attn_bucketer.py`).** A declared page range per deployment instead of the KV bucket ladder: the maximum is the page count of `max_model_len`, the pages per iteration are `ceil(max_pages / MAX_BUCKETS)` (1 page up to 4096 context at the default cap of 32, 8 pages at 32K), and the minimum is two iterations. For the batched kernel the pages per chunk stay pinned at `32 // num_seqs`, so every chunk fills the cores, and the chunk count becomes the declared range, from 2 to the chunks the longest context takes; the sequence buckets stay. The variant list becomes one dynamic variant per query bucket and one per sequence bucket instead of the (query, pages) and (sequences, pages) grids. The bucketer's cap reads torch-spyre's `MAX_BUCKETS` so the two never disagree.

**The metadata builder and the backend (`spyre_attn.py`, `spyre_head_major_attn.py`).** The real page count floored at the range's minimum instead of rounded to a bucket; the per-sequence page tables and mask stacks moved with `.to("spyre", dynamic=page_range)`; the batched kernel's chunk index moved with its chunk-major device layout and the declaration together, its chunk mask with the declaration; a per-step choice between the dynamic program and the static single-chunk program; the padded transfer; the batched query staged into a buffer of `max_num_batched_tokens` rows. No host slicing of a declared tensor anywhere.

**The kernels (`page_attn_head_major_prefill.py`, `page_attn_head_major_decode.py`, `batched_decode_head_major.py`, `tile_loop.py`).** The operand slice to the bucket (`[:num_blocks]`) replaced by `torch._check` statements on the declared range and a check tying the two declared tensors; the page body taking a tile of `G` pages per iteration under the existing `for_each_tile` walk; the batched kernel's chunk range passed in. The online softmax carry across pages is unchanged.

**What did not change.** The `torch.compile` calls (`dynamic=False` everywhere), the warm-up recorder, the compile guard, the KV write, the scheduler, the body's token buckets, the V2 runner.

## 4. The test setup

Every experiment is a pair of runs on one card, back to back: the plugin as it is today (static shapes) and the same plugin with dynamic shapes on, each from a fresh, empty compile cache. Nothing is quoted across separate runs as a gain, because the static-shape plugin itself moves 2 to 6 percent between engine starts on the same build. A difference inside that band is reported as level.

**Correctness first.** Before any timing, each run goes through the same eight prompts (1200 to 3900 tokens at 4096 context; four prompts of 5000 to 17000 tokens at 32K) greedily through the vLLM `LLM` API, as one batch and the longest alone, 64 output tokens, and the output tokens with dynamic shapes are compared with the static-shape run's token by token, with the logprob gap over the agreeing prefix recorded. A run whose tokens differ is not timed.

**Boot and latency.** `vllm bench latency`, the plugin's own tool. The first run from the empty cache gives the engine init time (the cold boot: model load, the body compiles, the attention compiles, warm-up) and the count of attention programs compiled. The later runs give the latencies: 5 iterations after 2 warm-up iterations for the 1- and 8-sequence benches, 3 after 1 for the 32- and 64-sequence ones. The plugin's compile guard runs at `error`, so a compile leaking into a timed iteration fails the run instead of polluting it, and the iteration spread is checked. The benches: a 512-token prefill, 1100 tokens in and 64 out at 1, 8 and 32 (or 64) sequences, and a 32- (or 64-) sequence prefill of 1100 tokens. The decode step at full batch is the generation bench minus the prefill bench, divided by 64.

**Server boot.** `vllm serve` at 4096 context with 32 sequences, timed from start to the server reporting ready, the number the boot-time epic defines.

**Warm start.** Every bench after the first starts the engine from the populated cache, so the later `init engine` lines give the populated-cache start for free.

## 6. Results on granite-3.3-8b-instruct, bf16, one card

### 6.1 Compile time by context length and sequence count

| configuration | attention programs, today | with dynamic shapes | cold boot, today | with dynamic shapes | gain |
|---|---|---|---|---|---|
| 4096 context, 32 sequences (the epic's first configuration) | 48 | **8** | 961 s | **370 s** | **61 %** |
| the same, with the body compiled at two token sizes instead of seven (32 and 512) | 48 | **8** | 961 s | **233 s** | **76 %** |
| 4096 context, 64 sequences | 54 | **9** | 1106 s | **424 s** | **62 %** |
| 32768 context, 4 sequences | 45 | **5** | 751 s | **335 s** | **55 %** |
| server boot to ready, `vllm serve`, 4096 context, 32 sequences | 48 | **8** | 1176 s | **379 s** | **68 %** |

Tokens identical in every row. The boot-time epic's 8-minute target (480 s) is met on its first configuration with dynamic attention alone, and with room to spare once the body's buckets are cut, which is the plugin's own setting. The gap widens with the sequence count: going from 32 to 64 sequences cost the static-shape plugin 145 s and six more programs, and the dynamic-shape plugin one program and under a minute, because the static grid multiplies sequence buckets by page buckets and the dynamic one does not. The 32K boot is 335 s and not lower only because the default cap of 32 iterations forces 8 pages per iteration there; section 8 takes it under four minutes.

### 6.2 Warm start from a populated cache

| configuration | today | with dynamic shapes |
|---|---|---|
| 4096, 32 sequences | 45.6 to 51.3 s | **23.5 to 25.2 s** |
| 4096, 64 sequences | 53 to 55 s | **27 to 30 s** |
| 32768, 4 sequences | 40.1 to 42.7 s | **18.5 to 20.3 s** |

About halved everywhere, since fewer programs are loaded from the cache. This is the number the plugin's populated-cache boot benchmark (#959) would track.

### 6.3 Run time, `vllm bench latency`, 1100 tokens in and 64 out

| configuration and bench | today | with dynamic shapes | difference |
|---|---|---|---|
| 4096 with 64 sequences, decode step at 64 sequences | 473 ms | **380 ms** | **20 % faster** |
| 4096 with 64 sequences, 64-sequence generation | 81.8 s | **74.1 s** | **9 % faster** |
| 4096 with 64 sequences, 64-sequence prefill | 51.5 s | **49.8 s** | **3 % faster** |
| 4096 with 64 sequences, 8 sequences | 17.0 s | **16.6 s** | **2.5 % faster** |
| 4096 with 64 sequences, 1 sequence | 9.96 s | 9.95 s | level |
| 4096 with 64 sequences, prefill of 512 tokens | 0.231 s | 0.239 s | within 3 percent |
| 4096 with 32 sequences, decode step at 32 sequences | 298 ms | **258 ms** | **13 % faster** (two engine starts, the static-shape run of the morning against the dynamic-shape run of the evening) |
| 32768 with 4 sequences, decode step between page buckets | | | **7 to 9 % faster**, level at a bucket boundary |
| 32768 with 4 sequences, prefill of 9K to 31K prompts | | | **10 to 18 % faster** |

Where the gain comes from: at full batch the attention step reads the real page count instead of the bucket's, and the larger the batch, the larger the share of a step that attention is, which is why the step gain grows from 13 percent at 32 sequences to 20 at 64. A short-context decode step at one sequence is a weight pass and does not move, so it stays level.

### 6.4 Two refinements, measured in one run at 4096 with 32 sequences

The first version of the proof of concept paid for two things at short context: the batched kernel's minimum of two chunks padded a lone request, and the per-step reserved copies below the maximum were slow (#6026). Two refinements were built and measured in one run against the first version:

| | first version | with the padded transfer | with the single-chunk program |
|---|---|---|---|
| prefill 512 | 0.256 s | **0.238 s, 7 % faster** | 0.250 s |
| decode, 1 seq | 10.05 s | 9.95 s | **9.61 s, 4.4 % faster** |
| decode, 8 seqs | 17.31 s | **16.55 s, 4.4 % faster** | 17.59 s |
| prefill, 32 seqs | 25.72 s | **24.67 s, 4.1 % faster** | 25.71 s |
| decode, 32 seqs | 43.09 s | **41.19 s, 4.4 % faster** | 43.29 s |
| decode step at 32 seqs | 271 ms | **258 ms, 4.8 % faster** | 275 ms |

The padded transfer, which moves each per-step tensor whole at the maximum and resizes it down, is a gain on every bench and is what every later run uses. The single-chunk program removes the floor at one sequence (10.05 to 9.61 s, reproduced three times and once more in a cold run with a dispatch trace, which showed both refinements running together without interfering). Both are on in every result from section 6.5 onward.

## 7. Results on the other checkpoints, 4096 context with 32 sequences, one run each

| model | attention programs, today | with dynamic shapes | cold boot, today | with dynamic shapes | boot gain | decode step at 32 sequences | tokens |
|---|---|---|---|---|---|---|---|
| granite-3.3-8b-instruct-FP8 | 48 | **8** | 1079 s | **494 s** | **54 %** | 247 to **203 ms, 18 % faster** | identical |
| granite-4.1-8b | 48 | **8** | 922 s | **379 s** | **59 %** | **4 % faster** | identical |
| granite-4.1-8b-fp8 with the fp8 KV cache (the epic's fourth configuration) | 48 | **8** | 1101 s | **504 s** | **54 %** | 199 to **182 ms, 9 % faster** | identical |
| granite-4.2-8b | 48 | **8** | 971 s | **422 s** | **57 %** | 316 to **276 ms, 13 % faster** | identical |
| granite-4.2-8b-fp8 | 48 | **8** | 1209 s | **525 s** | **57 %** | not measured, a hardware fault on the card during that run | not compared, the fault hit the probe |
| Ministral-3-14B, with its vision tower | 48 | **8** | 1231 s | **428 s** | **65 %** | 384 to **348 ms, 9 % faster** | identical |
| Ministral-3-8B, with its vision tower | 48 | **8** | 1067 s | **409 s** | **62 %** | 311 to **282 ms, 9 % faster** | identical |

The picture is the same on every checkpoint: 48 attention programs down to 8, the boot 2.2 to 2.9 times faster, tokens identical, the decode step at full batch 4 to 18 percent faster and the small batches level. The FP8 granite-3.3 runs its benches about a third faster than the bf16 one, so attention is a larger share of its step and its gain at full batch is the largest. Against the 8-minute target: granite-3.3-8b, granite-4.1-8b and granite-4.2-8b are under it with dynamic attention alone; the three FP8 checkpoints miss it by 14 to 45 s on the body's buckets, which one body bucket brings under it.

## 8. The loop iteration cap at 32K

The cap of 32 iterations was an assumption (each slot costs host work at launch). It was measured at 32K with 4 sequences: the static-shape plugin against dynamic shapes at 8 pages per iteration (today's cap), 2 pages (cap 128) and 1 page (cap 256), each a cold boot, the four long-prompt probe and five benches, in one run. The whole run was repeated on the second card and agreed within 0.5 percent on every cell.

| | today | 8 pages per iteration (cap 32) | 2 pages (cap 128) | 1 page (cap 256) |
|---|---|---|---|---|
| cold boot | 766 s | **347 s** | **294 s** | **236 s** |
| prefill of a 9K prompt | 10.00 s | **9.28 s** | **9.13 s** | **9.03 s** |
| prefill of a 17K prompt | 28.94 s | **24.69 s** | **24.32 s** | **24.39 s** |
| 17K in, 64 out, 1 sequence | 40.36 s | **35.38 s** | **35.17 s** | **35.03 s** |
| 5K in, 64 out, 4 sequences | 28.74 s | **28.17 s** | **27.37 s** | **27.38 s** |
| prefill of 512 tokens | 0.232 s | 0.353 s | 0.271 s | 0.267 s |
| tokens | the reference | identical | identical | identical |

Caps of 128 and 256 compile, boot and give identical tokens, with logprob gaps at or under 0.0001. The run time at 1 and 2 pages per iteration is equal within 0.5 percent on every bench, so 256 launch slots cost nothing measurable at this size. The boot falls with the tile because the per-sequence programs compile faster the fewer pages they unroll: **32K boots in under four minutes at one page per iteration against 12.8 minutes today, 69 percent less.** The long prompts gain **10 to 16 percent** at every tile. The 512-token prefill is the one bench behind at 32K, and the reason is known: its mask stack is padded to all 256 pages before the copy (33 MB a step), the short-request cost of a long-context deployment that a shorter range for short requests, or a mask built on the device, removes.

## 9. What was not measured yet

- Tensor-parallel sizes above one: one card. The epic's six TP4 rows stay as arithmetic (48 attention programs to 8 at any context).
- 32K context with 32 sequences: the batched kernel's chunk count at one page per chunk exceeds even a cap of 256.
- Sliding-window models: the Gemma checkpoints were gated or failed at config load in the plugin.
- The encoder and vision paths: they need the carry across compiled graph outputs (#6029).
- Serving under load with mixed prompt lengths: the server boot is in 6.1; the throughput and latency figures under a random-length load are being understood and are not part of this initial set.

## 10. What the numbers say, in order of weight

1. Compile time is the gain, and it is large and uniform ( more than 2X improvement for every tested case ) : 48 attention programs to 8 on every checkpoint tried, the server ready in a third of the time, and the epic's first configuration under its 8-minute target with dynamic attention alone.
2. Correctness held everywhere: tokens identical to today's plugin at every size from 4096 to 32K, batch and single, on seven checkpoints, and at caps of 32, 128 and 256.
3. Run time gains where attention is a large share of the step: 13 to 20 percent a decode step at full batch, 10 to 18 percent on long prompts, and level at short context.
4. The next lever on boot is the body's token buckets, the plugin's own setting: 137 s of the 370 s on the epic's first configuration, and what the FP8 checkpoints need to reach the target.
5. The cap of 32 iterations is not a cost at 128 or 256, so a long-context deployment can run one page per iteration and boot in under four minutes at 32K.
