# RFC: Platform simplification through symbolic shapes, for compile time and runtime

| Field | Value |
|---|---|
| Status | Draft, for discussion under the boot-time epic |
| Created | 2026-11-05 |
| Tracking | [#1432 Reduce spyre-inference boot/compile time](https://github.com/torch-spyre/spyre-inference/issues/1432), [#1515 Investigate how torch-spyre symbolic shape can reduce compile time](https://github.com/torch-spyre/spyre-inference/issues/1515) |
| Related | #1523, #1525, #1524 (the encoder items and their interim), #959, #566 (boot time as a tracked metric), #771 (denser block ladder), #942 (performance above 32 sequences), #1364 (compiles after warm-up), #1473 (vision encoders once dynamic shapes exist), #713 (long context); torch-spyre #5891, #5912, #5917, #5179, #5186 |

Terms used throughout. A **static-shape program** is what Spyre compiles today: one binary for one exact tensor shape. A **dynamic dimension** is a tensor dimension that torch-spyre lets a caller mark as variable over a range (a minimum, a maximum and a step between supported sizes, called the granularity). A **dynamic-shape program** is one binary compiled once over such a range, which runs at every size in it. A **bucket** is a fixed size the plugin rounds a varying quantity up to, so that a static-shape program can serve it. A **KV cache page** is 128 tokens of a sequence's history.

## 1. Motivation

Every program on Spyre is compiled for one exact shape. The plugin handles a varying size by rounding it up to a bucket and compiling one static-shape program per bucket, then compiling every bucket at start. For paged attention that is one program per (query bucket, page-count bucket) for the per-sequence kernels and one per (sequence-count bucket, page-count bucket) for the batched decode kernel. The number grows with the context length and with `max_num_seqs`, and so does the boot time. On granite-3.3-8b at the epic's first configuration (4096 context, 32 sequences, one card) that is 48 attention programs beside 7 body programs and 961 s of cold boot, and at 32K context with 4 sequences 45 attention programs and 751 s.

The epic's target is a boot under 8 minutes on row 1 and on six more rows at 32K with 32 sequences. The same bucketing also costs at run time, since a request pays for its bucket and not for its own length (#771, #1060), and it leaves compile hazards after warm-up whenever a shape escapes the bucket list (#1364).

torch-spyre is landing symbolic shapes: a dynamic dimension served by one dynamic-shape program whose device loop runs for the real size at launch. This RFC proposes to use that to remove the attention buckets from the plugin, shows what it measured, states what it does not change, and lists what has to land in torch-spyre first. It answers the seven questions of #1515 in the appendix.

## 2. Goals and non-goals

Goals: one attention program per kernel per query bucket, independent of the context length and of `max_num_seqs`. The attention page buckets, their presets, the per-variant grids and the per-variant warm-up compiles removed. Boot time on row 1 under the target with no cache. Output tokens identical to today's. No regression at short context, a gain at long context.

Non-goals: the transformer body's token buckets (a decode step is a weight pass, so the body wants one flat row count per phase and a loop over rows does not help it), tensor parallel sizes above 1 (not measurable on the one-card pods), the vision towers (#1473, a later step on the same mechanism), jagged attention, prefix caching changes.

## 3. What torch-spyre provides

A caller marks one dimension of a host tensor as dynamic when it moves it to the device:

```python
x_dev = x_cpu.to("spyre", dynamic={0: {"min": m, "max": M, "granularity": G}})
```

The device buffer is allocated once at `M` rows with the strides of `M`, the real rows are copied in, and dim 0 is marked dynamic for Dynamo (torch-spyre #5179). Inside a compiled kernel the dynamic size may reach exactly one thing: the iteration count of a `for_each_tile` loop over that dimension. Every address, stride and buffer extent is built at `M`, and the loop's iteration count is set at launch from the tensor's real size, so one dynamic-shape program serves every size in `[m, M]` that is a multiple of `G` (torch-spyre #5186, split into #5891 merged, #5912 approved, #5917 and a fourth part open). The compile is one Inductor compile at the maximum, not one per size.

Limits of the build measured (torch-spyre `43c4150a`: main plus #5179 plus #5186 plus one cross-kernel commit): dim 0 only, `M / G` at most 32 loop iterations (`MAX_BUCKETS`), and the dynamic dimension must be a loop's tile axis and nothing else.

## 4. Proposal

### 4.1 What becomes dynamic

| kernel | dynamic dimension | one program per |
|---|---|---|
| per-sequence prefill and decode (head-major) | the page count of the page table and the mask stack | query bucket |
| batched decode (head-major) | the chunk count of the chunk page index and the chunk mask (pages per chunk pinned at `32 // num_seqs`) | sequence-count bucket |

Everything else in the kernels keeps a fixed extent, so the dynamic size reaches only the page walk's iteration count. The kernels already walk pages with `for_each_tile`, so the kernel change is small: the operand slice (`[:num_blocks]`) replaced by four `torch._check` statements on the range, and the page body taking a tile of `G` pages.

### 4.2 What leaves the plugin

The KV page bucket list and its presets (`SPYRE_ATTN_KV_LADDER`, `SPYRE_ATTN_KV_BUCKETS`), the padding of a request to a page bucket, the (sequences, pages) variant grid of the batched kernel, the per-variant warm-up loops over those grids, and the main reason for the jagged grouping (it exists to avoid padding every sequence to the longest bucket, which exact chunk counts make small). The bucketer keeps the query buckets and the body's row counts. The warm-up and the compile guard stay, with far fewer things to compile and to watch. The V2 runner uses the same warm-up, so nothing here is specific to V1.

### 4.3 How the per-step tensors move to the device

On the current runtime, copying fewer rows than the maximum into a dynamic buffer is about ten times slower than copying the whole buffer. The plugin therefore pads each per-step tensor (page tables, mask stacks, the chunk mask) to the maximum on the host, copies it whole, and resizes the device tensor down to the real count, which moves no data. Measured exact, and 2.8 ms for a 512-query mask stack where the partial copy took 19 ms. Under load this padding is what keeps serving level with today's plugin (section 5.4).

### 4.4 Rollout

Behind switches, off by default, in one pull request against the plugin once torch-spyre's pins carry the stack and #5179: `SPYRE_ATTN_SYMBOLIC_PAGES` (the per-sequence kernels), `SPYRE_ATTN_SYMBOLIC_BATCHED` (the batched kernel), `SPYRE_ATTN_SYMBOLIC_PAD_TRANSFER` (the padding of section 4.3), `SPYRE_ATTN_SYMBOLIC_ONE_CHUNK` (a static program for the single-chunk case of the batched kernel, so a lone short request is not padded to two chunks). The plugin's attention device tests pass with them on. The bucket path stays until the switches have run the CI configurations, then the buckets go.

## 5. Evidence

### 5.1 Method

Everything below was measured with the plugin's own tools on one Spyre card (`vivekmankar-spyre-dev-pf`), one build (torch-spyre `43c4150a`, spyre-inference `1881aa45fa` plus the patch), today's plugin against the same plugin with the switches on, a fresh compile cache for every leg:

- Boot and latency: `vllm bench latency`, the first run of each leg from an empty cache gives the engine init time (the boot), the later runs give the latencies (5 iterations after 2 warm-ups for the small benches, 3 after 1 for the large ones). A compile guard at `error` fails any run where a compile leaks into a timed iteration.
- Serving: `vllm bench serve` against `vllm serve`, 200 random prompts of 1024 tokens plus or minus 80 percent, 128 output tokens, concurrency 32 (equal to `max_num_seqs`, the team's parity rule), prefix caching off.
- Correctness, before any number is quoted: eight prompts of 1200 to 3900 tokens (four of 5000 to 17000 at 32K) run greedily through the vLLM `LLM` API as one batch and the longest alone, output tokens compared with today's plugin, logprobs recorded.

Each pair below comes from one run. The plugin's own run-to-run drift between engine starts is 2 to 6 percent, so a difference inside that band is reported as a wash.

### 5.2 granite-3.3-8b-instruct, bf16: compile time

| configuration | attention programs, today | attention programs, with dynamic | cold boot, today | cold boot, with dynamic | gain |
|---|---|---|---|---|---|
| **4096 context, 32 sequences (the epic's first configuration)** | **48** | **8** | **961 s** | **370 s** | **61 percent** |
| **the same, with the body's decode buckets cut from six to one as well (compile sizes 32 and 512 instead of 1, 2, 4, 8, 16, 32 and 512)** | **48** | **8** | **961 s** | **233 s** | **76 percent** |
| **4096 context, 64 sequences** | **54** | **9** | **1106 s** | **424 s** | **62 percent** |
| 32768 context, 4 sequences | 45 | 5 | 751 s | 335 s | 55 percent |

Output tokens were identical to today's plugin in every row (8 prompts at 4096, 4 at 2048, 4 prompts of up to 17K tokens at 32K). The boot gap widens with the sequence count: going from 32 to 64 sequences cost today's plugin 145 s and six more programs, and the dynamic-shape plugin one more program and under a minute. With a populated compile cache the engine start halves too: 25 to 27 s today against 15 to 16 s at 2048, 53 to 55 s against 27 to 30 s at 4096 with 64 sequences.

### 5.3 granite-3.3-8b-instruct, bf16: run time (`vllm bench latency`, 1100 tokens in, 64 out)

Where the gain is large:

| configuration and bench | today | with dynamic | gain |
|---|---|---|---|
| **4096 with 64 sequences, decode step at 64 sequences** | **473 ms** | **380 ms** | **20 percent** |
| **4096 with 32 sequences, decode step at 32 sequences** | **298 ms** | **258 ms** | **13 percent** (two engines, the morning's and the evening's) |
| **32768 with 4 sequences, prefill of 9K to 31K prompts** | | | **10 to 18 percent** |
| 4096 with 64 sequences, 64-sequence generation of 64 tokens | 81.8 s | 74.1 s | 9 percent |

The gain is fewer pages walked: at full batch the attention step reads the real page count instead of the bucket's, and the larger the batch, the larger the share of the step that attention is. Every other bench is level or better, inside or just above the plugin's own run-to-run band: the 64-sequence prefill 3 percent faster, 8 sequences 2.5 percent faster, the 32K decode step 7 to 9 percent faster between page buckets and flat at a bucket boundary, 1 sequence equal (a short-context decode step is a weight pass and does not move), and the 512-token prefill within 3 percent.

### 5.4 granite-3.3-8b-instruct, bf16: serving under load (`vllm bench serve`, 4096 context, 32 sequences)

| | today | dynamic |
|---|---|---|
| server boot to ready | 1176 s | 379 s |
| attention programs | 48 | 8 |
| output throughput | 84.3 tok/s | 83.5 tok/s |
| TTFT median, p99 | 2337 ms, 26.5 s | 2323 ms, 26.3 s |
| ITL median, p99 | 345 ms, 443 ms | 355 ms, 435 ms |

Throughput and latency are a wash under this load, every figure inside the plugin's own run-to-run band. The decode-step gain of 5.3 does not show once prefills of mixed lengths interleave with the decodes. The padding of section 4.3 is what keeps it level: without it the dynamic-shape plugin runs 4 to 11 percent behind under the same load (80.4 tok/s, median ITL 372 ms, a third run the same night).

### 5.5 granite-3.3-8b-instruct-FP8: the same picture on FP8 weights

| 4096 context, 32 sequences | today | with dynamic |
|---|---|---|
| cold boot | 1079 s | 494 s |
| decode step at 32 sequences | 247 ms | 203 ms |
| 32-sequence generation, 1100 in 64 out | 36.1 s | 32.4 s |
| 32-sequence prefill | 20.2 s | 19.4 s |
| 8 sequences, 1100 in 64 out | 12.1 s | 11.7 s |
| 1 sequence, 1100 in 64 out | 6.23 s | 6.63 s |
| prefill of 512 tokens | 0.178 s | 0.183 s |
| correctness | the reference | tokens identical, 8 prompts |

The FP8 checkpoint runs its benches about a third faster than the bf16 one, so attention is a larger share of its step and the gain at full batch is larger: boot 54 percent less, the decode step 18 percent faster at 32 sequences. At one sequence and on the 512-token prefill the two are within a few percent of each other, the small batches being where attention is the smallest share of the step; that run did not have the single-chunk program of section 4.4 on.

### 5.6 granite-3.3-8b-instruct, bf16: the loop iteration cap

The cap of 32 loop iterations sets 8 pages per iteration at 32K. A run of the 32K configuration at caps of 32, 128 and 256 (8, 2 and 1 pages per iteration) is in progress on 5 November, measuring the boot, the long-prompt prefill and correctness at each. Its rows go here when they land. Until then no number is claimed for it.

### 5.7 Other models, each one run, 4096 context with 32 sequences, tokens identical

| model | attention programs, today | with dynamic | cold boot, today | cold boot, with dynamic | boot gain | decode step at 32 sequences |
|---|---|---|---|---|---|---|
| **granite-3.3-8b-instruct-FP8** | 48 | 8 | 1079 s | 494 s | 54 percent | **18 percent faster** |
| granite-4.1-8b | 48 | 8 | 922 s | 379 s | 59 percent | 4 percent faster |
| granite-4.1-8b-fp8 with the fp8 KV cache (the epic's fourth configuration) | 48 | 8 | 1101 s | 504 s | 54 percent | 9 percent faster |
| **granite-4.2-8b** | 48 | 8 | 971 s | 422 s | 57 percent | **13 percent faster** |
| granite-4.2-8b-fp8 | 48 | 8 | 1209 s | 525 s | 57 percent | not quoted, the card faulted during that run |
| **Ministral-3-14B, with its vision tower** | 48 | 8 | 1231 s | 428 s | **65 percent** | 9 percent faster |
| **Ministral-3-8B, with its vision tower** | 48 | 8 | 1067 s | 409 s | **62 percent** | 9 percent faster |

Boot is 2.2 to 2.9 times faster on every model tried, the same 48 to 8 programs each time, and the dynamic-shape plugin boots every one of them in 7 to 9 minutes where today's takes 15 to 21.

Against the epic's 8-minute target at TP1, 4096 with 32 sequences: granite-3.3-8b, granite-4.1-8b and granite-4.2-8b are under it with the dynamic-shape attention alone, and the three FP8 checkpoints miss it by 14 to 45 s on the body's token buckets, which this proposal does not touch and which one body bucket brings under it (233 s on bf16).

## 6. What dynamic shapes do not change

- The transformer body. A decode step at short context is a weight pass (137 ms a step for 16 GB of fp16 weights at one sequence), so attention changes do not move it, and time to first token for short prompts likewise. The body's token buckets are where the remaining boot time sits once attention is down to 8 programs: compiling the body at 7 token buckets against 2 is 137 s of the 370 s on the epic's first configuration.
- The loop iteration cap. A dynamic-shape program is compiled with one slot per possible iteration, and the host sets the iteration count at every launch, so the number of iterations is capped (32 today) and long contexts take several pages per iteration. Finer granularity at 32K is a trade between compile size, padding and launch cost, which section 5.6 measures.
- Tensor parallel sizes above 1 and the TP4 rows of the epic: not measurable on a one-card pod. For those the program count is the arithmetic (48 attention programs to 8 at any context), not a time.
- The vision towers and the encoder path: out of scope here, covered by #1523, #1525 and #1473 on the same mechanism (section 7).

## 7. What has to land in torch-spyre first

1. The stack (#5912, #5917, the fourth part) and #5179 merged, so the plugin's pin can carry the mechanism.
2. A dynamic dimension on a tensor that is also moved with a custom device layout. The batched kernel's chunk page index is moved with `device_layout=` (the plugin's chunk-major layout, so the kernel can gather one chunk's pages at a time), and #5179's `.to()` refuses `dynamic=` and `device_layout=` in the same call. The measurements above used a small Python-side relaxation of that check in the build; the relaxation has to become part of torch-spyre.
3. Two runtime changes found by this work, both recorded for the runtime owner: a partial copy into a dynamic buffer at the speed of a whole copy (then the padding of section 4.3 becomes unnecessary), and `resize_` of a dynamic buffer that carries a custom device layout.
4. For the encoder follow-ons only: the dynamic dimension carried across graph outputs, so a chain of per-block graphs keeps it (asked on torch-spyre #43 on 4 November; implemented with tests on the cross-kernel branch, not yet in a pull request). The decoder path in this RFC does not need it, since it marks only per-step tensors that enter a graph and never leave it.

## 8. Alternatives considered

A denser bucket list (#771) adds programs and boot time where this removes them. A populated compile cache (#959) helps restarts, not the first boot of a new configuration, and the dynamic-shape programs make the cache smaller too. Emitting static loops instead of unrolled programs was tested and found equal, so there is no codegen change to make.

## Appendix: the seven questions of #1515

1. **Mechanism.** `tensor.to("spyre", dynamic={dim: {"min", "max", "granularity"}})` on the host tensor, nothing in the `torch.compile` call (`dynamic=False` stays), four `torch._check` statements in the kernel stating the range and tying the two dynamic tensors. Dim 0 only, `max / granularity` at most 32, the dimension must be a loop's tile axis.
2. **One compile for many buckets.** Yes. One Inductor compile at the maximum per kernel per query bucket, the other buckets are not compiled at all. 25 attention programs to 5 at 2K, 48 to 8 at 4096 with 32 sequences, 45 to 5 at 32K.
3. **Correctness.** Output tokens identical to today's plugin at every size tried, 2K to 32K, batch and single, greedy with logprobs (largest logprob gap 0.0075), on granite-3.3-8b and five more checkpoints. The kernels are bit-exact against the static ones on the device probe at 2 and 8 pages per iteration.
4. **Run-time cost.** None in the launch: the dynamic-shape and the static-shape program at the same page count launch in the same time within 1 percent. The cost was in the per-step copy below the maximum, removed by padding before the copy. The gain is fewer pages walked: 7 to 18 percent at long context, 13 to 20 percent a decode step at full batch at 4096, a wash at short context and under the serve load.
5. **Changes in spyre-inference.** The bucketer (the page range, the chunking, one warm-up compile per kernel), the metadata builder (the real count instead of a bucket, two dynamic transfers, no host slice of a dynamic tensor), the head-major backend (the dynamic chunk index and mask), the three kernels (the range checks and the tile walk), nothing in the `torch.compile` calls, the warm-up and the guard unchanged, about 980 patch lines behind switches. The body buckets keep their own compiles.
6. **Compile cache.** The dynamic-shape programs hit the kernel cache across restarts (engine start 15 to 16 s against 25 to 27 s at 2K). Across TP ranks not measured, one card.
7. **Gaps.** The 32-iteration cap at long context, `dynamic=` with `device_layout=`, the slow partial copy, the `resize_` under a custom layout, and the body's token axis, which is not a loop's business.
