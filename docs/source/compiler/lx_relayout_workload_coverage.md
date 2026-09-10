# LX relayout workload coverage

## Purpose

This document is the workload contract for LX relayouts in Torch-Spyre and for their translation to KTIR. Each producer-to-consumer boundary below must remain expressible and testable as the compiler moves from SuperDSC to KTIR. Operation names alone do not establish coverage: the correct values must reach the correct cores, in the layout the next operation reads, without an unnecessary trip through HBM.

For example, our four-way V case starts with four different pieces on cores 0, 1, 2, 3. Cores 0, 8, 16, 24 each need all four pieces. Loading the complete tensor from HBM at one leader would not test that starting point.

This is a requirements inventory, not a passing KTIR test suite. It covers every record in the inspected Granite inventory, every E0–E10 boundary in the Gemma expert-body ledger, and the known paged-attention boundaries. Missing whole-model captures are explicit, not silently counted as covered.

The KTIR discussion is tracked in [ktir-mlir-frontend PR #53](https://github.com/torch-spyre/ktir-mlir-frontend/pull/53). Proposed operation names below are possible expressions, not tested syntax or requirements to use one particular lowering. The complete normalized Granite record list is in [Granite LX relayout inventory](lx_relayout_granite_inventory.md).

Historical inputs are pinned where they are public:

- [Granite ownership catalog](https://github.com/AdnanHoque/torch-spyre/blob/36804f23ede70325c21ba234a7e102537a8eda95/experiments/granite_relayout/artifacts/catalogs/sendnn_sdsc_lx_replay_manifest.json).
- [Granite P01–P14 families](https://github.com/AdnanHoque/torch-spyre/blob/36804f23ede70325c21ba234a7e102537a8eda95/experiments/granite_relayout/artifacts/catalogs/prefill_relayout_templates.json).
- [Gemma E0–E10 geometry ledger](https://github.com/AdnanHoque/torch-spyre/blob/4f5e1542add393051dae1ccd6526a751824f3e95/docs/moe/GEMMA4_RELAYOUT_GEOMETRY_LEDGER.md).

These are historical requirements and evidence pointers, not results for current Torch-Spyre or KTIR heads.

## 1. What counts as passing

Each edge needs three separate answers. A successful earlier stage does not imply the later ones pass.

| Stage | Required answer |
|---|---|
| Expression | A small KTIR example names the original source owners, the destination owners, the values each supplies, and their positions in the result. It passes the relevant verifier. |
| Execution | The lowered program delivers those values, and the real next operation reads them correctly. Check the emitted memory accesses as well as numerical output. |
| Application | The unchanged application selects this path, stays correct, and removes the identified spill. Measure performance separately if making a speed claim. |

All proposed KTIR examples and execution results in this inventory are **pending**. Saved Torch-Spyre/SenDNN artifacts provide examples to translate; they are not KTIR evidence or claims about today's PR heads.

### One reusable test contract

For every case, attach:

1. Logical shape, element type, physical layout, source and destination core IDs, and exactly which logical coordinates each core holds. Include offsets, padding, and physical addresses needed to interpret the saved layout.
2. Whether source values are independent tensor elements, unfinished sums, or already-completed sums. Equal coordinates do not make unfinished sums interchangeable.
3. The producing operation, consuming operation, schedule, and source/version identifiers. State whether the schedule was forced or selected normally.
4. The expected destination values, an independent reference, and the KTIR spelling. Specify local slice order, not just the set of values.
5. Emitted evidence of the source tier, destination tier, per-core address bounds, and synchronization. For an LX-to-LX test, no intermediate HBM write/reload is allowed. Initial cache/weight reads and final output writes are separate, legitimate edges.
6. Numerical checks after the actual consumer, including repeat-after-poison tests, plus an explicit verdict for expression, execution, and application separately.

The copy-only reference is simple: for each destination coordinate, find that same coordinate in the source ownership table and copy its value into the destination's specified local position. A coordinate may be delivered to several consumers. A consumer must not accidentally receive it twice or miss it.

Do not derive expected results through the KTIR helper under test. Use the original coordinate tables. For large tensors in a low-precision format, test coordinate IDs one bit at a time using exactly representable 0/1 values; a large floating-point ramp can round distinct IDs to the same value. Small worked examples below use exact small integers.

Reduction tests have a different reference: sum the independent contributions once. If the producing matmul has already completed that sum, the following copy must not sum it again.

### Failure checks shared by the cases

- Reverse two fragments: the test must catch the wrong order, even if shapes and element counts match.
- Omit a fragment or use the wrong batch/head/expert: the test must fail.
- Reuse a wrong core map with identical split counts: the test must fail.
- Fill unrelated LX storage differently before two calls: valid outputs must remain unchanged.
- Preserve buffers until their last reader finishes; check that overlapping addresses never hold different simultaneously live values.
- Report an unsupported expression or lowering explicitly. HBM fallback may keep an application correct, but does not pass an LX-transport requirement.

## 2. Granite: all captured prefill and decode relayouts

The pinned Granite ownership catalog contains **55 prefill records and 75 decode records**. These are 130 saved static relayout records, not 130 different communication patterns and not the proposal's separate 51-file inventory. Repeated-layer counts in the manifest do not create new geometries, but their consumer identities and operand numbers must remain traceable.

Create a parameterized test for **every** catalog record. The companion index gives each record a stable test ID and its exact JSON array index. Its full `source_pieces`, `destination_pieces`, and `owners` arrays are the test inputs. Do not replace them with a reconstruction from split counts or the manifest's descriptive `route_class` string.

Read-only preparation checks found no uncovered destination rectangles, overlapping distinct source/destination rectangles, or repeated owner IDs within a piece in these 130 records. That checks consistency of the saved inputs, not KTIR support.

### Prefill families P01–P14

These IDs come from the pinned P01–P14 family catalog. They group the prefill records for discussion; the 55 individual records remain required. Coordinates and core IDs come from the pinned Granite catalogs, including layouts with nonconsecutive active cores.

| Family | Boundary and concrete ownership change | Required result / check |
|---|---|---|
| P01 | K input to attention QK: 32 pieces of `[x=8, out=512, in=128]` become one complete region on 32 cores. | Gather the pieces and broadcast the complete region. Run the QK consumer; do not substitute a full K reload. |
| P02 | V input to attention PV: 32 pieces of `[x=8, in=512, out=128]` become one complete region on 32 cores. | Gather + broadcast, followed by PV. Keep this separate from P01 because the axes read by the matmul differ. |
| P03 | Normalized input to gate/up projections: each of 8 token groups has four 1024-column pieces of a 4096-column input; four consumers need the full group. | Gather hidden columns and broadcast to the four output-column consumers. Exercise all recorded gate/up operands. |
| P04 | Attention output to output projection: four 16-token pieces become a 64-token region on four projection cores. | Gather token rows + broadcast. This is not the same concatenation axis as P03. |
| P05 | Normalization input: four 1024-column pieces become a full 4096-column row region on one owner per token group. | Gather before the recorded normalization-statistic consumer. The transfer copies input values; it is not itself the normalization reduction. |
| P06 | Query/rotary output: `[token:8, head:4]` ownership becomes `[token:32]`. | Each destination receives a smaller token interval from all four head pieces. Split one axis and gather another; preserve head and token positions. |
| P07 | Rotary multiply operand: 16 source regions become 8 regions, each used on four cores. | Two-piece gather + four-way broadcast to the recorded pointwise input. Test both multiply operand variants. |
| P08 | K/head-layout transition before local restickify: 32 pieces remain 32 pieces but move to different owners. | Move each piece to its exact new core, then verify the restickify's read and output layout. Do not replace this edge with a post-attention transfer. |
| P09 | Initial normalization to Q, K, V projections: 16 token regions on even-numbered cores become 8 regions used by four cores each. | Gather pairs of row pieces + broadcast. Retain the sparse source IDs and test all three projection consumers. |
| P10 | Normalization operand 1: 8 regions become the same 8 regions on four owners each. | Broadcast to the recorded normalization operand. Preserve its physical scalar/padded representation. |
| P11 | Normalization operand 2: same count pattern as P10, a different input. | Test separately; swapping the two inputs must not pass. |
| P12 | Residual add: 16 full-width, 32-token pieces become 32 pieces with 64 tokens and 1024 columns. | Split columns while gathering two row pieces. Verify the actual add and the unchanged other operand. |
| P13 | Final hidden vector to LM head: 32 pieces of 128 values become the complete 4096-value vector on cores 0–27. | Gather + broadcast to 28 consumers, not 32. No false requirement that producer and consumer counts match. |
| P14 | Last-token selection: source `[512,4096]`, 8 token groups × 4 column groups; destination is row 511, split into 32 pieces. | Select on the cores that own row 511, then send the right column pieces. No movement of the discarded 511 rows; no assumption that the selected row is already on all destination cores. |

### Decode additions that prefill does not cover

All 75 decode records are required, not a rescaled prefill stand-in. In particular:

| Saved consumer family | Additional case to retain |
|---|---|
| Projection BMMs | Single-token input gathered from 32 owners; down-projection input gathered from **25** owners for width 12800; output-owner sets must come from the records. |
| QK and PV BMMs | Different source/destination head and sequence partitions, including a saved sequence extent of **768**. This is not the paged 513-token fixture. |
| Normalization statistics / normalization | Full-width gathers and single-source broadcasts; both normalization operands. |
| Softmax Max and Sum | Gather score pieces before the real max/sum consumer. Verify that the copy and the arithmetic reduction are not confused. |
| Softmax Sub and Mul | Broadcast per-row statistics back to the score/probability owners. Preserve which statistic belongs to which head. |
| Rotary Mul | Two source pieces assembled and supplied to the recorded receiving cores, with both input roles retained. |
| Restickify | Cross-core owner change followed by local layout conversion. Check the consumer's physical read. |
| KV-cache Scatter | The recorded relayout feeds a cache update. Copy the newly computed values correctly, then write only the indexed cache positions. A cache-update operation named Scatter is not automatically the same as an inter-core split-and-send operation. |
| Embedding Stcdp | One source region split into 32 destination regions. Preserve the saved source owner, rather than treating it as 32 existing sources. |

### Torch-specific Granite sites: do not lose them in the SenDNN list

A historical Torch replay identified two additional boundaries that are not present in P01–P14. Treat its source/schedule as historical, not evidence for current branches.

| ID | Required example | Evidence / remaining input |
|---|---|---|
| GR-X01 | Post-attention output: token32 → token8 × head4. Split and reassemble into the next identity/view consumer. | Historical layer `41_shuffle` → `42_identity`; recover the referenced full bundle before claiming an exact KTIR translation. It is not P08. |
| GR-X02 | MLP product: token32 → token16, full input region shared by cores `m` and `m+16`, then the down projection. | Historical layer `62_shuffle` → `63_batchmatmul`; recover the full bundle. This is absent from the literal SenDNN P01–P14 inventory. |

The historical replay also names transformed rotary-producer completion transfers without fully enumerating them. Register these as **GR-X03: capture/index required**. The P06 consumer transfer alone must not be taken as coverage of those preceding completion steps.

## 3. Gemma: every routed-expert boundary

The captured dense expert body uses `T=512`, `H=2816`, `F=704`, `E=128`; a padded alternative uses `F=768`. `M8×H4` means eight groups of token rows and four groups of hidden columns. Count the actual owners, not just these split labels.

There are three distinct schedules to retain:

- **Common-row control:** every operation splits token rows 32 ways; no inter-core relayout is needed between them.
- **Captured reduction-split chain:** gate/up split token rows 8 ways and the summed-over input 4 ways; their completed outputs feed **M32** activation/multiply; hidden is gathered for an M8×N4 down projection. Saved reduction-chain and layer captures document this chain.
- **Padded output-split chain:** at `F=768`, gate/up, activation, and multiply keep matching M8×F4 owners; hidden is gathered and broadcast for down. Historical isolated and common-order captures cover the relevant edges.

**Ledger correction for these tests:** the pinned ledger's S1 table mixes a K-split gate/up schedule with an X gather intended for N-split consumers, and names F4 pointwise consumers where the captured reduction-split composition uses M32. The saved composition wins. Do not require an unnecessary full-H gather for a K-split consumer that needs only its H shard; do not label the M32 completed-sum handoff as an F-axis split.

| ID / edge | Source → destination | Test requirement and expected answer |
|---|---|---|
| GM-E0 | HBM X → LX preheader | Load/stage X once in the chosen layout. Establish the actual physical owners before testing any later delivery. This initial HBM read is allowed. |
| GM-E1 | LX X → gate **and** up | For the padded N-split case, source core `4m+h` holds `X[64m:64m+64,704h:704h+704]`. All four cores `4m+n` need the full 2816 columns for those 64 rows. Gather + broadcast **once**, and let both matmuls read the delivered value. A saved shared-X capture is the historical example. For the K-split schedule, test the required H shards instead of inserting this gather. |
| GM-E2 | Gate matmul → activation | Padded case: same-owner read, no movement. Reduction-split case: combine the four contributions once, then distribute the completed 64-row region into the four 16-row M32 consumers. Never pass partial sums directly into activation. A saved reduction-chain capture documents it. |
| GM-E3 | Up matmul → gated multiply | Same completed-sum rule as E2, independently tested on up. Check the paired activation input too: correct up movement alone cannot prevent a mismatched multiply. A saved reduction-chain capture documents it. |
| GM-E4a | Hidden M8×F4 → down M8×N4 | At F768, four 192-column pieces per 64-row group become a full 768-column region at each of four down consumers. Gather + broadcast; test down's output. A saved isolated F768 capture documents it. |
| GM-E4b | Hidden M32 → down M8×N4 | At F704, cores `4m..4m+3` hold four consecutive 16-row full-F pieces. Gather those rows and broadcast the 64-row region to the four N consumers. A saved reduction-chain capture documents it. Different gather axis from E4a. |
| GM-E5 | Down → route multiply | Preferred: keep down's exact M8×H4 owners through the multiply, no copy. Also test a deliberately different consumer order: move the pieces correctly or reject that variant; never silently reinterpret them. A saved same-owner route-tail capture documents it. |
| GM-E6a | Route scalar M8 → M8×H4 | Source `m` owns 64 token scalars; destinations `4m..4m+3` all need those same scalars. Broadcast, then the **actual pointwise multiply**. The saved geometry passes through a matmul harness, while a direct pointwise probe fails; the latter is negative evidence, not a passing implementation. |
| GM-E6b | Route scalar M32 → M8×H4 | Gather four consecutive 16-row scalar pieces, then broadcast to four hidden-column consumers. Retain this separately from E6a. A historical route-scalar experiment contains this control. |
| GM-E6c | HBM route scalar → multiply | Correct direct-read alternative for a small scalar tensor. Must stay legal; no requirement to force a broadcast when it costs more. It does not count as passing E6a's LX path. |
| GM-E7 | Route multiply → expert contribution | Preserve token/hidden coordinates and the expert's route weight; same-owner read in the chosen chain. |
| GM-E8 | Contribution → accumulator add | Contribution and accumulator refer to the same token/hidden cells on each core. Sum each expert contribution once. |
| GM-E9 | Accumulator → next expert iteration | Same values and LX addresses survive across trips. With two experts, use different inputs/weights/routes so replacing, doubling, or exchanging contributions fails. Repeat with experts 0, 63, 127 at E128. Saved two-expert and 128-expert loop captures. |
| GM-E10 | Final accumulator → HBM result | One final write of the completed output. No per-expert activation spill or `[E,T,H]` intermediate. Clear accumulator state between launches. |
| GM-W | HBM expert weights → gate/up/down | Stream the correct expert's slices, advancing weights and route indices together. No requirement to hold all expert weights in LX. Test first/middle/last experts. |

### Two small Gemma tests that catch real errors

**Completed sum, then copy.** In one four-core row group let independent contributions be 1, 2, 4, 8 for every output cell. The completed value is 15. Test two starting contracts separately: (a) raw contributions that KTIR must reduce; (b) a producer that already completed the sum at its declared owner. In (b), the following delivery copies 15 and must not reduce again. Poison non-owner storage. The capture's terminal owners are `3,7,...,31`; that is this fixture's map, not a hardware-wide rule. Verify synchronization before reading them.

**Same splits, wrong order.** For T512/H2816, source core `4m+h` owns the 64-row × 704-column rectangle `(m,h)`. If destination order is `m+8h`, the corresponding block must move from `4m+h` to `m+8h`. If the next operation adopts the source order instead, zero-copy is correct. Reading the old addresses with the new interpretation is not. Use both variants; exercise the downstream multiply and accumulator.

The S0 control must remain correct without any movement. The complete chain test must compose E1–E10, not merely pass each edge in isolation. Preserve the chosen activation from each captured/application graph; a GELU micro-test is not proof for every model activation.

## 4. Paged attention: cache, K, scores, V, and carried output

These boundaries apply to both decode and prefill, but their schedules and storage needs differ. Do not infer one from the other.

### Shapes to keep distinct

| Case | Logical inputs | Scope |
|---|---|---|
| PA-D513 | B4, KV heads8, query heads/KV4, Qlen1, D128, page128, KV length513 | Five valid pages; use the recorded eight-position decode bucket and mask the unused positions. This shape comes from a historical unchanged-application capture. |
| PA-D1 | Same heads/dimensions, one page | Focused K/V mechanism tests using the saved one-page layouts; the saved four-way schedule is forced, not a production cost-model decision. |
| PA-P512 | One 512-query chunk; KV context includes the preceding chunks | Test QK and PV on each page, causal masking, and carried state. Exact production core maps require a pinned prefill capture; do not reuse decode maps by assertion. |
| PA-LONG | Final chunk of a 32768-token prefill, then decode at 32769 if the model permits it | Proposed boundary tests: 256 valid pages at 32768, 257 at 32769. Verify the selected implementation's bucket and skip/mask behavior rather than requiring an arbitrary iteration count. No fresh long-context evidence here. |
| PA-MIX | Three one-token decode requests plus a 509-token prefill chunk under a 512-token budget | Separate mixed-path compatibility test, not evidence for the decode-only kernel. Capture/map pending. |

For PA-D513 per page, Q is logically `[4,8,4,1,128]`; K is `[4,8,1,128,128]` in QK orientation; V is `[4,8,1,128,128]` in PV orientation. The last two sizes coincide, but their axes play different roles. Use nonsymmetric values so a transpose cannot pass accidentally.

| ID | Boundary | Required test |
|---|---|---|
| PA-00 | Newly computed K/V → persistent cache | Write only the selected slots/pages. Old cache entries remain unchanged. An HBM cache write is legitimate; this is not the temporary page spill we want to remove. |
| PA-01K / PA-01V | Page table + HBM cache → gathered page in LX | Runtime indices select the correct physical pages, independently for each request. Use nonconsecutive page IDs, shared read-only pages, different request lengths, and a partial final page. Initial HBM reads are allowed. |
| PA-02K | Gathered K fragments → K owners needed by QK | Translate the saved K source/destination descriptor and the official per-page K gathers. Derive exact logical pieces from the descriptor, not the name `mb`. Test every page position, not just page zero. |
| PA-03K | K owner → local K layout conversion | The saved local K layout-conversion descriptor has LX on both sides. Preserve all K coordinates while changing how they are stored for the matmul. Validate the real QK read. Do not judge residency from the string `ReStickifyOpHBM`; inspect memory declarations and accesses. |
| PA-04Q | Q producer → QK | Same-owner read or a required owner change according to the captured schedule. Keep Q valid across the page uses; test head grouping and batch separation. Exact current application capture still required. |
| PA-05 | QK scores → mask/max/exp/sum/normalization | Every query receives statistics from exactly its valid key positions. If scores or statistics change owners, translate those edges too. A passing K transfer alone does not cover them. |
| PA-06 | Normalized probabilities → PV | Deliver the right query/page probabilities to PV's owners. Test with nonuniform probabilities and distinct V rows; swapping pages must fail. Exact schedule capture required. |
| PA-07V | Gathered V fragments → PV | Test the saved simple V gather and four-way V gather + broadcast separately. Preserve original LX pieces; do not move page preparation into a leader HBM load. |
| PA-08 | Per-page PV result → combined output | Match the page-combination algorithm actually used: partial weighted outputs plus the corresponding normalization values. Simple addition of independently normalized page outputs is not a valid substitute. |
| PA-09 | Page-to-page state | The implementation's running max, sum, output, or stored per-page intermediates remain correct and live until consumed. Do not invent a new streaming algorithm to pass the migration test. For a streaming implementation, test stable LX state and bounded scratch across the loop. |
| PA-10 | Final attention output → projection/residual | Match the next operation's token/head/hidden ownership, including a reshape or inter-core move when required. Capture the real application edge; don't stop validation at attention output. |
| PA-11 | Block/query tails and padding | Poison invalid cache rows and padded positions. They must not affect valid outputs or cause invalid memory accesses. Repeat at lengths 127,128,129,512,513 and at the long-context boundary. |

### Exact four-way V delivery requirement

Use the saved four-way V physical descriptors for the production-shaped test. Its source split is `mb:32`; destination is `mb:8, qpk:4`. For each `g` from 0 to 7:

```text
producers:   4g, 4g+1, 4g+2, 4g+3
consumers:   g, g+8, g+16, g+24
result:      all four pieces, in their original logical order, at each consumer
```

Do not assume the descriptor's fused `mb` axis means only page-token position. The exact test obtains the axis interpretation from the saved shape/access mapping.

For a tiny test of the same core relationship, use logical X[8,8]. Source core `4g+r` holds `X[g,2r:2r+2]`. Set `X[g,j]=8g+j`. All four consumer cores `g+8q` must receive `[8g,8g+1,...,8g+7]` in that order. These integers are exactly representable. This toy test is derived from the core relationship; it is not a claim that X[8,8] is the production V shape.

Check both the default all-producer dependency and an explicitly written equivalent dependency. They should describe the same values. Shared sources across consumers are intentional. The consumers need not themselves be producers in that group. A single proposed `inter_tile_gather` with multiple receivers, or an equivalent on-chip sequence, is acceptable.

The saved four-way V case is **not** a full K+V-on-chip proof: its K layout conversion writes HBM. A separate local-K capture supplies a separate local-K example. Combining evidence from different schedules does not prove one composed kernel.

### Simple V broadcast variant

Add a small receiver-count test for a page already complete on its source core: two distinct source slices, each sent to its declared receiving group, up to 32 total consumers. Use different source values and validate all destination IDs. This is the 2→32 broadcast requirement associated with the original simple-V issue. The exact source-group split must be attached from that device test; this inventory does not invent it. This is **PA-07B: exact capture attachment pending**, separate from the four-source gather above.

## 5. Small cases that exercise the proposal's open choices

These supplement the workload cases; they do not replace the full captured shapes.

| ID | Input and expected answer | What the authors need to settle |
|---|---|---|
| KT-01 | Four-way V toy above, default and explicit equal dependencies. | R5 must not reject valid sharing merely because different receiving cores use the same producer pieces. This concern is about explicit dependencies; default gather already describes an all-producer assembly. |
| KT-02 | V group zero: producers 0–3, consumers 0,8,16,24. | Resolve §10.1 for copy deliveries: receiving-only cores are needed. No requirement to add dummy producers. |
| KT-03 | Source core0 holds `[2,3]`, core1 holds `[0,1]`; receiver needs `[0,1,2,3]`. | Show how logical order is preserved when it differs from core-ID order. Local selection/reordering is acceptable if fully expressed and stays on-chip. |
| KT-04 | Four 1×1 tiles of a 2×2 logical matrix: source0=(0,0), source1=(1,0), source2=(0,1), source3=(1,1). Receiver needs `[[0,1],[2,3]]` when values encode row-major coordinates. | Show multi-axis assembly, including exact result dimensions and placement. A flat concatenation `[0,2,1,3]` is wrong. Add the full multi-axis Granite shapes afterward. |
| KT-05 | Proposal's `[512,32,64]` select: source core `4m+h` owns rows `64m:64m+64`, sticks `8h:8h+8`. Destination core `d` needs row511, stick `d`. | Source `28+floor(d/8)` supplies destination `d`. Select locally, then split/send the selected row. The original layout is not a no-op; fresh HBM loads at destinations would change the starting contract. |
| KT-06 | Gemma row-major versus column-major core order, described above. | Equal split counts are insufficient. Preserve or explicitly change ownership. |
| KT-07 | Raw contributions 1,2,4,8 versus already-completed value15. | Distinguish reduction from delivery of a completed sum; declare completion owners and ordering. |
| KT-08 | A logical axis of 128 FP16 values stored as two 64-value sticks, plus a tail variant. | Logical slicing must agree with physical addresses. Do not split within an indivisible storage unit unless the backend explicitly supports that access. Check both valid factored views and wrong fused-axis descriptions. |
| KT-09 | Multiple consumers read one delivered X; then two expert/page iterations reuse scratch only after reads complete. | Demonstrate reuse and lifetime without making every arithmetic reader initiate another transfer. This does not require multiple delivery users of one future. |
| KT-10 | Remove one required piece; duplicate a piece inside one destination; corrupt an owner; supply a wrong output shape. | Deliberate bad variants must be rejected or fail the independent output check. Shared pieces across different destinations remain valid. |

For KT-05, let value(row,stick,lane) be identified by its coordinates. Core0's required `(511,0,:)` starts on source28. That one coordinate is enough to disprove a no-communication interpretation of the original source table. Test the proposal's 32-stick example and Granite P14's 4096-column example separately; they have the same relation but different widths.

## 6. Known gaps: do not claim all three workloads are covered yet

| ID | Missing evidence or unresolved case | Required next input |
|---|---|---|
| GAP-GR-APP | Historical SenDNN records do not inventory every current spyre-inference graph or datatype. | Pin current application/compiler/model versions; map every observed boundary to a listed test or add a new one. Include prefill and decode. |
| GAP-GR-X | Historical evidence points to remote bundles for Torch-specific sites and rotary completion. | Attach the full source/destination descriptors and final emitted payloads; do not treat the ledger description as a complete fixture. |
| GAP-GM-ATTN | Gemma attention, Q/K normalization, positional transforms, and cache paths are outside the pinned Gemma ledger. | Capture them through the HF-adapter path and map each boundary, including the factorized K/cache-write layout case. Do not assume Granite's head configuration applies. |
| GAP-GM-ROUTER | Router-logit calculation, top-k/expert selection, shared-expert path where present, and their joins are outside the pinned Gemma ledger. | Use the actual selected model/adapter graph; inventory these boundaries. Mark a feature not applicable only after checking that graph. |
| GAP-GM-DC | The pinned Gemma ledger is dense prefill expert-body evidence, not sparse decode evidence. | Capture actual decode routing and per-route/expert computation, token order, and output accumulation. Do not replace it with the 128-expert dense prefill loop. |
| GAP-GM-E6 | Exact route broadcast's pointwise consumer fails in the saved negative. | A passing KTIR broadcast followed by the real multiply, or an explicit supported HBM alternative with LX case still open. |
| GAP-GM-CHAIN | Some old artifacts came from uncommitted experimental integrations. | Preserve those as historical examples; obtain reproducible pinned-source composition evidence before migration sign-off. |
| GAP-PA | Prefill, mixed-path, all scores/probabilities/output boundaries, simple 2→32 source mapping, and long-context captures are incomplete in this inventory. | Attach exact application bundles. The PA rows are test requirements, not assertions that those precise schedules were already captured. |
| GAP-KTIR | Proposed copy-delivery operations do not yet have execution evidence here. | Authors supply KTIR examples, verifier tests, lowered programs, and numerical results. Existing indirect-access and memory-view facilities should be reused where sufficient. |

The criterion for “all edges” is closure of the inventory: every actual producer/consumer boundary in a pinned application capture maps to a passing case or an explicitly accepted fallback. Matching only the communication names, only the 14 families, or only one consumer class is not enough.

## 7. Handoff and order of work

1. Review these requirements with the KTIR authors and Torch-Spyre relayout maintainers. Start with KT-01/02 (V sharing and receiving-only cores), KT-05 (selection), and KT-07 (completed sums), because they decide semantics rather than implementation polish.
2. Supply the original pinned Granite catalogs and selected shareable descriptors from historical captures. Attach only shareable source ownership tables and descriptors. Keep model weights and captured activations out of public artifacts unless separately approved.
3. Have the authors write the smallest valid KTIR examples and identify whether each fits the current proposal, needs clarification, or needs a missing capability. No application rewrite or new ownership model is prescribed here.
4. Turn the agreed examples into verifier and execution tests. Run all 130 Granite records, both Gemma schedule variants and the full chain, and the distinct paged cases. Keep positive and deliberately wrong variants.
5. Close the GAP rows using unchanged spyre-inference and HF-adapter application captures. Compare correctness and emitted memory traffic before measuring speed. A faster isolated forced-schedule test is not a production application claim.

## 8. Maintenance

Add an edge when a pinned application capture exposes a producer/consumer boundary that is not already represented. Update its status only with the evidence required by Section 1. Keep historical, structural, numerical, application, and performance evidence separate. Never change a workload requirement merely to match the current implementation; record an explicit fallback or gap instead.
