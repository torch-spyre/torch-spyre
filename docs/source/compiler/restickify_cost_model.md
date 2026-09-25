# Restickify: DMA requests and transpose throughput

The [cost model](cost_model.md) prices proven DL16 transports using their physical
source access, including stick-axis swaps and staging copies. A core split can
make each input burst shorter without reducing total payload, so total cores and
aggregate bytes alone cannot rank these plans.

## Geometry and hardware limits

The extractor uses the input's device layout and per-invocation coordinates,
including stick-plane/element coordinates. It coalesces adjacent affine axes
from stride one outward. A split on an inner axis prevents coalescing its outer
neighbors. Physical padding does too. Unsupported non-affine accesses, missing
ownership and non-DL16 layouts retain the previous model.

Off-chip loads transfer 128-byte words, with a maximum burst of 32 words
(4096 bytes). These are hardware transfer limits, not calibrated tile dimensions.
For payload `P`, source run `R`, and `C` active cores:

```text
requests = P / clamp(R, 128, 4096)
byte_time = P / bw_restickify
transfer_time = byte_time + max(byte_time, requests * ns_per_request(C))
transpose_time = P / (C * per_core_transpose_bandwidth)
extra = max(0, max(transfer_time, transpose_time) - 2 * byte_time)
```

The existing model already charges balanced-copy bandwidth. Add only `extra`,
multiplied by the operation's loop trip count. Add it outside bundle compute
overlap: the swap uses the on-chip transpose pipeline and precedes its dependent
consumer.
This formula describes an HBM-to-HBM stick-axis swap. A plain copy or an HBM-to-LX
staging copy receives only the read-request excess over its existing byte charge,
without the transpose ceiling. An LX-resident input receives no off-chip request
charge. Symbolic residency preserves these rules while the solver chooses a plan.

Staging matters: the planner can see a copy feeding a transpose, while the final
program folds that copy into a direct off-chip read by the transpose. The copy's
division may differ from the consumer's, so charging the former does not price
the executed read. The objective uses a non-mutating view of the direct read
only when the existing address, ownership and loop checks prove copy removal
valid for **every candidate division**. It then prices the consumer's source
geometry and omits the removed copy. A failed proof or shared copy preserves the
original cost view. Sources eligible for input cloning and copies eligible for
an additional scratchpad shuffle also retain the original view, since these
later allocation choices can redirect the read. Allocation still plans the
original buffers; the late pass remains responsible for validating and performing
the actual rewrite.

The implementation folds payload into the request-count expression **before**
CP-SAT integerization. Keeping requests/byte as an intermediate can silently round
the entire request term to zero. Solver tests check actual choices and objective
values, not just the presence of split symbols.

## Calibration methodology

Calibrate the aggregate request interval as a function of active core count and
the per-core transpose throughput independently. These are empirical parameters,
not hardware transfer limits or preferred work divisions. A measured request-rate
plateau does not by itself establish a microarchitectural explanation.

Use controlled transport operations with explicit layouts and divisions, so the
planner does not choose the configurations used to calibrate its own model.
Check every configuration against a CPU reference and inspect the generated
transfers to verify that their burst lengths agree with the extracted source
runs. Keep device-event durations separate from synchronized host timings, and
report repeated measurements and their spread.

Vary the factors that distinguish the model's terms:

- Hold payload constant while changing source geometry and core division to
  isolate contiguous-run length from aggregate bytes.
- Vary payload independently to test scaling, including shapes outside the
  calibration set.
- Sweep low core counts and long input runs to expose the transpose throughput
  ceiling. Do not infer a request rate from a measurement dominated by that
  ceiling.
- Compare a single invocation with repeated invocations, and compare reuse of
  one input tile with advancing through a larger backing allocation. This
  distinguishes per-invocation costs from loop and reuse effects.
- Include a copy-only control with the same source access to separate input
  request costs from transpose costs.
- Validate source and destination residency cases independently. A coefficient
  fitted to an off-chip-to-off-chip transport does not establish the absolute
  latency of an off-chip-to-scratchpad transport.

Fit request-related increments without counting the existing byte-bandwidth
charge twice. Preserve conservative fallbacks where the measurements cannot
identify a parameter, and validate predicted rankings on held-out geometries
instead of introducing exact-size or preferred-split gates.

## Validation methodology

Deterministic tests should check geometry extraction, residency handling,
copy-removal proof failures, and the solver's selected divisions and objective
values. Hardware timing does not belong in deterministic unit-test assertions.

For end-to-end validation, compare full-model runs on the same hardware and
runtime environment. Hold model weights, dtype, batch size, prompt contents,
chunk size, generation settings, and host-thread settings constant. Include both
short and long sequences rather than extrapolating from one attention shape.

Use isolated compiler caches and record solver status and selected divisions.
Measure compilation separately from warm execution: complete warmup before
collecting repeated generation latencies, exclude loading and tokenization, and
collect profiles separately from unprofiled timing runs. Report the median and
spread, and check output consistency within and across variants. Repeated warm
runs measure runtime variation; independent compilations are needed to assess
plan stability.

Transport-level reference checks and matching generated outputs serve different
purposes. A matching generated token is a useful smoke check, not a general
model-accuracy guarantee; numerical validation should cover the affected
operations and representative model workloads.

## Scope and limitations

- No exact-size, repetition or preferred-split gate. Payload is the work visited
  per invocation, not the size of the KV backing allocation.
- DL16 source DMA calibration. Fully local transports and other device formats
  remain unchanged. HBM-to-LX uses the same source-request estimate, not a
  separate absolute-latency model.
- Non-affine/sub-stick accesses are not calibrated. Unsupported core counts are
  neutral rather than assigned an invented measured rate.
- Output burst fragmentation and interactions between independent operations in
  a fused bundle are not separately modelled by this term.
