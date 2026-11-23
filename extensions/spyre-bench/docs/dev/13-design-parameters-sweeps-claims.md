# 13. Design parameters, sweeps, the claim record and the experiment flow

An optimization changes a design parameter (a tile size, a degree of concurrency, a data type) and claims that a number
got better. This guide says how to describe the parameter, how to sweep it, what to record so the claim can be
checked, and the order in which a claim is tested. It uses made-up names; nothing here is a measurement.

## Design parameters

A **design parameter** is a value of the design under study: a lever setting (`tile_tokens`, `expert_concurrency`) or a
structural limit (a loop bound). It is not a workload facet. A workload facet describes the load or the system under
test (`input_len`, `tp`) and is already part of what a case asks for. A design parameter is asserted with a
`value_kind` (`config`, `limit`, `derived` or `tested`) and a `source` (a symbol, a file or a run id the value comes from).

Name the thing you mean. In mixture-of-experts work the word "group" has been used for five different things, so none of
them is called that:

| Term | Meaning | Name |
|---|---|---|
| tile | tokens per per-expert compute unit | `tile_tokens` |
| expert concurrency | experts executing at the same time inside one device | `expert_concurrency` |
| expert parallel | experts split across devices | `expert_parallel` |
| batch size | sequences batched together | `batch_size` |
| chunk | tokens in one prefill chunk | `chunk_tokens` |

Where a parameter lives today:

- Identity or property. A parameter that changes what is measured is a different benchmark, so it is part of the case
  identity. A value derived from others (a loop bound) or measured at run time (a trip count, how tokens were routed)
  is a property or a measurement and never enters the hash.
- Workload `flags`. A kernel `op` workload has no typed parameter block yet, so a design parameter is written under
  `flags`, which are tool flags passed verbatim. Every flag is identity unless the workload type lists it as non-identity.
  Adding a new flag therefore starts new series for the cases that carry it.
- Metric context. Declare the parameter as a facet of the measurement that used it, so the number carries the setting it
  was taken at (see the metric-context vocabulary and its `design_parameter` class).

A sample is `examples/moe-demo` in the engine: placeholder names, synthetic shapes, no results.

## Sweeps

A sweep varies one or two parameters over declared values and keeps everything else fixed. In a Benchmark it is a
`matrix` axis, either a list of values or an `axis_sweep` (`path`, then `values` or a `range`).

Rules for a sweep:

1. Every case of the matrix is a separate benchmark with its own id. The engine does not group them into a series.
2. Compare along the axis as a curve of (parameter value, metric). A curve is not a verdict: a verdict is allowed only
   between two cases that differ in nothing except what the author called the subject (a build, one flag).
3. Hold the Environment constant across the sweep, and take at least three repeats per case.
4. An `Analysis` can `hold` and `vary` registered attributes only. A design parameter that lives in `flags` is not a
   registered attribute, so the curve over it is read from the cases directly until the parameter has a typed home.

## The claim record

A quoted improvement is a record, not a sentence. "About half better" is not checkable; the record below is. Record all
of it, or show the claim with the missing items marked as unknown:

| Item | Why |
|---|---|
| metric and statistic, sample count and spread | a mean of one run and a median of many are different claims |
| baseline and candidate: artifact ids and the parameter values of both sides | the difference must be the intended one |
| Environment id and whether the run was clean | a noisy neighbour or another host class can move a number |
| cold or warm, load, sequence lengths, data type, model, device and card count | the metric context of the number |
| distance to the floor on both sides | an improvement above the maximum possible gain points at a bad baseline or a bad model |
| the hypothesis ledger id | links the claim to the question that was asked |

The maximum possible gain is one minus the ratio of the floor to the baseline. If the claimed gain is larger, the claim
is flagged rather than celebrated. If the baseline is already near the floor, a large gain is implausible; if it is far
from the floor, there is headroom.

An incomplete claim is still shown, with the missing facets visible, and it is not used to rank or to decide.

## Experiment flow

From a claim to a verdict, in this order:

1. Write the hypothesis: statement, subject (model, phase, op), the lever, `value_kind: tested`, and the source.
2. Predict: evaluate the lever in the cost model. A lever without a calibration is shown as a labelled range, not as a
   point.
3. Measure: an A/B of two artifacts or two flags on the same Environment, a sweep for the curve, at least three repeats,
   purpose `investigate`.
4. Compare from the end-to-end number down to phase, layer and kernel, showing both sides' distance to the floor.
5. Explain: per-expert timeline, cores in use, loop trips and tokens routed, memory placement.
6. Conclude: `confirmed` (explained and reproduced), `rejected`, or `unexplained`.

`unexplained` is the state of a reproduced gain that no test explains. It ends only with a recorded test result, and it
is never a headline. The tests that move a claim out of it:

| Test | Rules out |
|---|---|
| rerun on another card and another node | one bad card or memory region |
| memory placement sweep | baseline weights in a slow region |
| baseline distance to the floor, both sides | a bad baseline, a gain above the bound |
| clean run on both sides | a noisy neighbour |
| statistic and sample count, spread smaller than the gain | noise |
| build identity: only the intended difference | a second change in the build |
| routing realism: real or trace routing against uniform | an unrepresentative expert selection |
| per-expert timeline overlap | whether the gain really is concurrency |

What the engine does today is steps 3 and the identity rules; the ledger, the floor and the explanation views belong to
the Perf Lab, and this page records the order they are used in.
