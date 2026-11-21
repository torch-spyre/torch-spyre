# moe-demo

Sample MoE benchmarks with placeholder names and synthetic shapes; no results. Validate with
`spyre-bench validate examples/moe-demo --central examples/central`.

- `ops/moe-expert-loop-sweep.yaml`: a kernel sweep over expert concurrency and tile size.
- `ops/moe-expert-loop-int8.yaml`: an int8 case and an int32 index case. Each variant states its own `flags`; flags are
  identity, so the two cases differ in op, dtype and flags.
- `systems.yaml`: the shared System. Design parameters (`experts`, `top_k`, `phase`, `tile_tokens`,
  `expert_concurrency`) are workload `flags`, which are identity by default.
- `serve/moe-serve-e2e.yaml`: an end-to-end serving case; the optimization under test is one System flag, so the two
  sides of an A/B differ in one field.

The MoE metrics `kernel.moe.fetch_bytes` and `kernel.moe.loop_trips` are declared in the kernel pack. A sidecar that
writes them needs the op harness, which is not part of this example. The trace signals that explain a run are named by
the `role.moe.*` run properties; a role with no stored signal reads as "not captured", never as zero.

## Limitations

- Nothing in this tree produces `kernel.moe.*` values or the `role.moe.*` signals: the runner module the kernel pack names
  (`spyre_bench.kernel.bench_runner`) is not present here. The metrics and roles are declared so that a producer can
  report them; whether a given harness does is not established by this example.
- There is no typed `parameters:` block yet, so the parameters are untyped `flags` and a flag that should not change the
  benchmark id cannot be marked as such. A typed block with defaults omitted from identity is a separate change.
- A sweep has no series id: the cases of one matrix are separate benchmarks.
- An Analysis cannot `vary` over `case.flags.<name>`: only registered attributes can be held or varied, and flags are
  not registered, so the curve over `expert_concurrency` cannot be expressed as an Analysis yet.
- `df16` is the device-side name of `fp16`; write `fp16` in a workload `dtype`. `int8` and `int32` are accepted.
