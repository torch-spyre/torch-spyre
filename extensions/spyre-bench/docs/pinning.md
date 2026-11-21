# CPU pinning: guide

Pinning a benchmark to the CPUs near its accelerators can change host-bound latency (decode steps, launch overhead).
This guide shows how to request it, how the engine applies and checks it, how to read the effect table, and what
`unknown` and `suspect` mean. Every value in the examples is synthetic.

## For users

### Ask for a policy

| Policy | What the engine does |
|---|---|
| `none` | nothing; the command keeps the CPUs it is given |
| `device_local` | runs the command's process tree on the CPUs of the NUMA node(s) the devices hang off |
| `per_device` | each device gets its own CPUs: `assign` (generated from the host topology), `cpus_by_device` (lists you give), or neither (the stack under test pins itself and the engine only checks) |
| `explicit` | runs the tree on the CPU list in `cpus` |
| `platform` | the serving stack pins its own workers in-process; the engine launches nothing, records the policy and the observed `cpu_set`, and never judges it (`pinning_verified: unknown`, `cpu_set_source: platform`); `require_verified` is not valid with it |

Three ways to say it; the first one wins over the next for one plan:

```bash
# 1. flags (env twins BENCH_CPU_PINNING, BENCH_CPU_SET, BENCH_PIN_OPTS, BENCH_REQUIRE_PINNING)
spyre-bench plan ... --cpu-pinning per_device --pin-opt cores_per_device=4 --pin-opt spread=spread --require-pinning
# 2. a request file:  cpu_pinning: {policy: device_local, require_verified: true}
# 3. the Environment: spec.host.cpu_pinning (see examples/pinning-demo)
```

`per_device` options (`assign`): `cores_per_device` (physical cores), `spread` (`pack`: the device's own node;
`spread`: an equal share of every node; `round_robin`: own node, interleaved), `smt` (`primary_only`, `siblings`,
`both`), `reserve` (cores kept back per node), `reserve_cpus`, `spread_scope` (`all` or `socket`), `llc_align`.
`layout: ranks` (torchrun-style, one process per rank) pins each rank to its own list; `launcher: numactl` also binds
memory to the lists' nodes. Preview the lists without running anything:

```bash
python -m spyre_bench.pinning_plan --topology topo.json --devices 0:0,1:1 --cores-per-device 4 --spread spread --layout ranks
```

**One process tree** (the default layout) runs behind `taskset -c <cpus>`. **One process per rank** needs the command
to start the ranks through `{pin_wrapper}`, a generated script that pins by `LOCAL_RANK` and then runs the rest of the
argv (a pass-through for every policy that is not `per_device` + `layout: ranks`, so one command serves all policies):

```yaml
command: ["torchrun", "--standalone", "--nproc-per-node=2", "--no-python", "{pin_wrapper}", "{python}", "bench.py"]
```

The devices' NUMA nodes come from the host (the probe's `env.fp.device.numa_node`, one per device in order), never from
a scheduler's resource slice. Without them `device_local` and generated `per_device` cannot be resolved.

### What gets recorded

Every repeat records `run.exec.cpu_pinning` (the policy; `none` too), `run.exec.cpu_set_source`, and, for a pinned
request, `run.exec.pinning_verified`, `run.exec.cpu_set` and `run.exec.pinning_violations`. The derived label
`env.label.cpu_pinning` keeps pinned and unpinned runs in separate series. Variants of one policy (4 versus 8 cores,
pack versus spread) share that label: add your own label (`labels: {cpu_layout: spread-4}`) and list it in the
Project's `series_labels` if both must be tracked over time.

### Reading `unknown` and `suspect`

After the command starts, the engine samples the process tree every half second and compares each thread's CPU
affinity with the request. The worst repeat wins (`false` over `unknown` over `true`).

| `pinning_verified` | Meaning |
|---|---|
| `true` | every thread seen was inside the requested set (per rank, for `layout: ranks`) |
| `false` | a thread was outside it; `pinning_violations` counts them |
| `unknown` | it could not be judged: not Linux, an unreadable `/proc` entry, a command that ended before the first sample, a request that could not be resolved on this host, or the launcher tool missing |

With `require_verified`, `false` gives validity `suspect` (reason `pinning_violated`) and `unknown` gives `suspect`
(`pinning_unverified`); the verdict is `suspected`, never `regressed`. Without it the facts are still recorded and the
Result keeps its validity. A pinned result that is `unknown` says nothing about the effect: do not read it as "no
effect". Limits: only affinity masks are compared (a narrower container cpuset or a CPU quota is invisible), and an
orphan outside the command's process group, session and cgroup is not seen. A command shorter than about a second may
finish before it is sampled; keep benchmark commands alive for at least a few samples (warmup does this naturally).

### Stacks that pin themselves: `platform`

A serving stack may pin its own workers (each to the CPUs of its device's NUMA node, at start-up, unless the process
is already confined to fewer CPUs) and offer no switch to turn that off. After that change `none` no longer means
"unpinned": the same benchmark on the same node would measure the pinned behaviour under an unpinned label, and a
series would step by the effect size on the day the stack changes. Request `platform` for a stack version that pins
itself. The derived series label `env.label.cpu_pinning` then reads `platform`, which is a different series from
`none`: history before the change stays `none`, runs after it are `platform`, and neither is judged against the other
(reuse and baselines refuse a stored row with another pinning label). The engine cannot see inside the stack's
process start-up, so `platform` is recorded, never verified; the observed `cpu_set` shows what the tree was allowed on.

Rules of thumb: pick the policy per stack version (the Environment or the request names it; do not infer it); keep the
`none` history running only for stack versions that do not pin; to measure the stack's pinning against the engine's,
request `explicit`/`device_local`/generated `per_device` (a stack that leaves an already-confined process alone keeps the
engine's choice). `per_device` with no `assign` and no `cpus_by_device` is a different thing: the engine verifies that
each thread sits inside one device's node, which is only meaningful when the stack is known to do exactly that.

A benchmark command that runs `taskset` or `numactl` itself is not the engine's launcher: its processes are judged by
their masks like any other, so it is `true` when it narrows inside the requested set, `false` when it leaves it. To
state that the command pins itself, request `platform`.

### Measuring the effect

`examples/pinning-demo` is the runnable pattern: a latency Benchmark at two output lengths (`decode-probe`, plus
`decode-probe-tp2` behind `{pin_wrapper}`), Environments with and without pinning, a stub command, and the
`pinning-effect` Analysis.

```bash
cd extensions/spyre-bench
spyre-bench plan --catalog examples/pinning-demo --purpose on-demand --arch x86_64 --devices spyre/pf=2 \
   --subject-id <uuid> --cpu-pinning none --out none.json
spyre-bench plan ...same... --cpu-pinning device_local --require-pinning --out local.json
spyre-bench run --plan none.json  --catalog examples/pinning-demo --workdir examples/pinning-demo --env-file env.json --out out-none
spyre-bench run --plan local.json --catalog examples/pinning-demo --workdir examples/pinning-demo --env-file env.json --out out-local
spyre-bench analyze pinning-effect --catalog examples/pinning-demo --results out-none --results out-local
```

The table has one row per pinning policy within each node class: `n`, `mean`, `median`, `cv_pct`, `effect_pct`
against `none`, `within_noise` and `reading` (`better`/`worse`). `effect_pct` is the change of the per-token decode cost
(slope between the two output lengths, per repeat pair); `within_noise: true` means the effect is no larger than the
larger CV of the two cells, so treat it as no effect; `null` means fewer than two repeats. A cell holds one Result: keep
nodes apart with `env.node` in `across`, and compare one policy variant per analysis (pass only its `--results`).

### The measurement matrix

`scripts/bench-pinning-matrix` runs the standard experiment for one node class: one small model, TP1 and TP2, policies
`none`, `device_local`, `per_device`, 10 repeats each, then applies the acceptance rule and prints a table:

> a policy is worth adopting for the node class when its median per-token decode improves on `none` by more than
> `max(CV of both cells, 3 %)`, on at least 80 % of the repeat pairs, with no first-token regression beyond noise;
> otherwise record "no effect". Advisory: nothing gates, and the result should repeat on a second night.

```bash
scripts/bench-pinning-matrix --catalog benchmarks --out results/pinning --subject-id <uuid> --devices accel/a=2 \
   --env-file env.json --suite tp1=<tp1 suite> --suite tp2=<tp2 suite> \
   --decode-cases "tp1=<long case>,<short case>" --decode-cases "tp2=<long case>,<short case>" \
   --pin-opt per_device:cores_per_device=4 --pin-opt per_device@tp2:layout=ranks --policies none,device_local,per_device
```

Run it once per node class, with thread counts fixed and recorded (`OMP_NUM_THREADS`, `SPYRE_NUM_CPUS` as run
environment), a warm compile cache, and the policies on one node in one sitting. `--dry-run` prints the commands.
Product repositories wrap it in their own make target; real model names, node classes and measured values belong in
internal notes, not in this repository.

## For developers

| Piece | Where |
|---|---|
| plan generator (topology + devices + policy -> CPU lists, launcher prefix, rank wrapper), stdlib only | `spyre_bench/pinning_plan.py` |
| launch glue: resolve the request on this host, find the tool, wrap the argv, write `{pin_wrapper}`, fall back to `unresolved` | `spyre_bench/pinning_run.py` |
| observer: affinity walk, per-device and per-rank checks, launcher-stage skip | `spyre_bench/placement.py` |
| request-level override (`--cpu-pinning`) and validation | `spyre_bench/plan.py`, `engine.pin_problems`, `engine.overlay_pinning` |
| recording and the per-repeat samples | `spyre_bench/run.py` (`run_case`, `execute_command`) |

Flow of one case: `plan` puts the effective `host.cpu_pinning` in the plan record and the label in `labels`; `run`
calls `pinning_run.prepare` once (topology from `/sys`, device nodes from the env file, `resolve`), wraps each repeat's
argv, spawns it in its own session, records the root's `/proc` start time, and samples `verify_pinning` until it exits.
Samples taken while the tree is only the launcher (`taskset`, `numactl` or the wrapper script before its `exec`) are
skipped, so the transient unpinned moment is never a violation. The observer is independent of the launcher: for
`device_local` it re-resolves the node CPUs from sysfs; for lists it uses the request. `layout: tree` verifies against
the union (that is what was applied); `layout: ranks` verifies each process against the list of its `LOCAL_RANK`, and a
process without a rank (the launcher agent) against the union.

Tests: `tests/test_pinning.py` (fake `/proc` and `/sys`, golden launcher commands and wrapper, per-device and per-rank
verification). The `EndToEnd` class pins real processes with `taskset` and needs Linux; it describes a two-node
topology with a fake sysfs through the hidden `--sys-root` flag of `run` (`BENCH_SYS_ROOT`). Run it in a Linux
container when developing on another OS:

```bash
podman run --rm -v "$PWD":/src -w /src python:3.12-slim bash -c 'pip install -q pyyaml regex jsonschema pytest && python -m pytest -q tests/test_pinning.py'
```

Adding a policy knob: extend `Policy`/`ASSIGN_KEYS`, `gen_schemas.py` (then `spyre-bench gen-schemas`),
`engine.pin_problems`, and the golden tests; keep `pinning_plan.py` free of third-party imports.
