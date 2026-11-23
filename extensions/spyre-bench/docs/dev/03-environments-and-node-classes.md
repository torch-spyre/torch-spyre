# 3. Environments and node classes

An **Environment** is a request: what a case needs from the machine. A **NodeClass** is policy: what a class of hosts
looks like and how long a qualification keeps one of them reference-grade. The two together decide where a result can
come from and how far it can be trusted. This guide builds `examples/docs-environment`:

```text
examples/docs-environment/
  project.yaml         Project: series labels include the node class and the CPU pinning policy
  node-classes.yaml    two NodeClasses: a qualified pool and a shared pool
  environments.yaml    five Environments: plain, pinned (two ways), clean, and multi-host
  ops/op-scale.yaml    a Benchmark that does not name an Environment: the request chooses one
```

## What an Environment is, and is not

An Environment is *never identity*: the same case run under two Environments is the same benchmark, and the
Environment name is a run attribute. It also never says *where* or *how to launch*: choosing a node, holding devices
and cleaning a node belong to the launcher that consumes the plan. The engine reads the request to plan (is there
capacity, is a reference node needed, is clean access requested) and to judge the result afterwards.

For that reason a Benchmark cannot enumerate Environments as cases. Pick the Environment per request:

```bash validated
spyre-bench plan --catalog examples/docs-environment --environment two-cards-local --purpose nightly \
    --select smoke --arch x86_64 --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan.json
```

## Devices and the host

```yaml validated in=docs-environment/environments.yaml
apiVersion: bench/v1alpha1
kind: Environment
metadata: {name: two-cards, description: "two cards of one metal-a host; co-tenants are observed, not excluded"}
spec:
  devices:
    requests: [{name: cards, device_class: spyre/pf, count: 2, selectors: [{attribute: spyre.profile, eq: a}]}]
  host: {node_class: metal-a, arch: x86_64, exclusive: false}
  labels: {node.class: metal-a, device.count: 2}
  policy: {validity_floor: shared_env}
```

- **`devices.requests`**: each has a `name`, a `device_class` registered by a pack (`spyre/pf`, `cpu/socket`,
  `storage/nvme`, `net/link`), a `count`, optional `per_node`, and `selectors` over the class's registered attributes
  (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`). CPUs and disks are devices like any other.
- **`devices.constraints`** require requests to agree on an attribute (`match_attribute`, `scope: per_node | all`).
- **`nodes`** is the number of hosts; **`interconnect.requests`** asks for links between them. A multi-host case
  states its network as part of what it needs:

```yaml validated in=docs-environment/environments.yaml
apiVersion: bench/v1alpha1
kind: Environment
metadata: {name: pair-of-hosts, description: "two hosts and a fast link; the interconnect is part of what the case needs"}
spec:
  nodes: 2
  devices:
    requests: [{name: sockets, device_class: cpu/socket, count: 1, per_node: true}]
    constraints: [{match_attribute: cpu.cores, requests: [sockets], scope: per_node}]
  interconnect:
    requests: [{name: fabric, device_class: net/link, count: 1, selectors: [{attribute: net.gbps, gte: 10}]}]
  host: {node_class: metal-a, arch: x86_64, exclusive: true}
  labels: {node.class: metal-a, device.count: 0}
```

- **`host`**: `node_class` (a NodeClass name), `arch`, `reference_node` (true = only a reference node will do),
  `exclusive`, `resources` (`cpu`, `memory` such as `256Gi`), `cpu_pinning` and `clean` (below).
- **`labels`** are declared, coarse facts stored as `env.label.<key>`. They are what series labels and label-mode reuse
  match on, so keep them few and stable (`node.class`, `device.count`).
- **`policy.validity_floor`** may only tighten the purpose's floor, never loosen it.
- **`base` + `patch`** derive an Environment from another one. A patch is a JSON Merge Patch: objects merge, **lists
  replace**, so a changed device request is written in full.

Devices are checked against a *leg* at plan time. With no `--devices` or `--nodes` the capacity check is skipped; with
`--devices spyre/pf=1` a two-card case is skipped with reason `capacity`:

```bash validated
spyre-bench plan --catalog examples/docs-environment --environment two-cards --purpose nightly --select smoke \
    --arch x86_64 --devices spyre/pf=1 --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-capacity.json # exit=4
```

```json validated subset-of=plan-capacity.json
{"summary": {"run": 0, "skip": 1, "reasons": {"capacity": 1}}, "leg": {"devices": {"spyre/pf": 1}}}
```

## CPU pinning

`host.cpu_pinning` states where the benchmarked process tree must run: `none`, `device_local` (the CPUs local to the
requested devices), `per_device` (each thread inside the CPUs local to one device) or `explicit` (a `cpus` list such as
`0-7,16-23`, only valid with `explicit`).

```yaml validated in=docs-environment/environments.yaml
apiVersion: bench/v1alpha1
kind: Environment
metadata: {name: two-cards-local, description: "same request, process tree kept on the CPUs local to the devices"}
spec:
  base: two-cards
  patch:
    host: {cpu_pinning: {policy: device_local, require_verified: true}}
```

- The engine does **not** apply a policy; the launcher or the benchmark's own command does. The engine samples the
  process tree while the command runs (Linux, through `/proc`) and records what it saw as `run.exec.cpu_pinning`,
  `run.exec.cpu_set` and `run.exec.pinning_verified`.
- `require_verified: true` turns an observed violation, or a placement it could not observe, into validity `suspect`
  (reason `pinning_violated` or `pinning_unverified`), so a verdict is `suspected`, never `regressed`.
- Pinning is a **run property**, not part of `env_id`: a pinned and an unpinned run on one node class are the same
  hardware. Its request is exposed as the derived label `env.label.cpu_pinning` (`none` when absent). List that label in
  the Project's `series_labels` (as the example Project does) and the two never share a series. Reuse and rolling
  baselines refuse a stored row whose pinning label differs, whether or not it is a series label.
- Two mistakes the validator catches: a topology policy needs an accelerator request to be local to, and the
  `cpu_pinning` label is derived from the request, so you may not declare it next to one:

```yaml validated base=docs-environment as=ops/op-local.yaml expect="error:needs an accelerator request"
apiVersion: bench/v1alpha1
kind: Environment
metadata: {name: cpus-only-local}
spec:
  devices:
    requests: [{name: cpus, device_class: cpu/socket, count: 2}]
  host: {node_class: metal-a, arch: x86_64, cpu_pinning: {policy: device_local}}
  labels: {node.class: metal-a, device.count: 0}
---
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-local}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  environment: cpus-only-local
  workload: {type: kernel/op, op: local, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  variants: [{name: op_local}]
```

```yaml validated base=docs-environment as=ops/op-declared.yaml expect="error:labels.cpu_pinning is derived"
apiVersion: bench/v1alpha1
kind: Environment
metadata: {name: declared-label}
spec:
  base: two-cards
  patch: {host: {cpu_pinning: {policy: device_local}}, labels: {cpu_pinning: device_local}}
---
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-declared}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  environment: declared-label
  workload: {type: kernel/op, op: declared, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  variants: [{name: op_declared}]
```

## Clean requests

An ordinary run shares the node with other work; the engine records that as `run.clean.exclusive: observed`. Some
purposes (`release`, `reference`) demand more, because a co-tenant adds noise to every number. `host.clean` asks for a
clean run:

```yaml validated in=docs-environment/environments.yaml
apiVersion: bench/v1alpha1
kind: Environment
metadata: {name: two-cards-clean, description: "reference purposes: wait for the node, hold it, release it"}
spec:
  base: two-cards
  patch:
    host: {reference_node: true, exclusive: true, clean: {mode: wait, max_wait: 4h, max_hold: 3h}}
    policy: {validity_floor: valid}
```

- `mode: wait` is the only mode: the launcher waits up to `max_wait` for a node with no other work, holds it up to
  `max_hold`, then releases it. `max_wait` and `max_hold` are required and are policy-capped by the launcher.
- The engine **never places and never cleans.** It plans: a case with a clean request is skipped (`capacity`) when the
  leg offers no exclusive access, and `capacity_timeout` or `max_hold_too_short` when the expected wait or the repeats'
  timeouts do not fit. It never runs unclean while claiming clean. Through the command line there is no way to state
  that access exists, so this plan always skips; a launcher that holds a node supplies that fact through the engine API.

```bash validated
spyre-bench plan --catalog examples/docs-environment --environment two-cards-clean --purpose reference --select smoke \
    --arch x86_64 --reference-node --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-clean.json # exit=4
```

```json validated subset-of=plan-clean.json
{"cases": [{"name": "op_scale_n1024", "action": "skip", "reason": "capacity", "clean": {"requested": true, "expected_exclusive": "enforced"}}]}
```

- The result's cleanliness is judged from facts the run records (`run.clean.exclusive`: `enforced`, `observed`, `none`
  or `unknown`; `run.clean.load_clean`; `run.clean.devices_exclusive`). A ValidityPolicy purpose states what it needs:
  `clean: any | clean | enforced` (guide 6). `clean` accepts `enforced`, or `observed` plus a clean load check plus
  enforced device exclusivity. `none` and `unknown` never qualify.
- A `reference_node: true` request (or a Suite with `requires_reference`) is skipped with `reference_node_required`
  unless the leg is a reference node (`--reference-node`).

## Series labels

`series_labels` in the Project is the list of `env.label.*` keys that split a series. The rule of thumb: a label belongs
there if changing it would legitimately change the number (device count, node class, pinning policy), and not if it is
bookkeeping (a team name, a ticket). Changing a label *value* on an existing Environment starts a new series, because
the series key hashes the label values with the `benchmark_id`. To change them deliberately, add a new Environment name
and keep the old one until the old series is retired.

## NodeClass: a class of hosts

```yaml validated in=docs-environment/node-classes.yaml
apiVersion: bench/v1alpha1
kind: NodeClass
metadata: {name: metal-a, description: "reserved pool of identical two-socket hosts"}
spec:
  arch: x86_64
  match: {labels: {example.com/pool: metal-a}}
  host_profile: {governor: performance, turbo: "on", smt: "on", cstate_policy: fixed, thp: madvise, cpu_quota_min_cores: 8}
  qualification: {expiry_days: 14, probe: repeat-probe, cv_limit_pct: 2.0, checks: [host_profile, governor]}
```

- **`match`** says which nodes belong to the class: explicit `nodes`, a `node_regex` or `labels`. It is required.
- **`host_profile`** is the expected host configuration (`governor`, `turbo`, `smt`, `cstate_policy`, `thp`,
  `numa_balancing`, `cpu_quota_min_cores`). Quote `"on"` and `"off"`: YAML 1.1 reads them as booleans. Packs add gates
  that compare the observed fact to this profile (for example governor equals `host_profile.governor`).
- **`qualification`** is the A/A policy: `probe` names the repeat probe, `cv_limit_pct` is the largest acceptable
  run-to-run variation, `checks` lists what the probe verifies and `expiry_days` says how long a passing probe keeps a
  node reference-grade. A class without a `qualification` block cannot make a result `qualification: current`.
- **Expiry applies to the node's status, not to results.** A result produced while the node was qualified stays valid
  after the qualification lapses; a *new* run on a node whose last passing probe is older than `expiry_days` is capped
  at `unqualified_env`.
- `extensions` carries pack-specific blocks (for example the device function a class exposes), validated by that
  pack's extension schema.

Qualification rows are written by the tool, never authored; their shape is in `examples/tool-output/qualification.yaml`.
A row counts when the node matches, the outcome is `pass`, `cv_pct` is at most the class's `cv_limit_pct`, and it is the
latest such row within `expiry_days`.

## Fingerprints and their versions

`env_id` hashes the **fingerprint**: facts a probe observes about the host and devices (`env.fp.kernel`,
`env.fp.governor`, `env.fp.cpu.model`, `env.fp.device.firmware`, ...), with no names or serials. The facts reach the
engine through the probe file given to `run`:

```json validated env-file
{"fingerprint": {"env.fp.kernel": "kernel-x", "env.fp.governor": "performance"}, "observations": {}, "node": "host-a", "qualified": true, "clean": {"exclusive": "observed"}}
```

```bash validated
spyre-bench run --plan plan.json --catalog examples/docs-environment --workdir examples/docs-environment \
    --env-file env.json --out results/env-run
```

Facts the file leaves out are *unknown* and lower the validity; they never raise it. Keys outside the pack registry are
dropped and noted in the Result. `qualified` must be a JSON boolean, not the string `"true"`.

Nine host and device topology attributes (`env.fp.host.sockets`, `env.fp.device.numa_node`, ...) are recorded but
**not hashed** under `fingerprint_version: 1` (the default), so a probe that starts reporting them leaves every
existing `env_id`, baseline and reuse decision intact. A Project opts in with `fingerprint_version: 2`, which also
hashes them; new Results carry the version they were built with, and recompute honours it:

```yaml validated base=docs-environment as=project.yaml
apiVersion: bench/v1alpha1
kind: Project
metadata: {name: docs-environment, owner: example-team}
spec:
  component: docs-environment
  packs: [{name: kernel, version: "0.2"}, {name: spyre, version: "0.2"}]
  default_suites: [ops]
  default_environment: two-cards
  visibility: public
  fingerprint_version: 2
  groupings:
    - {name: tier, ordered: true, cardinality: one, values: [sanity, core]}
    - {name: suite, ordered: false, cardinality: many}
  series_labels: [env.label.node.class, env.label.cpu_pinning]
---
apiVersion: bench/v1alpha1
kind: Suite
metadata: {name: ops}
spec: {type: perf}
---
apiVersion: bench/v1alpha1
kind: Selection
metadata: {name: smoke}
spec:
  cases: [op_scale_n1024]
  run: {repeats_min: 1}
```

Treat the switch as a series event: every Environment's `env_id` changes at once, so baselines and reuse that key on
`environment: id` restart. Coordinate it with whoever owns the baselines and do it once, at a planned moment. The
`core.env_complete` gate (at least 80% of the hashed fingerprint attributes present) counts the attributes the active
version hashes.

## Checklist

1. The Environment names only what the case needs; labels are few, stable and listed in `series_labels` only if they
   change the number.
2. Pinned and clean variants are separate Environments built with `base` + `patch` from the plain one.
3. A NodeClass has a `match`, a `host_profile` you would defend, and a `qualification` only if you run the probe.
4. `plan` under every Environment you added, with and without `--reference-node`, `--devices` and `--nodes`, to see the
   skip reasons you expect.
5. A change to `fingerprint_version`, a series label or a label value is announced to the baseline owners first.
