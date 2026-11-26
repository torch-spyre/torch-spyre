# 7. Write a pack

A pack teaches the engine one domain without adding a kind. Before a pack, `kernel/op` is just a string; after it,
the engine knows which fields a kernel-op Workload may carry, which of them are identity, which metrics it reports and
how they compare. This guide builds `examples/packs/docs-queue`, a pack for message-queue benchmarks, and the catalog
that uses it (`examples/docs-pack`). Use it as a template.

```text
examples/packs/docs-queue/
  pack.yaml                         the registration (kind: Pack)
  schemas/system-broker.json        JSON Schema of an mq/broker System
  schemas/workload-produce.json     JSON Schema of an mq/produce Workload
examples/docs-pack/                 a catalog that uses it
```

```bash validated
spyre-bench validate examples/docs-pack --pack examples/packs/docs-queue
```

## When to write a pack

- You benchmark a domain the built-in packs do not know (a database, a queue, a compiler).
- You need your own System or Workload `type` strings, metrics, device classes or probes.

You do not need one to add a Benchmark of a kind that exists (use `kernel/op` or `llm/serve`), to add a metric to an
existing domain (edit that pack: [guide 5](05-add-a-metric.md)), or to add an Environment (guide 3).

## Identity and compatibility

```yaml validated in=packs/docs-queue/pack.yaml
apiVersion: bench/v1alpha1
kind: Pack
metadata: {name: docs-queue, owner: example-team, description: "message-queue benchmarks: the pack the developer guides build"}
spec:
  prefix: mq
  version: 0.1.0
  requires: {core: "0.2"}
```

- `prefix` (1 to 16 lower-case letters and digits) namespaces everything the pack adds: types are `mq/broker`,
  metrics start `mq.`. Prefixes are global, so check the built-in ones first (`core`, `llm`, `spyre`, `kernel`,
  `comms`).
- `version` is a semantic version string. A Project asks for a *compatible range* (`version: "0.1"` accepts `0.1.x`) and
  a pack that does not satisfy it is a validation error. Raise the minor version for additions and the major version
  when a rename or removal can break a catalog.
- `requires.core` names the core version the pack was written against; `depends` lists other packs it builds on.

## Types: the vocabulary of Systems and Workloads

```yaml validated in=packs/docs-queue/pack.yaml
  types:
    - {name: broker, for: System, schema: schemas/system-broker.json}
    - {name: produce, for: Workload, schema: schemas/workload-produce.json, levels: [service], runner: producer,
       load: [closed], default_metrics: [mq.throughput, mq.latency.p99, mq.errors.ratio], non_identity: [progress_s]}
```

A type is `for` a `System`, a `Workload` or a `Store`, has a JSON Schema, and for Workloads also:

| Field | Meaning |
|---|---|
| `levels` | stack levels this Workload can be benchmarked at |
| `runner` | the name of a declared runner (below) |
| `load` | the load executors the type accepts (`closed`, `open`, `batch`, `iterations`, `ramp`); empty means no load block |
| `default_metrics` | the metrics a case records when its Benchmark names none |
| `non_identity` | field names under the type that are run attributes, not identity (`progress_s`, an output path) |
| `scenarios` | named presets of the load block (the `llm/serve` type has `single_stream`, `server`, ...) |
| `refs` | typed fields that reference a Model or Asset: identity then uses the referenced id, not the name |

Everything in the schema that is not `non_identity` is hashed into the `benchmark_id`. That decides how benchmarks
split into series, so write the schema deliberately: strict (`additionalProperties: false`), with enums for choices and
`minimum` for counts, and `required` for what a case cannot be defined without.

```json validated file=packs/docs-queue/schemas/workload-produce.json
{
 "$schema": "https://json-schema.org/draft/2020-12/schema",
 "$id": "https://schemas.bench.invalid/v1alpha1/packs/docs-queue/workload-produce.json",
 "title": "mq/produce",
 "type": "object",
 "additionalProperties": false,
 "properties": {
  "message_bytes": {"type": "integer", "minimum": 1},
  "acks": {"enum": ["none", "leader", "all"]},
  "progress_s": {"type": "integer", "minimum": 1}
 },
 "required": ["message_bytes"]
}
```

A violation is reported against the case that has it:

```yaml validated base=docs-pack as=queue/produce.yaml pack=packs/docs-queue expect="error:0 is less than the minimum of 1"
apiVersion: bench/v1alpha1
kind: System
metadata: {name: broker-1p}
spec: {type: mq/broker, backend: broker-a, token: p1, partitions: 1}
---
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: produce}
spec:
  type: perf
  level: service
  tier: sanity
  system: broker-1p
  workload: {type: mq/produce, message_bytes: 0, load: {executor: closed, concurrency: 1, duration: 1m}}
  variants: [{name: produce_zero}]
```

An unknown type names the registered ones and the nearest spelling. Two reserved System fields stay outside the schema
and outside identity: `backend` (the comparison axis) and `extensions`.

## Attributes: what the engine can group, order and plot by

```yaml validated in=packs/docs-queue/pack.yaml
  attributes:
    - {key: system.partitions, class: identity, role: root, type: int, order: numeric, plot: true}
    - {key: case.message_bytes, class: identity, role: root, type: int, order: numeric, plot: true}
    - {key: env.fp.broker.version, class: environment, role: root, type: string, fingerprint: true}
```

An attribute is a dotted key with a `class`: `identity` (hashed values of a case), `subject` (about the build under
test), `label` (a declared `env.label.*`), `environment` (an observed fact; `fingerprint: true` puts it into `env_id`)
or `run` (a property of one run). It has a `role` (`root` or `derived` from another), a `type`
(`string`, `int`, `float`, `bool`, `enum`, `uuid`), optional `values`, `unit`, `order` (`numeric`, `lexical`,
`declared`), `plot` and `stored_in` (the table that holds it).

Registering is what lets an Analysis use the key as `vary` or `compare.by`, and gives plotting tools its type and
order. Identity values are hashed whether or not you register them. Adding a fingerprint attribute changes `env_id`
for every environment that reports it, so treat it as a series event ([guide 3](03-environments-and-node-classes.md)).

## Runners, probes and gates

```yaml validated in=packs/docs-queue/pack.yaml
  runners:
    - {name: producer, protocol: process/v1, command: [python3, -m, mq_bench.producer]}
  probes:
    - {name: broker-version, protocol: process/v1, command: [python3, -m, mq_bench.probe], produces: [env.fp.broker.version]}
  gates:
    - {id: mq.broker_ready, phase: pre, reason: BROKER_NOT_READY, check: "broker accepts a test message"}
```

- A **runner** declares how a Workload type is executed (`process/v1` runs a command, `python/v1` calls an entry
  point) and is referenced by the type's `runner`. What `spyre-bench run` executes today is the Benchmark's own
  `run.command`; the runner declaration is the contract a launcher or a shared harness implements, and validation checks
  the references.
- A **probe** observes environment facts and says which attributes it `produces`. A probe's output reaches a run through
  `--env-file` (guide 3); the engine validates the keys against the registry and drops the rest with a note.
- A **gate** is a named check in a `phase` (`pre`, `in`, `post`, `ingest`) with a `reason` code in upper snake case.
  The engine's validity machinery consumes the *outcome* of gates; the check itself is implemented by the harness.
  Keep `check` a one-line statement of what passes.

## Extensions, vocabularies and device classes

- **`extensions`** add a pack-owned block to a NodeClass, System or Environment (`extensions: {mq: {...}}`), validated
  by the schema the pack names. Set `identity: true` only when the block's fields change the measured thing.
- **`vocabularies`** add values to `levels`, `test_types` and `purposes`. Add a level rather than reuse one that means
  something else; the example third-party pack adds one:

```yaml validated in=packs/postgres-example/pack.yaml
  vocabularies:
    levels: [transaction]
```

- **`device_classes`** register things an Environment can request (`name: vendor-or-domain/class`, a `kind` of
  `accelerator`, `cpu`, `storage` or `network`, and typed attributes that selectors can test, such as
  `net.gbps`). The core pack registers `cpu/socket`, `storage/nvme` and `net/link`.
- **`resolvers`** turn a build reference into `subject.id` and `subject.dep.*` attributes; **`analyzers`** declare named
  analysis steps with what they consume and produce.

## Using a pack

Pass the directory to every verb that loads a catalog: `--pack DIR` (repeatable; `BENCH_PACK`), then list it in the
Project with a compatible version:

```yaml validated in=docs-pack/project.yaml
apiVersion: bench/v1alpha1
kind: Project
metadata: {name: docs-pack, owner: example-team}
spec:
  component: docs-pack
  packs: [{name: docs-queue, version: "0.1"}]
  default_suites: [queue]
  default_environment: broker-host
  visibility: public
  series_labels: [env.label.node.class]
```

```bash validated
spyre-bench plan --catalog examples/docs-pack --pack examples/packs/docs-queue --purpose pr --select smoke \
    --arch x86_64 --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan.json
```

A Project that asks for a range the pack does not satisfy is refused with the versions it saw:

```yaml validated base=docs-pack as=project.yaml pack=packs/docs-queue expect="error:does not satisfy 0.2"
apiVersion: bench/v1alpha1
kind: Project
metadata: {name: docs-pack, owner: example-team}
spec:
  component: docs-pack
  packs: [{name: docs-queue, version: "0.2"}]
  default_suites: [queue]
  default_environment: broker-host
```

Built-in packs are the ones listed in `spyre_bench/packs/index.yaml`; to make a pack built in, add its directory under
`spyre_bench/packs/`, list it in the index and in `pyproject.toml`'s package data, and add a validating example. Prefer
a separate directory registered with `--pack` until more than one repository needs it.

## What belongs in the public engine and what does not

Put a pack in the public repository when it describes a domain in generic terms: the types, the registry, generic
probes and gates. Keep a pack **outside** it (a directory owned by the team and passed with `--pack`) when it carries
facts specific to one organisation: device classes of private hardware, extension fields for its node classes,
resolvers for its internal artifact stores, gates that call internal services, or thresholds.

A pack cannot modify another pack's entries. An organisation pack adds its own prefix, declares `depends` on the public
pack, and registers what is missing (a device class, an extension block, a resolver). If you find yourself wanting to
edit a built-in entry for one site, the fact belongs in an Environment, a NodeClass or an `extensions` block instead.

## Testing a pack

1. `spyre-bench validate <catalog> --pack <dir>` on a catalog that uses every type, with one case per executor.
2. Negative cases: one catalog edit per rule (a schema violation, a misspelled type, an unregistered metric, a version
   range outside the pack's). Guides 2 to 6 show the pattern; each negative is a block the snippet test checks.
3. `spyre-bench gen-schemas --check` if you touched the core schemas (not needed for a pack).
4. `spyre-bench plan` then `run` with a stub command: the Result should carry every metric the type's
   `default_metrics` promise.

## Checklist

1. Prefix is unique and short; version is semver; the Project range matches.
2. Every type has a strict schema; identity fields are deliberate; run attributes are in `non_identity`.
3. Metrics have unit, direction, category and noise ([guide 5](05-add-a-metric.md)).
4. Nothing organisation-specific is in a public pack: no hostnames, no thresholds, no internal service names.
5. An example catalog and a stub command exercise the pack in a test.
