# Pinning experiments

Where a process's threads run on the host's CPUs can change a measurement, especially on machines with several NUMA
nodes or sockets. A pinning experiment answers "does it matter for this benchmark, and by how much?"

**Status: Available** in the engine for the vocabulary, the request, the observation and the analysis. **Planned**: the
launcher that applies a policy for you, per-device CPU assignment, and the dashboard comparison being populated by real
data.

## Vocabulary

| Policy (`run.exec.cpu_pinning`) | Meaning |
|---|---|
| `none` | no pinning requested |
| `device_local` | restrict to the CPUs local to the NUMA node(s) of the devices in use |
| `per_device` | each worker stays within the CPUs local to its own device |
| `explicit` | a CPU list you give |

The engine records what it observed: `run.exec.cpu_set`, `cpu_set_source`, `pinning_verified` (`true`, `false`,
`unknown`) and `pinning_violations`. A request with `require_verified: true` turns a violation (or an unobservable
placement) into validity `suspect`, so the verdict is `suspected`, never `regressed`.

## Pinned and unpinned runs never share a series

The request is exposed as a series label (`env.label.cpu_pinning`) and reuse and baselines refuse a stored row whose
label differs. So pinning on or off starts a separate history. Do not mix them to "get more data".

## Run an experiment

1. Pick the benchmark and an environment with at least two NUMA nodes.
2. Define the request in the Environment (`host.cpu_pinning: {policy: ..., require_verified: true}`), one Environment
   per policy, otherwise identical.
3. Run each policy with enough repeats for a noise estimate (at least five).
4. Compare with an `Analysis` that varies `run.exec.cpu_pinning`:

```bash
spyre-bench analyze pinning-effect --catalog examples/pinning-demo --results results.json   # array of Results from both policies, see Comparing
```

The example in `examples/pinning-demo` derives a per-token decode cost from two cases and compares it across pinning
policies within each node class. Read `within_noise` before reading the effect.

## Reading the result

- An effect inside the noise band means pinning does not matter for this benchmark on this machine.
- Effects can differ per node class; look at each class separately.
- A run that required verification and could not prove it is capped; do not use it for the comparison.

## When it fails

| Symptom | Fix |
|---|---|
| `pinning_unverified` | the observer could not read the placement (non-Linux, restricted procfs); the run is capped |
| `pinning_violated` | threads ran outside the requested set; check the launcher or the container's CPU limits |
| compare refuses the pair | the two sides differ in subject, environment or node; fix the one differing field |

## Who to ask

The benchmark owner and the platform maintainers; topology facts come from the machine's operators.
