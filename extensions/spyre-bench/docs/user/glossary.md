# Glossary

| Term | Meaning |
|---|---|
| Analysis | a stored query over Results: `derive` computes a metric from two cases, `compare` tabulates it by an attribute |
| anchor | a fixed comparison point (a good run, a release, a reference) used instead of a rolling window |
| baseline | the earlier runs a verdict is judged against; rolling (previous 20 eligible runs) unless anchored |
| benchmark | a template that expands into cases; its id derives from what it asks |
| band | the allowed movement around the baseline: the larger of a policy floor and twice the baseline's robust noise |
| case | one concrete benchmark (one expansion), the unit that has a `benchmark_id` |
| catalog | the folder of YAML definitions for a product |
| clean run | a run on a machine whose resources were reserved for it |
| environment | the request (devices, host, labels) and, once resolved, the machine's fingerprint |
| `env_id` | hash of the fingerprint attributes only |
| fingerprint | the machine facts that define an environment |
| identity | the derived id of a case; changing an input changes it |
| invalidation | marking data untrustworthy (`suspect`, `invalidated`, `restored`) |
| node class | a class of hosts with the same expected profile |
| pack | a registration of types, attributes, metrics and device classes |
| placement | the nodes and devices a repeat actually ran on |
| purpose | why a run happened (`nightly`, `pr`, `on-demand`, ...); decides whether it builds baselines |
| reference | an approved comparison target that is not our own measurement |
| repeat | one execution of a case; several repeats give the noise estimate |
| reuse | using a stored valid result instead of measuring again |
| series | the history of one benchmark on one environment label set (`series_key`) |
| subject | the thing being measured (a build or image) |
| trust | who produced the data: `ci`, `lab`, `external`; server-derived |
| validity | whether a run's measurement was sound: `valid` down to `invalid` |
| verdict | the advisory judgement of one metric: pass, improved, regressed, suspected, no_baseline |
