# Phase 3b provenance schema fixtures

`valid_v1.json` is a merged multi-graph sidecar. Handle `200` is normalized and references fused constituents `101` and `102`; identity `vsancadvtjfcq6cv` proves an empty direct-handle list is valid; identity `atqydvnuutl766na` occurs in two compile scopes; the first compiler kernel retains two exact registrations; and the second compile records the honest provenance-level-0 gap.

`valid_cache_replay_v1.json` models cache replay of the first compile from `valid_v1.json`. It intentionally has the same compile ID and occurrence IDs, complete Spyre handle data, empty registrations, and `unavailable-cache-replay`. A merge keeps the richer `ok` projection and registrations from `valid_v1.json`; availability does not create a second occurrence.

Every compile and occurrence key is the real digest of the documented canonical payload. The tests recompute those keys, event names, handle relationships, ATen closure, registration aliases, upstream reverse mappings, and document status.

`fixture_manifest.json` defines invalid fixtures as deterministic mutations of a valid document. Reader diagnostics are separate from the writer-side `diagnostics` map stored in an artifact. Readers check `schemaVersion` first, then apply JSON Schema validation, then semantic validation, so a forward version deterministically reports `unsupported-schema-version`.
