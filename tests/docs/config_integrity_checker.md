# OOT Config Checker — Usage Guide
**Authors:** Anubhav Jana (IBM Research, India), Ashok Pon Kumar Sree Prakash (IBM Research, India)

---

The config checker scans your YAML configs against the actual test file and reports three classes of problems:

| Check | What it catches | CI behaviour |
|---|---|---|
| **CHECK 1 — Duplicates** | The same test name listed in more than one config |
| **CHECK 2 — Missing** | A collectable test name not covered by any config |
| **CHECK 3 — Dead patterns** | A config pattern that matches zero collectable names |

Reference PR: [torch-spyre/torch-spyre#2391](https://github.com/torch-spyre/torch-spyre/pull/2391)

---

## Quick-start commands

### Run against all torch-spyre configs (default)

```bash
make check-all-configs
```

Scans every YAML under `tests/configs/torch_spyre_tests/` and auto-discovers which test file each config targets.

---

### Scope to a specific test file

Useful when you've just added tests to one file and want fast feedback without scanning everything.

```bash
make check-all-configs TEST_FILE=tests/test_launch_jobplan.py
```

---

### Override the config directory

Useful when you only care about a sub-tree of configs (e.g. inductor configs only).

```bash
make check-all-configs CHECK_CONFIGS=tests/configs/torch_spyre_tests/inductor
```

---

### Override both config directory and test file

The narrowest possible scope — one config directory against one test file.

```bash
make check-all-configs \
    CHECK_CONFIGS=tests/configs/torch_spyre_tests/inductor \
    TEST_FILE=tests/inductor/test_inductor_ops.py
```

---

## Makefile variables reference

| Variable | Default | Description |
|---|---|---|
| `CHECK_CONFIGS` | `tests/configs/torch_spyre_tests` | Root directory scanned for YAML configs |
| `TEST_FILE` | *(auto-discover)* | Pin the check to a single test file; unset = all discovered test files |

---

## User workflow — adding a new test

Follow these steps whenever you add new test methods to an existing or new `test_*.py` file.

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Write your test(s) in tests/<path>/test_foo.py                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Create / update a config YAML + add a GHA matrix entry          │
│                                                                     │
│  tests/configs/torch_spyre_tests/<subdir>/test_foo_config.yaml      │
│                                                                     │
│  test_suite_config:                                                 │
│    files:                                                           │
│      - path: ${TORCH_DEVICE_ROOT}/tests/<path>/test_foo.py          │
│        unlisted_test_mode: skip                                     │
│        tests:                                                       │
│          - names:                                                   │
│              - MyTestClass::test_new_feature.*                      │
│            mode: mandatory_success                                  │
└──────────────────────┬──────────────────────┬───────────────────────┘
                       │                      │
                       │         ┌────────────▼────────────────────────┐
                       │         │  2b. Large test file?               │
                       │         │  Distribute work — splitting a      │
                       │         │  test file into focused shards      │
                       │         │  (see section below)                │
                       │         └─────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. Run the duplicate / missing checker locally                     │
│                                                                     │
│                                                                     │
│  make check-all-configs         │
│                                                                     │
│  Expected output:                                                   │
│    CHECK 1: No duplicates.                                          │
│    CHECK 2: All N collectable names covered.                        │
│    CHECK 3: All patterns match at least one collectable name.       │
│    RESULT: 0 duplicates/missing  |  0 dead patterns                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Open a PR — two CI gates run automatically                      │
│                                                                     │
│  (a) oot-config-checker-tool   ->  reruns CHECK 1/2/3 in CI         │
│     Look at this run to verify no duplicates / missing tests        │
│                                                                     │
│  (b) Enforce Test CI Coverage  -> verifies the config is also       │
│     registered in a workflow matrix entry                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. Run the actual test suite CI                                    │
│                                                                     │
│  Trigger the relevant workflow run for your config:                 │
│    torch_spyre_tests.yaml  /  upstream_tests.yaml  / etc.           │
│                                                                     │
│  Check that your new test variant appears and passes                │
│  (mode: mandatory_success > must be green).                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Distributing work across configs — splitting a test file

When a test file is large (e.g. `test_inductor_ops.py` with 160+ collectable names), split the coverage across **multiple focused config YAMLs** rather than one monolithic file. Each YAML owns a logical shard:

### Example: regex shard (pointwise ops)

```yaml
# tests/configs/torch_spyre_tests/inductor/test_inductor_ops_pointwise_config.yaml
test_suite_config:
  files:
    - path: ${TORCH_DEVICE_ROOT}/tests/inductor/test_inductor_ops.py
      unlisted_test_mode: skip
      tests:
        - names:
            # Covers: test_pointwise, test_sqrt, test_rsqrt, test_log_,
            #         test_alias_operands, test_cmp, test_logical_not, test_bool
            - TestOps::test_(pointwise|sqrt|rsqrt|log_|alias_operands|cmp|logical_not|bool).*
          mode: mandatory_success
          tags:
            - ops__inductor-pointwise
```

One regex pattern covers an entire family of ops. Use the `tags` field to give the shard a meaningful pytest marker (`-m ops__inductor-pointwise`).

### Example: explicit list shard (keepdim reductions)

```yaml
# tests/configs/torch_spyre_tests/inductor/test_inductor_ops_reductions_config.yaml
test_suite_config:
  files:
    - path: ${TORCH_DEVICE_ROOT}/tests/inductor/test_inductor_ops.py
      unlisted_test_mode: skip
      tests:
        - names:
            - TestOps::test_min_keepdim.*
            - TestOps::test_max_keepdim.*
            - TestOps::test_aminmax_keepdim.*
            - TestOps::test_sum_keepdim.*
            - TestOps::test_mean_keepdim.*
            - TestOps::test_vector_norm_keepdim.*
            - TestOps::test_matrix_norm_keepdim.*
            - TestOps::test_linalg_norm_keepdim.*
            - TestOps::test_argmin_keepdim0
            - TestOps::test_min_tuple_output_keepdim0
            - TestOps::test_logsumexp_keepdim0_known_xfail
            # test_mean has a param "3d_dim0_keepdim" — explicit entry needed
            # because the base method lives in the misc shard
            - TestOps::test_mean_3d_dim0_keepdim
          mode: mandatory_success
```

Use explicit name lists when the set of tests is tightly bounded and you want to document exactly what is included. Comments explain non-obvious entries (like the `test_mean_3d_dim0_keepdim` case above).

### Example: custom reducer shard

```yaml
# tests/configs/torch_spyre_tests/inductor/test_inductor_ops_custom_reduce_config.yaml
test_suite_config:
  files:
    - path: ${TORCH_DEVICE_ROOT}/tests/inductor/test_inductor_ops.py
      unlisted_test_mode: skip
      tests:
        - names:
            - TestOps::test_reduce_keepdim0.*
            - TestOps::test_reduce_edge_keepdim0.*
          mode: mandatory_success
```

### Splitting guidelines

| Guideline | Rationale |
|---|---|
| One logical concern per YAML | Easier to review; CI failure immediately identifies the failing shard |
| Use `tags` for large shards | Lets you run `-m ops__inductor-pointwise` locally without loading all 160+ variants |
| Add comments for non-obvious names | Future maintainers won't know why `test_mean_3d_dim0_keepdim` lives in the reductions shard |
| Prefer regex over long explicit lists when the pattern is stable | Regex shards automatically pick up new ops that match the pattern |
| Use explicit lists when the set is bounded and documented | Prevents accidental inclusion of unrelated new tests |
| After splitting, always run `make check-all-configs` | Verify CHECK 1 (no cross-shard duplicates) and CHECK 2 (full coverage) |

---

## Reading the checker output

```
Scanning 44 config file(s) under tests/configs/torch_spyre_tests
 
Reference: /home/.../tests/inductor/test_inductor_ops.py
  163 collectable name(s)
   77 helper-only method(s) (excluded from MISSING check)
 
Test file: test_inductor_ops.py  (92 pattern(s) across 10 config(s))
  CHECK 1: No duplicates.
  CHECK 2: All 163 collectable names covered.
  CHECK 3: All patterns match at least one collectable name.
 
RESULT: 0 duplicates/missing  |  0 dead patterns
```

- **collectable names** — parameterised test variants the checker can see (these must all be covered).
- **helper-only methods** — methods that are not `test_*` or have no `@ops`/`@modules` decorator; excluded from the missing check.
- **patterns across N config(s)** — total regex/literal entries seen across all YAMLs for this test file.
- **dead patterns** — a pattern in a YAML that matches nothing; usually caused by a test rename. Warning only — does not fail CI, but should be cleaned up.
