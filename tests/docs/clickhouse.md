# ClickHouse DB Guide — `spyre` Database

## Instances

| | Dev | Prod |
|---|---|---|
| **Host** | `<dev clickhouse host>` | `<prod clickhouse host>` |
| **Port** | `443` | `443` |
| **User** | `<user>` | `<user>` |
| **Password** | `<password_dev>` | `<password_prod>` |
| **DB** | `spyre` | `spyre` |

---

### Create DB

```sql
CREATE DATABASE IF NOT EXISTS spyre;
```

## Table Schemas

### `spyre.test_runs`

```sql
CREATE TABLE spyre.test_runs
(
    run_id          UUID,
    workflow        LowCardinality(String),
    suite_name      String,
    filename        String,
    branch          LowCardinality(String),
    commit_sha      FixedString(40),
    gha_run_id      UInt64,
    gha_run_attempt UInt8,
    triggered_at    DateTime,
    ingested_at     DateTime DEFAULT now(),
    total_tests     UInt32,
    passed          UInt32,
    failed          UInt32,
    skipped         UInt32,
    errors          UInt32,
    xpass           UInt32,
    duration_s      Float32,
    retention_days  UInt16,
    xfail           UInt32,
    pr_number       UInt32
)
```

### `spyre.test_cases`

```sql
CREATE TABLE spyre.test_cases
(
    run_id       UUID,
    case_id      UUID,
    classname    String,
    name         String,
    op_name      LowCardinality(String),
    dtype        LowCardinality(String),
    status       LowCardinality(String),
    duration_s   Float32,
    fail_message String,
    triggered_at DateTime,
    workflow     LowCardinality(String)
)
```

### `spyre.run_properties`

```sql
CREATE TABLE spyre.run_properties
(
    run_id       UUID,
    case_id      UUID,
    prop_name    LowCardinality(String),
    prop_value   String,
    triggered_at DateTime
)
```

### Performance data

### `spyre.benchmark_runs`

```sql
CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id      UInt64 DEFAULT rand64(),
    source_file String NOT NULL,
    created_at  DateTime DEFAULT now()
)
```

### `spyre.perf_benchmarks`

```sql
CREATE TABLE IF NOT EXISTS perf_benchmarks (
    benchmark_id  UInt64 DEFAULT rand64(),
    run_id        UInt64 NOT NULL,
    record_type   String NOT NULL,
    operation_name Nullable(String),
    config_name    Nullable(String),
    stack          String NOT NULL,
    input_shapes   Nullable(String),
    batch_size    Nullable(Int32),
    prompt_length Nullable(Int32),
    run_mode      Nullable(String),
    kernel_name   Nullable(String),
    total_duration_ms       Nullable(Float64),
    kernel_mean_ms          Nullable(Float64),
    memcpy_htod_ms          Nullable(Float64),
    memcpy_dtoh_ms          Nullable(Float64),
    memset_device_ms        Nullable(Float64),
    memory_transfer_mean_ms Nullable(Float64),
    pt_util_percent         Nullable(Float64),
    num_runs       Nullable(Int32),
    custom_op_file Nullable(String),
    version_info   Nullable(String),
    created_at DateTime DEFAULT now()
)
```

### `spyre.sendnn_runs`

```sql
CREATE TABLE IF NOT EXISTS sendnn_runs (
    run_id      UInt64 DEFAULT rand64(),
    source_file String NOT NULL,
    created_at  DateTime DEFAULT now()
)
```

### `spyre.sendnn_benchmarks`

```sql
CREATE TABLE IF NOT EXISTS sendnn_benchmarks (
    benchmark_id  UInt64 DEFAULT rand64(),
    run_id        UInt64 NOT NULL,
    record_type   String NOT NULL,
    operation_name Nullable(String),
    stack          String NOT NULL,
    input_shapes   Nullable(String),
    batch_size    Nullable(Int32),
    prompt_length Nullable(Int32),
    run_mode      Nullable(String),
    total_duration_ms       Nullable(Float64),
    kernel_mean_ms          Nullable(Float64),
    memory_transfer_mean_ms Nullable(Float64),
    pt_util_percent         Nullable(Float64),
    custom_op_file Nullable(String),
    created_at DateTime DEFAULT now()
)
```

---

## Useful SQL Queries

### List all tables

```sql
SHOW TABLES FROM spyre;
```

### Check recent test runs (last 30, main branch, PRs only)

```sql
SELECT
    gha_run_id,
    pr_number,
    branch,
    commit_sha,
    workflow,
    sum(passed)      AS passed,
    sum(failed)      AS failed,
    sum(xfail)       AS xfail,
    sum(xpass)       AS xpass,
    sum(skipped)     AS skipped,
    sum(total_tests) AS total
FROM spyre.test_runs
GROUP BY gha_run_id, workflow, branch, pr_number, commit_sha
HAVING pr_number > 0 AND branch = 'main'
ORDER BY min(triggered_at) DESC
LIMIT 30;
```

### Find runs with missing test_cases (data integrity check)

```sql
SELECT tr.run_id, tr.gha_run_id, tr.commit_sha, tr.workflow, tr.triggered_at
FROM spyre.test_runs tr
LEFT JOIN spyre.test_cases tc ON tr.run_id = tc.run_id
WHERE tc.run_id IS NULL
ORDER BY tr.triggered_at DESC;
```

### Count test cases per run (verify inserts)

```sql
SELECT run_id, count(*) AS case_count
FROM spyre.test_cases
WHERE run_id IN ('<run_id_1>', '<run_id_2>')
GROUP BY run_id;
```

### Find runs by commit SHA

```sql
SELECT run_id, gha_run_id, commit_sha, workflow, triggered_at
FROM spyre.test_runs
WHERE commit_sha LIKE '<first_8_chars>%';
```

---

## Data Export/Import via curl (NDJSON) (Run the select * from the table and downoad the result as .ndjson file)

### Step 1 — Export from Source (Prod -> file)

#### Export `test_runs`

```bash
curl -u 'default:<password_prod>' \
  '<prod_host>/?query=SELECT%20*%20FROM%20spyre.test_runs%20FORMAT%20JSONEachRow' \
  -o test_runs.ndjson
```

#### Export `test_cases`

```bash
curl -u 'default:<password_prod>' \
  '<prod_host>?query=SELECT%20*%20FROM%20spyre.test_cases%20FORMAT%20JSONEachRow' \
  -o test_cases.ndjson
```

#### Export `run_properties`

```bash
curl -u 'default:<password_prod>' \
  '<prod_host>/?query=SELECT%20*%20FROM%20spyre.run_properties%20FORMAT%20JSONEachRow' \
  -o run_properties.ndjson
```

---

### Step 2 — Import into Target (file -> Dev)

#### Import `test_runs`

```bash
curl -u 'default:<password_dev>' \
  --data-binary @test_runs.ndjson \
  '<dev host>/?query=INSERT%20INTO%20spyre.test_runs%20FORMAT%20JSONEachRow'
```

#### Import `test_cases`

```bash
curl -u 'default:<password_dev>' \
  --data-binary @test_cases.ndjson \
  '<dev host>/?query=INSERT%20INTO%20spyre.test_cases%20FORMAT%20JSONEachRow'
```

#### Import `run_properties`

```bash
curl -u 'default:<password_dev>' \
  --data-binary @run_properties.ndjson \
  '<dev host>/?query=INSERT%20INTO%20spyre.run_properties%20FORMAT%20JSONEachRow'
```

---

# Sample INSERT Queries

## Insert into `spyre.test_runs`

```sql
INSERT INTO spyre.test_runs
    (gha_run_id, pr_number, branch, commit_sha, workflow, passed, failed, xfail, xpass, skipped, total_tests)
VALUES
    (27252059134, 2593, 'main', '72fd7dd695eead8494e2b2f9cd7879c5ebf08640', 'model-module-tests',   36,    0,   0, 0, 0,   36),
    (27252059117, 2593, 'main', '72fd7dd695eead8494e2b2f9cd7879c5ebf08640', 'model-ops-tests',     1096,  0, 988, 0, 0, 2084),
    (27181639121, 2411, 'main', 'c65450c2df18c102f8554c78c6f9a220c8a505a9', 'model-module-tests',   36,   0,   0, 0, 0,   36);
```

> **Note:** `run_id` is auto-generated by the ingestion pipeline. If inserting manually and `run_id` ends up as `00000000-0000-0000-0000-000000000000`, use `ALTER TABLE ... UPDATE` to fix it with the correct UUID from the source DB.

---

### Insert into `spyre.test_cases`

```sql
INSERT INTO spyre.test_cases
    (run_id, case_id, classname, name, op_name, dtype, status, duration_s, fail_message, triggered_at, workflow)
VALUES
    ('5fa0826b-ad49-4d1d-81d9-545c82f0425a', '848abf0f-7496-4ce5-a1d0-3d969ac63c7c', 'test.test_modules.TestModulePRIVATEUSE1', 'test_forward_GraniteAttention_spyre_float16', NULL, 'float16', 'passed', 0.003, NULL, '2026-06-10 06:39:19', 'model-module-tests'),
    ('5fa0826b-ad49-4d1d-81d9-545c82f0425a', 'c75ed867-1fb6-4db4-812a-cdaa80e0cbcb', 'test.test_modules.TestModulePRIVATEUSE1', 'test_forward_GraniteAttention_spyre_float32', NULL, 'float32', 'passed', 0.001, NULL, '2026-06-10 06:39:19', 'model-module-tests'),
    ('9996484a-0ecf-4cc7-a7f2-6ce832ef6a26', 'cae5fb9c-538e-4ca6-8c25-203901988183', 'tests.models.test_model_ops_v2.TestSpyreModelOpsPRIVATEUSE1', 'test_model_ops_db_torch_Tensor_contiguous__49_spyre_bfloat16', 'torch_Tensor_contiguous', 'bfloat16', 'passed', 6.899, NULL, '2026-06-09 12:13:46', 'model-ops-tests');
```

> **Important:** Do NOT use `--` inline comments inside a `VALUES` clause in ClickHouse — it causes a parse error. Keep all values comment-free.

---

### Insert into `spyre.run_properties`

```sql
INSERT INTO spyre.run_properties
    (run_id, case_id, prop_name, prop_value, triggered_at)
VALUES
    ('637783f7-90a6-4769-87b3-9ce6ef4e0882', '245aac98-59a1-4b84-8719-6e7d04957f6a', 'tag', 'platform__x86_64', '2026-06-10 06:27:27'),
    ('a4d52626-0b7c-491c-90b7-4f4453e2a725', '5908df62-664b-43ff-b904-85baa5380ff9', 'tag', 'dtype__int64', '2026-06-08 22:13:57'),
    ('a4d52626-0b7c-491c-90b7-4f4453e2a725', '5908df62-664b-43ff-b904-85baa5380ff9', 'tag', 'model__granite-3.3-8b-instruct-fms__spyre', '2026-06-08 22:13:57'),
    ('a4d52626-0b7c-491c-90b7-4f4453e2a725', '5908df62-664b-43ff-b904-85baa5380ff9', 'tag', 'op__torch_Tensor_add', '2026-06-08 22:13:57'),
    ('a4d52626-0b7c-491c-90b7-4f4453e2a725', '5908df62-664b-43ff-b904-85baa5380ff9', 'tag', 'platform__x86_64', '2026-06-08 22:13:57'),
    ('a4d52626-0b7c-491c-90b7-4f4453e2a725', 'e38a3910-bd20-4516-bf39-8a3b08330fac', 'tag', 'dtype__int64', '2026-06-08 22:13:57'),
    ('a4d52626-0b7c-491c-90b7-4f4453e2a725', 'e38a3910-bd20-4516-bf39-8a3b08330fac', 'tag', 'model__granite-3.3-8b-instruct-fms__spyre', '2026-06-08 22:13:57');
```

---
