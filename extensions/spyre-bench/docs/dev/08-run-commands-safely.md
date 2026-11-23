# 8. Choose and use run commands safely

A Benchmark's `run.command` is code that runs on shared machines, often from a pull request. The engine treats it as
untrusted by design: it gets a small environment, no credentials it was not handed explicitly, its own process group
and bounded output. This guide says what that means for the person writing the command, and for the person running a
plan. The demo Benchmark is `examples/docs-benchmark/ops/env-probe.yaml`; its command reports which `DEMO_` and
`SPYRE_` variables reached it.

```yaml validated file=docs-benchmark/ops/env-probe.yaml
# yaml-language-server: $schema=../../../spyre_bench/core/schemas/resources.schema.json
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: env-probe, owner: example-team, description: "shows what the command's environment holds"}
spec:
  type: perf
  level: kernel
  tier: core
  system: op-lib-cpu-fp32
  workload: {type: kernel/op, op: probe, dtype: fp32, shape: [1], load: {executor: iterations, iterations: 1}}
  run:
    repeats: 1
    timeout: 1m
    command: ["{python}", "tools/show_env.py", "--token", "example-not-a-secret", "--raw", "{raw_dir}", "--out", "{repeat_dir}"]
  variants:
    - name: env_probe
```

## Writing the command

1. **An argv array, never a shell string.** The validator refuses a string. A shell is one `["sh", "-c", ...]` away if
   you must, but then quoting and secret redaction are yours to get right.
2. **Use tokens, not guesses.** `{python}` is the interpreter the engine runs under, `{raw_dir}` and `{repeat_dir}` are
   where you write, `{case.*}` and `{system.*}` are the case's values. Relative paths resolve against `--workdir`.
3. **Write only where you were told.** Raw output (what the ingest reads) goes in `{raw_dir}`, per-repeat evidence and
   `metrics.json` in `{repeat_dir}`. `run` deletes and recreates those directories for the case.
4. **Exit 0 only on success.** Any other exit marks the repeat failed; so does exceeding `run.timeout`.
5. **Do not detach.** The engine runs each repeat in its own session and kills the whole process group on timeout,
   interrupt or SIGTERM, and kills stragglers when the command exits. A daemon you start with `setsid` outlives that
   and may write after the Result is built.
6. **Assume no network** unless the Benchmark's purpose needs it, and no credentials unless the invoker passes them.
7. **Print no measurements to stdout in public CI.** Logs are public there. Honour `BENCH_QUIET=1`, which `run --quiet`
   (or `BENCH_QUIET=1`) sets for the command.

## What the command's environment holds

The environment is an **allowlist**, matched case-insensitively. It contains:

| Group | Names |
|---|---|
| Basics | `PATH HOME USER LOGNAME SHELL TMPDIR TMP TEMP TERM TZ VIRTUAL_ENV CI LD_LIBRARY_PATH LANG LANGUAGE PWD HOSTNAME NODE_NAME` |
| Proxies | `HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY`, either case |
| Certificates | `SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE` |
| Build | `MAKEFLAGS MAKELEVEL MFLAGS CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH CPATH` |
| Caches | `HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TRANSFORMERS_CACHE` |
| Prefixes | `LC_ PYTHON SPYRE_ OMP_ MKL_ OPENBLAS_ NUMEXPR_ XDG_ CUDA_ TORCH_ VLLM_ KMP_ GOMP_ HF_DATASETS_ CMAKE_ CCACHE_` |

Everything else (tokens, warehouse settings, inherited `BENCH_*`) reaches the command only when the invoker names it:

- `--pass-env NAME` (repeatable; `BENCH_PASS_ENV`) passes one variable as it is.
- `--pass-env-prefix PREFIX` (repeatable; `BENCH_PASS_ENV_PREFIXES`, comma list) widens the prefix list.
- The Benchmark's own `env:` block sets knobs (`{DEMO_FLAG: 64}`); those are run attributes, not secrets.

Two filters apply to every inherited name, **prefixes included**, so a broad prefix cannot leak a credential:

1. A name with a *whole segment* (this list applies to inherited environment names; flag redaction below uses a smaller set) (split on `_ - .`) among `TOKEN PASS PASSWD PW PWD SECRET KEY CRED CREDENTIAL AUTH
   COOKIE BEARER DSN PAT SESSION PRIVATE CERTKEY APIKEY PASSPHRASE` is withheld. `SPYRE_BYPASS` and `SPYRE_KEYBOARD`
   pass, `SPYRE_KEY` does not.
2. A value that embeds URL credentials (`scheme://user:pw@host`) or looks like a token (a JWT; `sk-`, `ghp_`, `hf_` and
   similar prefixes; a long mixed letter-and-digit run) is withheld.

Every withheld name is listed in the repeat's `status.json` notes with the reason, never the value. A name you pass with
`--pass-env` is passed even if it looks secret: that is a decision, so make it deliberately. Run the demo with one
variable passed, one inherited by prefix but secret-looking, and one outside the list:

```bash validated
spyre-bench plan --catalog examples/docs-benchmark --purpose pr --select env_probe --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-env.json
SPYRE_API_TOKEN=abc123 DEMO_REGION=eu DEMO_OTHER=1 spyre-bench run --plan plan-env.json \
    --catalog examples/docs-benchmark --workdir examples/docs-benchmark --out results/env --pass-env DEMO_REGION
```

```json validated subset-of=results/env/cases/*/raw/env.json
{"count": 1, "env": {"DEMO_REGION": "eu"}}
```

Only `DEMO_REGION` arrived: `DEMO_OTHER` is outside the allowlist and `SPYRE_API_TOKEN` has a secret-looking segment.
The notes say so:

```json validated subset-of=results/env/cases/*/repeat-001/status.json
{"returncode": 0, "status": "ok", "notes": ["SPYRE_API_TOKEN withheld from the command (secret-looking name); name it with --pass-env to pass it"]}
```

From an interactive shell you will also see a note for `PWD`, whose name is on the list too. The command's working
directory is `--workdir` regardless, so that note is harmless.

## Secrets in argv, plans and logs

Do not put a secret in `run.command`, in a `{case.*}` value or in a flag. If a command needs a credential, take it from
an environment variable the invoker passes with `--pass-env`, and read it inside the command.

The engine still defends against a slip. Every argv it writes or prints (the results directory, `plan --json`,
`--dry-run`, `status.json`) is redacted by whole name segment and by value shape: values of `--token`, `--password`,
`--auth`, `--bearer`, `--cookie`, `--dsn`, `--key`, `--api-key`, `-p x`, `--user a:b`; `NAME=value` words and bare
`NAME value` pairs with a secret name, also inside a word (`--env=HF_TOKEN=x`); URL credentials; Authorization and
Cookie headers; and token-shaped words. `--passes`, `--author`, UUIDs and host names are left alone. A secret of no
recognisable shape in free text is **not** caught. The demo command carries a flag that is redacted:

```json validated subset-of=results/env/plan.redacted.json
{"cases": [{"name": "env_probe", "command": ["{python}", "tools/show_env.py", "--token", "<redacted>", "--raw", "{raw_dir}", "--out", "{repeat_dir}"]}]}
```

Two consequences to remember:

- The file `plan --out` writes is **not** redacted, because `run` executes it. Treat `plan.json` as sensitive if a
  command ever held a secret, and never publish it as a CI artifact. `run` refuses a plan that holds `<redacted>`.
- `plan.json` is **trusted input**: its `command` and `ingest` argv run exactly as written. Run only a plan you made
  yourself, or one produced by a pipeline step you trust. A CI job that runs a pull request's catalog executes the
  pull request's commands; plan and run in a job with no credentials and no write access, or take the plan from a
  trusted checkout.

## The ingest is a different trust domain

The benchmark command never sees the warehouse credentials. With `--ingest` the engine runs the ingest *after* the
benchmarks finish, in its own session, with stdin closed, in a fresh empty working directory and home, with an
environment of the base allowlist (minus `PYTHON*`, and without the invoker's `--pass-env-prefix` extras) plus `PYTHONSAFEPATH=1`, `CLICKHOUSE_*`, `INGEST_*` and what you name with
`--ingest-pass-env`. A benchmark can plant files in its working directory; it cannot plant them where the ingest looks.
The ingest package must be importable by an interpreter you name (`--ingest-python`, or `--ingest-pythonpath` for a
checkout outside `--workdir` and `--out`), otherwise `run --ingest` exits 2 before running anything. A custom
`--ingest-cmd` is trusted code with the same environment.

The simplest safe split, used by workflows that hold the credential only in one step: run with `BENCH_INGEST=0`, then
let a separate step with the credential ingest `results/<run>/cases/*/raw`.

## Bounds

| Bound | Default | Flag |
|---|---|---|
| per-repeat time | the Benchmark's `run.timeout` | `--timeout` caps it |
| each log (stdout, stderr) | 16 MiB; the rest is drained and dropped with a marker | `--max-log-bytes` |
| `metrics.json` | 8 MiB; larger is ignored with a note | `--max-metrics-bytes` |
| raw output per case | 1 GiB; over it a warning in `result_notes` | `--max-raw-bytes` |

A command whose log cannot be written (disk full) fails the repeat (`log_write_failed`) rather than blocking it.

## Checking before you run

`--dry-run` prints every rendered command and creates and runs nothing; a bad token exits 2 with nothing executed:

```bash validated
spyre-bench run --plan plan-env.json --catalog examples/docs-benchmark --workdir examples/docs-benchmark --dry-run
```

And a token the engine cannot resolve is a plan-time error, not a run-time surprise:

```yaml validated base=docs-benchmark as=ops/op-badtoken.yaml
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-badtoken}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  workload: {type: kernel/op, op: badtoken, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  run: {command: ["{python}", "tools/x.py", "{case.nope}"]}
  variants: [{name: op_badtoken}]
```

This file *validates* (the tokens are checked when the case is planned, because the set of case values depends on the
case) and then fails when planned. Always run `plan` on a new command:

```bash validated
spyre-bench plan --catalog examples/docs-benchmark --purpose pr --select smoke --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-ok.json
```

## Checklist

1. The command is an argv array and uses tokens for every path it writes.
2. It needs no variable outside the allowlist, or the invoker's documentation names each one for `--pass-env`.
3. No secret in the definition. No measurement on stdout under `BENCH_QUIET`.
4. It exits non-zero on failure and leaves no process behind.
5. `plan` and `run --dry-run` print what you expect; one real `run` produced a `metrics.json`.
6. Whoever runs a plan from a pull request does it without credentials.
