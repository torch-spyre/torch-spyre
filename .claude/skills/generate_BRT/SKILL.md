---
name: torchspyre-brt
description: Use ONLY when a torch-spyre developer wants to generate a bug-reproduction test (BRT) for a torch-spyre GitHub issue. Triggers on phrases like "torch-spyre bug", "generate repro test", "BRT for torch-spyre", "test for torch-spyre issue", or when the user is debugging a torch-spyre failure and needs a failing pytest case.
---

# Generate a torch-spyre bug-reproduction test (BRT)

This skill walks a torch-spyre developer through generating a bug-reproduction test using the `testgen` framework. The framework is in the `torchspyre` branch of the `test-gen` repository.

## What you should do

1. **Clone the `test-gen` repository (torchspyre branch)** into a temporary location.

   ```bash
   git clone -b torchspyre https://github.ibm.com/Shivank-Rajput/test-gen.git /tmp/testgen-torchspyre
   cd /tmp/testgen-torchspyre
   ```

   If the clone fails, ask the user for the correct URL or access credentials.

2. **Read the project documentation** to understand the workflow and current torch-spyre handling:

   - Read `/tmp/testgen-torchspyre/README.md` in full.
   - Read `/tmp/testgen-torchspyre/CLAUDE.md`, especially:
     - §3 (Architecture)
     - §5 (How execution actually works)
     - §6 point 11 (Torch-spyre execution uses OpenShift pods)
      - §10 (Known limitations, especially the native-repo exploration caveat)

   Summarize the key facts back to the user:
   - The pipeline generates a pytest BRT via a multi-agent graph.
   - For torch-spyre, execution happens in an OpenShift pod, not Docker.
   - `--pod-yaml` is required: there is no default pod template. The user must provide a pod.yaml that can create a torch-spyre pod in their namespace.
   - Unresolved/open issues run `git reset --hard HEAD` in the pod; resolved PR issues pin to `base_commit` and rebuild.

3. **Configure the LLM endpoint and authentication** before running anything. The framework reads these environment variables:

   ```bash
   # The endpoint must end in /v1
   export TESTGEN_LLM_BASE_URL=https://your-endpoint/v1

   # API key for the endpoint
   export TESTGEN_LLM_API_KEY=EMPTY-or-your-key

   # Model id must match what the endpoint serves
   export TESTGEN_MODEL=moonshotai/Kimi-K2.5
   ```

   Ask the user for their endpoint URL, API key, and model name if they are not already set. Do not guess credentials.

   **Model-specific setup:**

   - **IBM RITS endpoint:** RITS authenticates via a custom header (`RITS_API_KEY`), not the
     standard `Authorization: Bearer` header. Set both the header name and the key:
     ```bash
     export TESTGEN_API_KEY_HEADER=RITS_API_KEY
     export TESTGEN_LLM_API_KEY=$RITS_API_KEY
     ```
     `config.py` defaults `TESTGEN_API_KEY_HEADER` to `RITS_API_KEY`, so set the API key explicitly.

   - **GLM-4.6-style models:** GLM-4.6 often returns an empty `content` field and puts the
     entire answer (including ```python / ```json blocks) in a `reasoning` field. Without
     `--reasoning-as-content` the fixer and integrator see empty responses and produce no
     usable patch, so execution never happens. Always pass this flag for GLM-4.6:
     ```bash
     .venv/bin/python -m testgen.main ... --reasoning-as-content
     # or: export TESTGEN_REASONING_AS_CONTENT=1
     ```

   - **Token limits:** some models (especially reasoning models like GLM-4.6) emit very long
     reasoning traces for the fixer. If you see `out=16384` tokens in the logs, the response
     may be truncated. Raise the cap:
     ```bash
     export TESTGEN_MAX_TOKENS=32768
     ```

4. **Confirm OpenShift access and a usable pod template.**

   Tell the user (do not run it yourself):

   > Before running `testgen`, you MUST do ALL of these:
   >
   > 1. Run `oc login` to authenticate to the cluster.
   > 2. Verify access with `oc whoami` and `oc project torch-spyre-cicd` (or whatever namespace is in your pod.yaml).
   > 3. Provide a `pod.yaml` that can create a torch-spyre pod in your namespace. There is no built-in default template. Pass it with `--pod-yaml /path/to/your/pod.yaml`.
   >
   > If you do not have OpenShift access, you can still run with `--generate-only`, but the test will not be executed and the quality is usually much lower because the feedback router cannot revise the test.

   Wait for the user to confirm they have done steps 1–3 before proceeding.

5. **Collect the issue inputs.** Ask the user for:

   - The path to a local `.md` file containing the issue description (e.g. `issue.md`).
   - The path to their local `torch-spyre` repository clone (e.g. `~/torch-spyre`).
   - The path to their `pod.yaml` (required for execution; no default).

6. **Run the appropriate `testgen.main` command.**

   **ALWAYS run with `nohup` (or an equivalent detached form) and redirect stdout and stderr to a
   log file.** Never run it in the foreground. Immediately tell the user the exact log-file path so
   they can follow along. For example:

   ```bash
   cd /tmp/testgen-torchspyre
   uv sync
   mkdir -p outputs_torchspyre   # the log lives here; create it before the redirect opens
   nohup .venv/bin/python -m testgen.main \
       --issue-md /path/to/issue.md \
       --repo-path /path/to/torch-spyre \
       --pod-yaml /path/to/pod.yaml \
       --output-dir outputs_torchspyre \
       > outputs_torchspyre/run.log 2>&1 &
   echo "started; follow with: tail -f outputs_torchspyre/run.log"
   ```

   **BE PATIENT — this is expected to take a long time (tens of minutes to hours).** The run makes
   many LLM calls (K-fix consensus generates K fixes + K tests, then revises over several router
   visits) and each call to a reasoning model can take minutes; on top of that, every pod execution
   includes scheduling, image pull, and possibly a native rebuild. **A run that is quiet is almost
   always still working, not hung.** Do NOT kill it, do NOT assume failure, and do NOT restart it
   just because there has been no output for a while.

   **How to decide whether to keep waiting:**
   - **Keep waiting** if the log shows normal progress (LLM-call timing lines, `[POD …]` lines,
     agent activity) and there are no fatal errors — **including transient `503` / `502` / `429`
     errors from the LLM endpoint or the cluster.** A few 5xx/timeout blips are normal; the LLM
     cache and retries mean the run recovers on its own. These are NOT a reason to intervene.
   - Poll **infrequently** (e.g. every few minutes), not in a tight loop — check with
     `tail -n 50 <logfile>` and `oc get pods`. A running pod / recent log lines = healthy.
   - **Only intervene** on a genuine fatal error (see *Common errors* below): `Unauthorized`/not
     logged in, missing/invalid `--pod-yaml`, `ImagePullBackOff`, `TESTGEN_RESET_FAILED` /
     `TESTGEN_COLLECTION_FAILED`, repeated auth (`401`/`403`) or `404` from the LLM endpoint, or the
     process having actually exited (check with `jobs` / `ps`). Persistent (not occasional) 5xx for
     many minutes with no progress also warrants a look at the endpoint.

   For an unresolved/open torch-spyre issue from a local `.md` file (the common case) the command is
   as above (`--issue-md` + `--repo-path` + `--pod-yaml`).

   Wrap **every** variant below in the same `nohup … > <logfile> 2>&1 &` form as above — never run
   any of them in the foreground.

   For GLM-4.6 add `--reasoning-as-content`:

   ```bash
   nohup .venv/bin/python -m testgen.main \
       --issue-md /path/to/issue.md \
       --repo-path /path/to/torch-spyre \
       --pod-yaml /path/to/pod.yaml \
       --output-dir outputs_torchspyre \
       --reasoning-as-content \
       > outputs_torchspyre/run.log 2>&1 &
   ```

   If the user prefers the built-in resolved torch-spyre instance instead:

   ```bash
   nohup .venv/bin/python -m testgen.main \
       --torchspyre \
       --pod-yaml /path/to/pod.yaml \
       --output-dir outputs_torchspyre \
       > outputs_torchspyre/run.log 2>&1 &
   ```

   **No OpenShift access?** Run with `--generate-only` to skip the pod:

   ```bash
   nohup .venv/bin/python -m testgen.main \
       --issue-md /path/to/issue.md \
       --repo-path /path/to/torch-spyre \
       --output-dir outputs_torchspyre \
       --generate-only \
       > outputs_torchspyre/run.log 2>&1 &
   ```

   Warn the user that `--generate-only` produces a test without execution feedback, so the router cannot revise it and the quality is usually much worse.

   If `uv` is not available, fall back to creating a Python 3.10–3.12 virtual environment and installing dependencies from `pyproject.toml` / `requirements.txt`.

7. **After the run completes**, inspect the outputs and extract the test:

   - `/tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/result.json` — outcome, router visits, final patch.
   - `/tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/state.json` — resumable state.
   - `/tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/model_patch.patch` — the finalized git patch.
   - `/tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/conversation.jsonl` — per-agent LLM conversation log.
   - `/tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/test_output.txt` or the pod log if execution failed.

   **Reconstruct the generated test into a runnable `.py` file:**

   ```bash
   cd /tmp/testgen-torchspyre
   .venv/bin/python -m testgen.extract_test outputs_torchspyre/<instance_id>
   ```

   This writes `extracted_test_<filename>.py` in the instance directory. For a merge placement it
   checks out the base commit and applies the patch so you can see the test in its host file.

   **Reconstruct the fixer's candidate fix (for debugging the F→P oracle):**

   ```bash
   .venv/bin/python -m testgen.extract_fix outputs_torchspyre/<instance_id> --unresolved
   ```

   **Apply the finalized patch directly in the target repository:**

   ```bash
   cd /path/to/torch-spyre
   git apply /tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/model_patch.patch
   ```

   If the run fails with `system_error`, check for these markers in the pod log or conversation log:
   - `TESTGEN_RESET_FAILED` — the pod could not `git reset --hard` to the requested commit.
   - `TESTGEN_COLLECTION_FAILED` — pytest collected no tests (bad import, syntax error, or mismatched test node id).
   - `Unauthorized` from `oc apply` — the user is not logged into the OpenShift cluster.
   - `consensus: generated 0/3 usable fixes, 0/3 tests` — the fixer/test_generator returned empty content (likely GLM-4.6 without `--reasoning-as-content`) or hit the token limit.

   Explain the failure to the user and suggest the next step (fix the issue description, adjust the pod yaml, re-run with `--resume-from integrator`, pass `--reasoning-as-content`, raise `TESTGEN_MAX_TOKENS`, etc.).

## Common errors and how to fix them

When something goes wrong, match the symptom below before guessing. The most common class by far is **OpenShift/`oc` problems** — and the most common of those is simply *not being logged in*.

### OpenShift / `oc` errors (execution)

- **`oc` commands fail / `Unauthorized` / `error: You must be logged in to the server`.**
  The user is **not logged into the cluster** (or their token expired). This is the single most
  frequent failure. Fix: `oc login <cluster-url> --token=<token>` (or `oc login -u <user>`), then
  re-run. Verify first with `oc whoami`.
- **`oc: command not found`.** The OpenShift CLI is not installed / not on `PATH`. The user must
  install `oc` (the OpenShift client) before any pod execution can happen. There is no way around
  this except `--generate-only` (no execution).
- **`error: the server doesn't have a resource type` / "namespace not found" / `forbidden`.**
  The user is logged in but pointed at the **wrong project/namespace**, or lacks write access to the
  namespace named in the `pod.yaml`. Fix: `oc project <namespace>` (e.g. `oc project torch-spyre-cicd`)
  and confirm they can create pods there. The namespace in `--pod-yaml` must match one they can write to.
- **`ValueError: pod_yaml_path is required` / missing `--pod-yaml`.** There is **no default pod
  template**. The user must pass `--pod-yaml /path/to/pod.yaml`. If they don't have one, they cannot
  run execution — offer `--generate-only`.
- **`--pod-yaml` file not found / invalid YAML.** The path is wrong or the file is malformed. Check the
  path exists and is a valid pod spec for their namespace/image.
- **Pod never becomes ready / deploy timeout / `ImagePullBackOff` / `ErrImagePull`.** The cluster
  can't schedule or pull the image (`icr.io/ai_sw_accel/2.0/torch-spyre:latest` by default). Watch with
  `oc get pods` and `oc describe pod <name>`. Causes: busy cluster (just slow — wait), missing image-pull
  secret, or a wrong image in the `pod.yaml`. A busy cluster is contention, not a hang.
- **`oc cp` fails with `tar: … Cannot open: Permission denied`.** OpenShift runs the pod as an arbitrary
  UID and `/tmp` is sticky/world-shared, so a leftover file from a prior run's UID blocks the copy. The
  harness already copies the eval script into the pod's HOME (`/home/senuser`) with a unique name to avoid
  this; if it still appears, the pod's HOME may not be writable — check the `pod.yaml`'s `securityContext`.

### Pod-side execution markers (in the pod log / `test_output.txt`)

- **`TESTGEN_RESET_FAILED`.** The pod could not `git reset --hard` to the requested commit — for a
  resolved instance the `base_commit` isn't present in the pod's clone. Maps to `system_error`. For an
  open issue this should be `git reset --hard HEAD` and shouldn't happen; if it does, the pod's repo is
  in a bad state.
- **`TESTGEN_COLLECTION_FAILED`.** pytest collected **no tests** — a bad import, a syntax error, or a
  mismatched test node id. Maps to `system_error`. Usually a bug in the generated/placed test; re-run
  with `--resume-from integrator` after inspecting the placed test.
- **Stale native extension / base tests fail before the generated test runs.** On a *resolved* instance,
  if the `uv sync … --reinstall-package torch-spyre` rebuild was skipped or interrupted, the installed
  `.so` no longer matches the checked-out source. Also happens if a patch touches C/C++/CUDA but the
  rebuild didn't trigger. Re-run so the rebuild completes; raise `TESTGEN_EXEC_TIMEOUT` if the compile is
  being killed (default 1200s → try 1800s on a slow/busy cluster).

### LLM / endpoint errors (generation)

- **Empty responses / `consensus: generated 0/K usable fixes, 0/K tests` / nothing executes.** For
  **GLM-4.6-style models** the answer comes back in a `reasoning` field with empty `content`. Always pass
  `--reasoning-as-content` (or `export TESTGEN_REASONING_AS_CONTENT=1`) for GLM.
- **Truncated responses (`out=16384` / `finish_reason=length`).** The token cap is too low for a long
  reasoning trace. Raise it: `export TESTGEN_MAX_TOKENS=32768`. Don't lower it.
- **`401`/`403` from the LLM endpoint.** Wrong or missing API key/header. For **IBM RITS** the key goes
  in a custom header, not `Authorization: Bearer`: `export TESTGEN_API_KEY_HEADER=RITS_API_KEY` and
  `export TESTGEN_LLM_API_KEY=$RITS_API_KEY`.
- **`404` / connection errors from the LLM endpoint.** Usually `TESTGEN_LLM_BASE_URL` is wrong — it must
  **end at `/v1`** (the OpenAI SDK appends `/chat/completions` itself), and `TESTGEN_MODEL` must exactly
  match a model the endpoint serves.
- **Endpoint stalls mid-run.** Every successful LLM call is cached to `<output-dir>/llm_cache.sqlite`;
  just re-run the same command and it fast-forwards to the failed call. Don't `--clear-llm-cache` unless
  you deliberately want fresh calls.

### Input / setup errors

- **`--repo-path` or `--issue-md` not found.** The paths must point at an existing local `torch-spyre`
  clone and an existing `.md` file. `--repo-path` is cloned into `/tmp` for the code graph; `--issue-md`
  is read as the problem statement.
- **`uv: command not found`.** Fall back to a manual Python 3.10–3.12 venv and install from
  `pyproject.toml`.
- **Run takes a very long time / seems stuck.** Expected — full runs with a pod + K-fix consensus take a
  while. Run under `nohup` with stdout/stderr redirected to a log file (see step 6), and check progress
  with `oc get pods` and by tailing the log, not by assuming a hang.

### Pod cleanup error
- Sometimes after the pipeline has finished, the deployed pod may not be cleaned up. 
- You can figure out when this has happend by reading the last few lines of stdout, or, by running `oc get pods` and checking if the created pod is still alive.
- Inform the user about the status of the pod.

## Important caveats to mention

- The agent code graph indexes the local `--repo-path` clone in `/tmp`. Python files are parsed with `ast`; C/C++/CUDA files are parsed with tree-sitter by default (`native_builder.py`) so the structure tools (`find_method`, `find_class`, `file_outline`) also work for native sources. 
- The execution pod uses the pre-built image `icr.io/ai_sw_accel/2.0/torch-spyre:latest`. If the user needs a different image, they must edit their `pod.yaml`.
- Keep `--max-router-visits` modest (default 3) for cost/time. Use `--generate-only` if they just want to inspect the generated test without running it in the pod, but warn them the quality will be lower.
