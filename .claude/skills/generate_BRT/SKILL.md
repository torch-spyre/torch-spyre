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

   **Important** This process might take time so run with `nohup` and direct stdout and stderr to appropriate log files, let the user know of this file. Periodically check if everything is okay. 
   For an unresolved/open torch-spyre issue from a local `.md` file (the common case):

   ```bash
   cd /tmp/testgen-torchspyre
   uv sync
   .venv/bin/python -m testgen.main \
       --issue-md /path/to/issue.md \
       --repo-path /path/to/torch-spyre \
       --pod-yaml /path/to/pod.yaml \
       --output-dir outputs_torchspyre
   ```

   For GLM-4.6 add `--reasoning-as-content`:

   ```bash
   .venv/bin/python -m testgen.main \
       --issue-md /path/to/issue.md \
       --repo-path /path/to/torch-spyre \
       --pod-yaml /path/to/pod.yaml \
       --output-dir outputs_torchspyre \
       --reasoning-as-content
   ```

   If the user prefers the built-in resolved torch-spyre instance instead:

   ```bash
   .venv/bin/python -m testgen.main \
       --torchspyre \
       --pod-yaml /path/to/pod.yaml \
       --output-dir outputs_torchspyre
   ```

   **No OpenShift access?** Run with `--generate-only` to skip the pod:

   ```bash
   .venv/bin/python -m testgen.main \
       --issue-md /path/to/issue.md \
       --repo-path /path/to/torch-spyre \
       --output-dir outputs_torchspyre \
       --generate-only
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

## Important caveats to mention

- The agent code graph indexes the local `--repo-path` clone in `/tmp`. Python files are parsed with `ast`; C/C++/CUDA files are parsed with tree-sitter by default (`native_builder.py`) so the structure tools (`find_method`, `find_class`, `file_outline`) also work for native sources. 
- The execution pod uses the pre-built image `icr.io/ai_sw_accel/2.0/torch-spyre:latest`. If the user needs a different image, they must edit their `pod.yaml`.
- Keep `--max-router-visits` modest (default 3) for cost/time. Use `--generate-only` if they just want to inspect the generated test without running it in the pod, but warn them the quality will be lower.
