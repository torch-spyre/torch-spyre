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
     - §10 (Known limitations, especially the Python-only exploration caveat)

   Summarize the key facts back to the user:
   - The pipeline generates a pytest BRT via a multi-agent graph.
   - For torch-spyre, execution happens in an OpenShift pod, not Docker.
   - The default pod template is at `/home/shivankr/spyre-scripts/k8s/getting-started-torch-spyre/pod.yaml` unless overridden.
   - Unresolved/open issues run `git reset --hard HEAD` in the pod; resolved PR issues pin to `base_commit` and rebuild.

3. **Configure the LLM endpoint** before running anything. The framework reads these environment variables:

   ```bash
   export TESTGEN_LLM_BASE_URL=https://your-endpoint/v1   # must end in /v1
   export TESTGEN_LLM_API_KEY=EMPTY-or-your-key
   export TESTGEN_MODEL=moonshotai/Kimi-K2.5              # or whatever the endpoint serves
   ```

   Ask the user for their endpoint URL, API key, and model name if they are not already set. Do not guess credentials.

4. **Confirm OpenShift access and a usable pod template.**

   Tell the user (do not run it yourself):

   > Before running `testgen`, you must:
   >
   > 1. Run `oc login` to authenticate to the cluster.
   > 2. Ensure you have a `pod.yaml` that can create a torch-spyre pod in your namespace. The default used by the framework is:
   >    `/home/shivankr/spyre-scripts/k8s/getting-started-torch-spyre/pod.yaml`
   >
   >    If you need a different namespace or resource limits, copy that template, edit it, and pass `--pod-yaml /path/to/your/pod.yaml` to `main.py`.

   Wait for the user to confirm they have done both before proceeding.

5. **Collect the issue inputs.** Ask the user for:

   - The path to a local `.md` file containing the issue description (e.g. `issue.md`).
   - The path to their local `torch-spyre` repository clone (e.g. `~/torch-spyre`).
   - (Optional) The path to their custom `pod.yaml` if they are not using the default.

6. **Run the appropriate `testgen.main` command.**

   For an unresolved/open torch-spyre issue from a local `.md` file:

   ```bash
   cd /tmp/testgen-torchspyre
   uv sync
   .venv/bin/python -m testgen.main \
       --issue-md /path/to/issue.md \
       --repo-path /path/to/torch-spyre \
       --pod-yaml /path/to/pod.yaml \
       --output-dir outputs_torchspyre
   ```

   If the user prefers the built-in resolved torch-spyre instance instead:

   ```bash
   .venv/bin/python -m testgen.main \
       --torchspyre \
       --pod-yaml /path/to/pod.yaml \
       --output-dir outputs_torchspyre
   ```

   If `uv` is not available, fall back to creating a Python 3.10–3.12 virtual environment and installing dependencies from `pyproject.toml` / `requirements.txt`.

7. **After the run completes**, inspect the outputs:

   - `/tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/result.json` — outcome, router visits, final patch.
   - `/tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/state.json` — resumable state.
   - `/tmp/testgen-torchspyre/outputs_torchspyre/predictions.json` — SWT-bench-format patch.
   - `/tmp/testgen-torchspyre/outputs_torchspyre/<instance_id>/test_output.txt` or the pod log if execution failed.

   If the run fails with `system_error`, check for these markers in the pod log:
   - `TESTGEN_RESET_FAILED` — the pod could not `git reset --hard` to the requested commit.
   - `TESTGEN_COLLECTION_FAILED` — pytest collected no tests (bad import, syntax error, or mismatched test node id).

   Explain the failure to the user and suggest the next step (fix the issue description, adjust the pod yaml, re-run with `--resume-from integrator`, etc.).

## Important caveats to mention

- The agent code graph indexes the local `--repo-path` clone in `/tmp`. For torch-spyre this is mostly Python; C++/CUDA sources are visible to `grep`/`read_file` but the AST code graph does not index them by default, so native-only bugs may be hard to localize.
- The execution pod uses the pre-built image `icr.io/ai_sw_accel/2.0/torch-spyre:latest`. If the user needs a different image, they must edit their `pod.yaml`.
- Keep `--max-router-visits` modest (default 3) for cost/time. Use `--generate-only` if they just want to inspect the generated test without running it in the pod.
