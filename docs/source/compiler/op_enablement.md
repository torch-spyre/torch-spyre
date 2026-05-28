# Op enablement overview

This workflow guides developers from initial operator verification through full test retrieval and bug classification.

1. Create an op enablement Issue
2. Verify basic functionality
3. Add a unit test in torch-spyre
4. Verify your change with actual parameters in the target models
5. Submit a PR for implementation or tests
6. Report failures as bug Issues
7. Classify bugs with labels

# Create an Op Enablement Issue
Create an op enablement Issue in GitHub for the target operator.
This Issue will serve as the parent for all related test and bug Issues.
https://github.com/torch-spyre/torch-spyre/issues

# Verify Basic Functionality With Simple Parameters
Start by running the target op with minimal, simple parameter settings.
The goal is to confirm that the op runs without errors under basic conditions.

If the op is not yet implemented, the developer must implement the op first.
A detailed implementation procedure is available in the torch-spyre [wiki](https://github.com/torch-spyre/torch-spyre/wiki/OpFunc-development-use-cases),
and developers should follow that guide before proceeding with testing.

# Add Unit Tests to torch-spyre
Add [unit tests](https://github.com/torch-spyre/torch-spyre/tree/main/tests) for the op in the torch-spyre repository.
At this stage, you can keep the tests simple and focus on core behavior.

# Verify your implementation with actual parameters used in target models

It is important to verify your operations using the actual parameters used in the target models to meet project goals.
Run tests for your operation as follows:

```
pytest -c pytest_models.ini -s -rsapd tests/models/test_model_ops.py -k  op name
```

Specify an `op name`, for example `torch_add`. Note that the op name must use `_` instead of `.`.

# Fix failures

Fix failures at least for float16 and float32 if you saw the failures.
Integer types and bfloat16 are not supported now. So, you do not have to take care of these failures.

# Submit a PR

After completing the tests, submit a PR for the op implementation or the corresponding tests.

# Report Test Failures as Bug Issues

For each known failing test that is not fixed:

- Create a new bug Issue, or link to an existing relevant Issue
- Set the Issue type to "Bug"
- Include error logs, reproduction steps, and any relevant context
- Use IBM Bob to help summarize and organize the issues

# Assign Classification Labels

Once the root cause is identified, assign labels to categorize the bug.
Example labels include:

- Type conversion mismatch
- Padding
- Scalar tensor
- Size 1 dim

These classifications help frame the problem and streamline triage.

If you do not have permission to assign labels, ask the leads to do it.
