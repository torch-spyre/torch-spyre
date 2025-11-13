# Requirement files

These requirements files are currently used for CI purposes:

- [lint.txt](lint.txt): requirements for linter checks defined in pre-commit checks
- [test.txt](test.txt): requirements for torch-spyre unit and e2e tests
- [dev.txt](dev.txt): includes lint, test requirements as well as other dev requirements for code generation

Dependencies are defined here in addition to the [pyproject.toml](../pyproject.toml) to let the CI job function
without project dependencies that are not yet available in CI.

Eventually dependencies will converge to a single place.
