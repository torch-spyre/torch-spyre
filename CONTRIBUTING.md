# Contributing to torch-spyre

Thank you for your interest in contributing to torch-spyre! There are many ways to
contribute, and we appreciate all contributions.

Please:

* Open an issue before large PRs
* Follow our code style and tests
* Sign off commits with `git commit -s`
* Sign commits with a GPG or SSH key registered on your GitHub account (`git commit -s -S`, or set `git config commit.gpgsign true` once). `main` requires signed commits, so an unsigned commit blocks the merge. See [GitHub's guide to signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).

# Code Quality

The PR needs to meet the following code quality standards:

* We adhere to Google Python style guide and Google C++ style guide.
* Pass all linter checks. Please use pre-commit to format your code. See [this link](https://pre-commit.com/#usage) if pre-commit is new to you.
* The code needs to be well-documented to ensure future contributors can easily understand the code.
* Include sufficient tests to ensure the project stays correct and robust. This includes both unit tests and integration tests.
* Please add documentation to docs/ if the PR modifies the user-facing behaviors of torch-spyre. It helps torch-spyre users understand and utilize the new features or changes.
* For installing in a dev environment, please install using requirements/dev.txt
