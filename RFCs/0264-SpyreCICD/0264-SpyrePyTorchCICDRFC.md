# [RFC] PyTorch CI/CD for IBM Spyre

**Authors:**
* @tharapalanivel

## **Summary**
This RFC document aims to propose and discuss the CI/CD required for the upstreaming of IBM Spyre support in PyTorch. It describes an approach to incrementally enable running open-source PyTorch tests within scope on Spyre, starting nightly and progressively switching to running on every PR without blocking merging. Additionally, the goal is to collect the minimal signal and roundtrip times for that signal and have Spyre integration on Pytorch's overall health indicator be available on [PyTorch HUD](https://hud.pytorch.org).

## **Motivation**
CI/CD is a crucial foundation to any software development process and it is necessary that the latest PyTorch stack is continuously verified for support on IBM Spyre to provide the advantage of the hardware to the larger PyTorch community. Moreover with CI/CD enabled for the IBM Sypre software stack, it ensures that code changes are validated early and often in a timely manner through automated builds and tests which will greatly reduce errors and speed up development and testing velocity.

## **Proposed Implementation**

### Internal Spyre Software stack CI/CD
For implementing the internal CI/CD pipeline for the Spyre software stack that will be triggered on every PR to the internal stack, the main steps are to:
1. connect our strategic version control system, Git, with our choice of CI/CD platform which will be Jenkins internally
2. implement process for building the internal software stack to create the spyre runtime image
3. run unit tests for all internal Spyre software components
4. install `torch-spyre` and other dev/test dependencies on top of the runtime image and run `torch-spyre` unit and integration tests
5. run PyTorch tests (see [PyTorch CI/CD on Spyre](#pytorch-cicd-on-spyre) for additional details)
6. set up webhooks in relevant internal Git repositories to trigger the Jenkins pipelines to kick off #2-#5

In the case of creating a semantic versioned release of the Spyre software stack, it is also required to automate the upstream push of the spyre runtime image from step #2 to ICCR/Quay if all tests pass.

### torch-spyre CI/CD
A similar open-source pipeline for building, testing and releasing `torch-spyre/torch-spyre` using Github Actions on every PR merged to `main` will include the following workflows:
1. run code linter of choice (`ruff`) which includes sorting, spell check, formatting, etc. to ensure code quality and consistency
2. run all `torch-spyre` unit tests through Github Actions that do not require device access
3. run PyTorch tests (see [PyTorch CI/CD on Spyre](#pytorch-cicd-on-spyre) for additional details)
4. kick off workflow to trigger unit tests that require Spyre hardware based on the latest spyre runtime image
5. create and verify release artifacts by building source dist and wheel, validating artifacts with various tools and uploading artifacts to GHA
6. push release artifacts to Test PyPI

When a new Github release is published, an additional step of pushing the release artifacts to Production PyPI will also be appended.

### PyTorch CI/CD on Spyre

To enable PyTorch CI/CD for Spyre, the reuse of existing PyTorch CI/CD infrastructure will be maximized and workflows from other hardware accelerators will be mirrored. This includes adopting Docker for builds, using label-based triggers for CI/CD pipelines, and similar patterns. There are 3 factors that progressively increase throughout the overall implementation:
1. **Infrastructure/Hardware:** OpenShift cluster with Control Plane (3 nodes virtual or baremetal) and 256 AIU 1.0 cards, gradually enabling upto 2048 cards
2. **Test Coverage:** Internal tests and limited PyTorch CI tests to running 80% in-scope PyTorch tests (out-of-scope tests include all training tests and GPU specific tests eg. cuda tests)
3. **Frequency of CI being run/triggered:** run nightly to on every PR (without blocking merging ability)

Additionally, there are also 2 versions of the overall stack that will be built and tested through CI/CD:
* **latest Spyre SW stack + torch-spyre main + pinned PyTorch version:** on PR workflows to the internal or `torch-spyre` stack, setup a Jenkins pipeline that will rebuild the spyre runtime image with the code changes from the PR and a pinned PyTorch version and run all enabled PyTorch tests currently in scope for Spyre
* **latest Spyre SW stack + torch-spyre main + PyTorch main:** run PyTorch tests nightly or weekly based on the latest internal and PyTorch stacks

### Implementation Phases
A phased approach will be used for the overall CICD implementation covering all the 3 sections/CICD pipelines above:
* Stage 0:
  * Image build pipeline exists for internal spyre software stack, image gets pushed to ICCR if all existing internal unit tests and `torch-spyre` tests w/wo device succeed
  * Lint, testing and packaging workflows are added for `torch-spyre` (pushing to pypi is not a requisite here)
  * Infrastructure with 24 AIU 1.0 cards is validated in a dedicated OpenShift cluster
* Stage 1:
  * PR test pipeline exists for internal spyre software stack, PRs going into `torch-aiu-main` branches of internal stack will only get merged if all existing internal unit tests and `torch-spyre` tests w/wo device succeed
  * In `torch-spyre` testing workflows w/wo device are enabled, lint/test passing gates PR merges and pypi automation is added
  * 5% in-scope PyTorch CI tests are enabled for Spyre and run nightly using 128 AIU 1.0 cards
* Stage 2:
  * 25% in-scope PyTorch CI tests are enabled for Spyre and run nightly using 512 AIU 1.0 cards
* Stage 3:
  * Tests added to `torch-spyre` using device emulation will be enabled to run on every `torch-spyre` PR
  * 50% in-scope PyTorch CI tests are enabled for Spyre and run 3-4 times a day using 1024 AIU 1.0 cards
* Stage 4:
  * 80% in-scope PyTorch CI tests are enabled for Spyre and run 3-4 times a day using 2048 AIU 1.0 cards
* Stage 4:
  * 80% in-scope PyTorch CI tests are run on every PR to the internal Spyre stack and `torch-spyre` w/o blocking merge and signal is integrated to PyTorch HUD
  
## **Metrics**

Some metrics borrowed from the Accelerator Group's Quality Criteria for CI/CD are:
* **Accessibility:** CI/CD is publicly accessible and test results are transparent to the PyTorch open-source community
* **CI:** CI gates every PR relevant for the proposed compute platform by executing both proposed compute platform specific tests and tests for other affected platforms (if affected by the change)
* **CD:** CD supports preview nightly builds and release builds
* **Tests Stability:** CI/CD configuration has less than 5% tests marked as flaky
* **CI/CD Stability:** CI/CD has minimal outages with the recovery period not longer than 3 days

Additional points to consider are:
* the time taken for a single CI run and of particular concern is the on a per commit cycle time which should be within a reasonable range to provide good developer experience
* CI/CD covers the models and deployment patterns that are of priority to IBM product releases and other key users
* a separate nightly/weekly test bucket is created if it is deemed necessary to run all the Spyre tests required but cannot be run on every PR due to time or hardware limitations

## **Drawbacks**
This approach has multiple dependencies, namely:
* the readiness of the PyTorch-native Spyre SW stack
* the availability of unit, integration and functional tests at various levels of the stack to run
* the availability of the hardware (i.e. Openshift Cluster with AIU cards, enough memory, etc.) and CI/CD platform (e.g. Jenkins with sufficient CPU, memory and # of pods access)

## **Alternatives**
A potential alternative in terms of the chosen CI/CD platform for the internal implementation is SPS (Secure Pipelines Service) rather than Jenkins. However, the current Spyre CI/CD team's expertise lies in Jenkins and should IBM recommend migrating all internal CI/CD workloads to SPS, this decision will be revisited.

Overall enabling CI/CD as outlined above is fundamental for the continuous support of PyTorch on Spyre, it is required to provide timely and stable releases to IBM products running on Spyre hardware accelerator and PyTorch community users experimenting with Spyre.

## **Prior Art**
<!--
Discuss prior art (both good and bad) in relation to this proposal:
* Does this feature exist in other libraries? What experience has their community had?
* What lessons can be learned from other implementations of this feature?
* Published papers or great posts that discuss this
-->
TODO

## **How we teach this**
<!--
* What names and terminology work best for these concepts and why? How is this idea best presented?
* Would the acceptance of this proposal mean the PyTorch documentation must be re-organized or altered?
* How should this feature be taught to existing PyTorch users?
-->
TODO

## **Unresolved questions**
<!--
* What parts of the design do you expect to resolve through the RFC process before this gets merged?
* What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
* What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?
-->

## Resolution

TBD

### Level of Support
<!--
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.
-->

#### Additional Context

### Next Steps

#### Tracking issue
<!---
<github issue URL>
-->

#### Exceptions
