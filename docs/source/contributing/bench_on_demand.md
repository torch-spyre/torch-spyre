# Benchmarks on demand for a pull request

Ask for a benchmark run of a pull request by commenting on it. The run is advisory: it never blocks a merge and
never changes a required check.

## Asking

`/bench` still works as a deprecated spelling.

Comment on the pull request (you need write access to the repository; from a fork add `sha=<head commit>`):

```text
/bench sanity
/bench ops@sanity reuse=prefer
/bench op_add_fp16,op_mul_fp16 force=op_add_fp16
/bench confirm        # run a plan that asked for confirmation
/spyre-bench cancel         # stop the run for this pull request
```

Applying the `bench` label has the same effect as `/spyre-bench sanity`; the label is removed again afterwards.

The first token is the selection: a tier (`sanity`), suites with an optional tier (`ops+more@sanity`), a list of case
names, or `#name` for a named selection. The flags and their limits are in the engine documentation
(`extensions/spyre-bench/docs/README.md`, "On-demand benchmarks for a PR"). On this host only the `sanity` tier is
available, and `reuse=never`, `force=all`, `profile`, `env` and `extended` are refused.

The workflow answers with one comment that it edits in place: how many cases will run, be reused or be skipped, the
estimate in device-hours, what you asked for versus what will run, and the reason for every skip. A plan above half of
the limit waits for `/spyre-bench confirm`; a plan above the limit does not run. When the run finishes a second comment gives
the outcome of each stage. No measured values are posted on a public pull request.

This workflow route measures on x86 only. The internal comment-bot to Jenkins route (the same `/spyre-bench` command) covers amd64, ppc64le and s390x.

## Switches (shipped off)

The route is dark until two things are on together: the repository variable `BENCH_ONDEMAND_ENABLED` is `true` and the
engine's `ondemand_policy.json` has `enabled` true (the internal poller has a byte copy of that file and flips with it).
`enforce_budgets` in the same file ships off: the device-hour limits, the unknown-cost rule and the per-PR window and
supersede counting then only report in the preview that a request "would have been refused" and it proceeds. Tier
limits, the head pin, the fork `sha=`, writer-only trigger and the start-of-run re-check are always enforced.

## How it works

1. `bench-pr-comment` starts from the default branch on `issue_comment` or `pull_request_target`. A first job reads the
   commenter's permission from the API; for anyone without write access nothing else runs and nothing is posted.
2. The second job installs the engine from the default branch (never from the pull request), reads the head sha and
   head repository from the API, then runs `spyre-bench ondemand parse` and `authorize`.
3. For an allowed request it fetches only `benchmarks/` of the head, as data, in a step that has no token, and runs
   `spyre-bench plan` on it to produce the preview. A second `authorize` checks the plan itself: its estimate, and the
   tier of every case that would run. Nothing from the head is executed in this workflow.
4. It dispatches `bench-on-demand` (same-repository head, results stored in the dev instance) or
   `bench-on-demand-fork` (fork head), pinned to the head sha. A newer request for the same pull request cancels the
   older run (one concurrency group per pull request, shared by both workflows).

## Fork pull requests

A fork head runs only when a user with write access names the commit they reviewed:
`/spyre-bench sanity sha=1a2b3c4d5e6f` (at least 12 characters). The prefix must match the head read from the API at that moment, so a force-push just
before the comment cannot be approved by accident, and a push after it cancels the approval. The fork workflow refers
to no secrets (`grep -n 'secrets\.' .github/workflows/bench-on-demand-fork.yml` stays empty), does not derive an
artifact identity, does not ingest, and comments counts only.

## Security model

| Threat | Control |
|---|---|
| A stranger comments `/spyre-bench` on a public pull request | A gate job reads the commenter's permission from the collaborator API (not the event's `author_association`); without write access nothing else runs and nothing is posted. |
| A fork head runs code on the shared runner pools | A write-access user must name the head commit (`sha=`); the request is bound to that sha, a moved head is refused, and a new comment is a new request and a new check. |
| The fork changes the engine, the workflow or the Makefile to get secrets | The workflows and the engine come from the default branch. The fork workflow references no secrets, ingests nothing, and its checkouts do not keep credentials. Its outcomes are read by a separate reporting job that checks out nothing. |
| Comment text becomes a command, a mention or a link | The comment is passed through an environment variable and parsed by a strict grammar (names are letters, digits, `.`, `_`, `-`); every error and every string derived from a comment or a plan is stripped of control and bidirectional characters, backticks, tags, pipes, link and image syntax and issue references, `@` is neutralised, and URL schemes are broken, before it is posted. |
| The fork head is parsed with a token in reach | The plan step has no token in its environment, symlinks in the fetched tree are removed first, and plan errors are sanitized before they are posted. |
| A case or named selection from a higher tier slips past the public limit | The limit is checked again on the plan: any case that would run with a tier the host does not allow denies the request. |
| One user floods the runners | One run per pull request at a time, a per-request device-hour limit, and a per-user hourly limit counted from this workflow's own `/spyre-bench` runs. |
| A label or comment races a force-push | The sha is read from the API, planned, and read again right before dispatch; a different sha means no dispatch. |
| The dispatch input is forged by someone who can run workflows by hand | The measuring workflows re-check that the ref is a full sha and re-validate every `BENCH_*` key and value against an allow-list before using them. |
| A stale or forged confirm | `/spyre-bench confirm` reads only the bot's own preview comment, not older than a day, and re-authorizes as the original requester with the recorded sha (and recorded approval for a fork); the confirmer needs write access. |

A same-repository head is pushed by someone with write access, so it runs with the dev store credentials like any
workflow that branch could carry; it needs no `sha=`. A fork head's code does run on a shared card runner: the
workflow holds no secrets, but isolation of the runner pod itself (service account, network, shared caches) is a
property of the runner set and is not established by these workflows.

The plan shown in the preview is made from the head's `benchmarks/` with the same environment name as the measuring
run, but the measuring run plans again with the head's own Makefile; for a fork head the tier and budget limits
therefore bound an honest request and are not a sandbox: the sandbox is the named commit, the absence of secrets, and
the runner.

A preview is spent once the bot has said the run started, finished or was cancelled, so `/spyre-bench confirm` cannot start
it a second time, and confirming keeps the original request time and requester.

The limits (tiers, device-hour budgets, rates) are set in one file, `extensions/spyre-bench/spyre_bench/ondemand_policy.json`; the workflows read them through the engine and hold no numbers. This route has no store credentials, so it does not write `ci_requests`; it counts a user's recent runs from its own run list.

Cost bounds on this route: a run that declares no `run.est` for a case is refused (its cost is unknown), the per-PR
24-hour window is counted from the measuring runs themselves (each charged the whole per-request limit, because their
own estimate is not known to the request job), and the runner job's 240-minute timeout on one device is the hard
bound that equals the public per-request limit. Each measuring run first asks the API whether the pull request is
still open, still at the requested head, and was requested less than a day ago.

Fork heads run on the default runner labels of the measuring workflow, the same pools `/spyre-test` uses for fork
heads. The repository variable `BENCH_FORK_RUNNER_LABEL` is optional: when set it replaces the card-size label, so the
owner can move fork runs to a separate runner set later without changing the workflows.

Residual risk, stated plainly: a fork's Makefile and `mk/` run on the same pools as same-repository jobs, so isolation
of the runner pod (service account, network, shared caches, the prebaked venv) is a property of those pools and is not
established by these workflows. What bounds it: only a user with write access can trigger a run, and only after naming
the exact head commit (`sha=`); the request is bound to that sha and re-checked when the job starts; the fork workflow
references no secrets and stores nothing; and the pods are ephemeral where the runner set is per-identity.

Limits of this route today: clean (exclusive node) runs are not available here; reuse is looked up when the run
starts, so the preview assumes nothing is reusable; the per-pull-request daily device-hour window needs a store and is
enforced by the internal route only; two requests that race each other are resolved by which dispatch lands last; the
engine's Python dependencies are pinned for the two parsers only.
