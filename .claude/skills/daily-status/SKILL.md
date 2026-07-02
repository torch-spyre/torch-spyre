---
name: daily-status
description: "Generate a daily status update for any squad's Slack workflow. Pulls recent git activity, open PRs, and blockers to produce a formatted update matching the team's standup template."
---

# Daily Status Update Generator

Generate a status update for a squad Slack standup thread. Adapts to
whatever format the team uses (P/B/R, Yesterday/Today/Blockers, custom).

## Arguments

The user may pass arguments to customize the output:

- `/daily-status` — default P/B/R format, last 24 hours, current repo
- `/daily-status since Friday` — custom time window
- `/daily-status --repos torch-spyre,spyre-inference` — multi-repo
- `/daily-status --format ytb` — Yesterday/Today/Blockers format
- `/daily-status --label vLLM` — filter to issues with a specific label

## Steps

1. Identify the user by running `git config user.name` and
   `gh api user --jq .login`.

2. The repo scope defaults to the current repo from
   `gh repo view --json nameWithOwner --jq .nameWithOwner`. If the user
   specifies `--repos`, iterate over each. If the user specifies an org,
   query all repos under that org.

3. Use P/B/R format unless the user specifies otherwise:

   | Format flag | Sections |
   |---|---|
   | `pbr` (default) | Progress / Blockers / Review |
   | `ytb` | Yesterday / Today / Blockers |
   | `custom` | Ask the user for section names |

4. The time window defaults to "last 24 hours". Parse natural language
   like "since Friday", "this week", "last 3 days".

5. Gather activity by running the following, scoped to the time window
   and repo list:

   ```bash
   # Commits authored in the time window (across all branches)
   git log --all --author="$(git config user.name)" --since="<WINDOW>" \
     --format="%h %s" --no-merges

   # PRs authored (open or merged in the window)
   gh pr list --author="@me" --state all --limit 20 \
     --json number,title,state,updatedAt,headRefName \
     --jq '.[] | select(.updatedAt > "<ISO_TIMESTAMP>") | "\(.state) #\(.number): \(.title)"'

   # Issues assigned to the user (optionally filtered by label)
   gh issue list --assignee="@me" --state open --limit 20 \
     --json number,title,labels,updatedAt \
     --jq '.[] | "#\(.number): \(.title) [\(.labels | map(.name) | join(", "))]"'
   ```

   If `--label` is specified, add `--label "<LABEL>"` to the issue query.

6. Look for blockers:

   ```bash
   # PRs waiting on external review
   gh pr list --author="@me" --state open --limit 10 \
     --json number,title,reviewDecision,updatedAt \
     --jq '.[] | select(.reviewDecision != "APPROVED") | "#\(.number): \(.title) (review: \(.reviewDecision // "PENDING"))"'

   # CI failures on open PRs
   gh pr list --author="@me" --state open --limit 10 \
     --json number,title,statusCheckRollup \
     --jq '.[] | select(.statusCheckRollup | map(select(.conclusion == "FAILURE")) | length > 0) | "#\(.number): \(.title) — CI failing"'
   ```

   If nothing is blocking, write "None".

7. Find PRs where feedback was addressed and reviewers need to
   re-check:

   ```bash
   # PRs with changes requested that were pushed to recently
   gh pr list --author="@me" --state open --limit 10 \
     --json number,title,reviewDecision,updatedAt \
     --jq '.[] | select(.reviewDecision == "CHANGES_REQUESTED") | "#\(.number): \(.title)"'
   ```

8. Format the output as a message matching the chosen template,
   ready to paste into Slack:

   ### P/B/R format (default)
   ```
   **P:**
   - <progress bullets>

   **B:**
   - <blocker bullets or "None">

   **R:**
   - <review bullets or "None">
   ```

   ### Yesterday/Today/Blockers format
   ```
   **Yesterday:**
   - <what was done>

   **Today:**
   - <planned work — infer from open issues/PRs in progress>

   **Blockers:**
   - <blockers or "None">
   ```

   Rules:
   - One line per bullet. Use PR/issue numbers as references (e.g., "#1234").
   - No emojis unless the user requests them.
   - Keep it concise — the reader skims dozens of these in a thread.

9. Show the formatted update and ask if the user wants to adjust
   anything before posting.

## Notes

- If `gh` is not authenticated, instruct the user to run `gh auth login`.
- For multi-repo updates, group bullets by repo only if there are items
  from more than one repo. Otherwise skip the repo header.
- The skill works in any GitHub-hosted project, not just torch-spyre.
  It reads the repo context from the current working directory.
- The output is scoped to whoever is running it via their git/gh
  identity. No squad configuration needed.
