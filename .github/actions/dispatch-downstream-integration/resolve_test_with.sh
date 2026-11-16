#!/usr/bin/env bash
# Resolve the `Test-With:` companion for one downstream repo from a PR body; prints ref= / repository=.
# Grammar follows spyre-frameworks' parse_test_with.py: `[owner/]repo#N` or `<clone url>@<branch>`.
# Env: PR_BODY, COMPANION (repo name), DEFAULT_OWNER (default torch-spyre). Lines for other repos are ignored.
# `ref` is always a commit SHA and `repository` is owner/name, as the sibling workflows' checkouts require.
set -euo pipefail

owner_default="${DEFAULT_OWNER:-torch-spyre}"
max_lines=16
seen=0
found=0
ref=""
repository=""
declare -A done_tokens=()

fail() { echo "::error::$*" >&2; exit 1; }

# An unparsable line is fatal only when it names this leg's repo; otherwise it is another repo's
# (or an internal component's) and is ignored, so it cannot block an unrelated dispatch.
mentions_companion() { [[ "$1" =~ (^|[^A-Za-z0-9_.-])${COMPANION}(\.git)?($|[^A-Za-z0-9_.-]) ]]; }

while IFS= read -r line; do
  line="${line%$'\r'}"
  [[ "$line" =~ ^[[:space:]]*[Tt][Ee][Ss][Tt]-[Ww][Ii][Tt][Hh][[:space:]]*:[[:space:]]*(.*)$ ]] || continue
  rest="${BASH_REMATCH[1]}"
  if [[ ! "$rest" =~ ^([^[:space:]]+)([[:space:]]+#.*)?[[:space:]]*$ ]]; then
    mentions_companion "$rest" && fail "malformed Test-With line: '${rest}'"
    echo "ignoring unparsable Test-With line: '${rest}'" >&2
    continue
  fi
  tok="${BASH_REMATCH[1]}"
  [ -z "${done_tokens[$tok]:-}" ] || continue
  done_tokens[$tok]=1
  seen=$((seen + 1))
  [ "$seen" -le "$max_lines" ] || fail "more than ${max_lines} Test-With lines"

  if [[ "$tok" =~ ^https?://([^/]+)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)@(.+)$ ]]; then
    host="${BASH_REMATCH[1]}"
    owner="${BASH_REMATCH[2]}"
    name="${BASH_REMATCH[3]%.git}"
    branch="${BASH_REMATCH[4]}"
    [ "$host" = github.com ] || continue
    [ "$name" = "$COMPANION" ] || continue
    [ "$owner" = "$owner_default" ] || fail "Test-With '${tok}': repository must be in ${owner_default}"
    [[ "$branch" =~ ^[A-Za-z0-9._/-]+$ && "$branch" != -* && "$branch" != *..* ]] \
      || fail "Test-With '${tok}': invalid branch name"
    cand_ref=$(gh api "repos/${owner}/${name}/commits/${branch}" --jq .sha) \
      || fail "Test-With '${tok}': branch not found"
    cand_repo="${owner}/${name}"
  elif [[ "$tok" =~ ^(([A-Za-z0-9_.-]+)/)?([A-Za-z0-9_.-]+)#([0-9]+)$ ]]; then
    name="${BASH_REMATCH[3]}"
    [ "$name" = "$COMPANION" ] || continue
    owner="${BASH_REMATCH[2]:-$owner_default}"
    [ "$owner" = "$owner_default" ] || fail "Test-With '${tok}': PR references must be in ${owner_default}"
    pr="${BASH_REMATCH[4]}"
    info=$(gh api "repos/${owner}/${name}/pulls/${pr}" \
      --jq '[.state, (.merged // false | tostring), (.head.sha // ""), (.head.repo.full_name // "")] | @tsv') \
      || fail "Test-With '${tok}': PR not found"
    IFS=$'\t' read -r state merged cand_ref cand_repo <<<"$info"
    if [ "$merged" = true ]; then
      echo "Test-With '${tok}': already merged; testing main" >&2
      continue
    fi
    if [ "$state" != open ] || [ -z "$cand_ref" ] || [ -z "$cand_repo" ]; then
      fail "Test-With '${tok}': PR is not open or has no head repository"
    fi
  else
    mentions_companion "$tok" && fail "malformed Test-With line: '${tok}'"
    echo "ignoring unparsable Test-With line: '${tok}'" >&2
    continue
  fi

  found=$((found + 1))
  [ "$found" -le 1 ] || fail "more than one Test-With companion for ${COMPANION}"
  ref="$cand_ref"
  repository="$cand_repo"
done <<<"${PR_BODY:-}"

echo "ref=${ref}"
echo "repository=${repository}"
