#!/usr/bin/env python3
"""Report which test tiers already have results for an artifact, for delta execution.

A run that needs `regression` when `integration` results already exist for the SAME
artifact only has to execute the configs regression adds. This script answers the
"what already exists" half; filter_configs.py --exclude-tiers does the selection.

Identity is what makes this safe, and it is structural rather than enforced here:
artifact_id embeds id12, the content-addressed digest, so different content is a
different artifact_id and cannot match. There is no way for these results to belong
to different bytes than the run about to execute.

The product repo does not know its own artifact_id -- the orchestrator writes that on
artifact_results, keyed by run_id -- so the artifact is reached the other way round:
--commit-sha resolves the artifacts whose runs carry this sha, and coverage is read
for those. A commit maps to several artifacts (one per component/arch), so the lookup
is scoped by arch and requires the run to have actually reported.

Fails OPEN by design. Any error -- unreachable ClickHouse, missing credentials, bad
response -- prints nothing and exits 0, so the caller runs the FULL tier. A delta is
a latency optimisation; degrading to a full run is always correct, while degrading to
a partial run silently under-tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

# Only the tier ladder. A config's labels also carry suite groups and one-off
# markers; those never denote a tier and must not suppress execution.
TIER_LABELS = ("smoke", "unit", "integration", "regression", "trunk")

# test_cases.tags are NAMESPACED (`testtype__trunk`, not `trunk`) -- the same
# namespace__value form the pytest tags use, which also carries op__ and dtype__ tags
# on the same array. Matching the bare tier name finds nothing, and a no-match is
# indistinguishable from "nothing covered", so this prefix is load-bearing.
TIER_TAG_PREFIX = "testtype__"

# Membership is read from the declared label sets, never inferred from a ladder:
# `integration ⊂ regression` holds only for configs that declare both. Measured on
# prod, integration ⊄ trunk had 616 case-level violations, so ladder inference would
# silently drop real tests.

# Coverage for the artifacts built from this commit, on this arch. Joined through
# artifacts.sources -- Array(Tuple(repo, git_ref, git_sha)) -- because a GHA run knows
# its sha, not its artifact_id (the orchestrator owns artifact_id, keyed by run_id).
# NOT artifacts.props['commit_sha']: that key does not exist, and a Map miss returns
# empty rather than erroring, so keying on it would silently report "nothing covered"
# forever -- a full run every time, which is safe but makes the feature dead code.
QUERY = """
SELECT DISTINCT tag
FROM {db}.test_case_runs AS r
INNER JOIN {db}.artifact_results AS ar USING (run_id)
INNER JOIN (
    SELECT artifact_id, tupleElement(s, 3) AS git_sha
    FROM {db}.artifacts
    ARRAY JOIN sources AS s
) AS a ON a.artifact_id = ar.artifact_id
INNER JOIN {db}.test_cases AS c USING (test_case_id)
ARRAY JOIN c.tags AS tag
WHERE a.git_sha = {commit_sha}
  AND ar.arch = {arch}
  AND ar.state IN ('passed', 'failed')
  AND tag IN {tiers}
  AND r.ts >= now() - INTERVAL {horizon} DAY
""".strip()


def _quote(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def covered_tiers(url: str, user: str, token: str, db: str,
                  commit_sha: str, arch: str, horizon: int,
                  timeout: int = 20) -> list[str]:
    sql = QUERY.format(
        db=db,
        commit_sha=_quote(commit_sha),
        arch=_quote(arch),
        tiers="(" + ",".join(_quote(TIER_TAG_PREFIX + t) for t in TIER_LABELS) + ")",
        horizon=int(horizon),
    )
    endpoint = url.rstrip("/") + "/?" + urllib.parse.urlencode(
        {"database": db, "default_format": "TSVRaw"}
    )
    req = urllib.request.Request(endpoint, data=sql.encode(), method="POST")
    if user:
        import base64
        cred = base64.b64encode(f"{user}:{token}".encode()).decode()
        req.add_header("Authorization", f"Basic {cred}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    found = set()
    for line in body.splitlines():
        tag = line.strip()
        if tag.startswith(TIER_TAG_PREFIX):
            found.add(tag[len(TIER_TAG_PREFIX):])
    # Bare tier names out: filter_configs.py matches config labels, which are bare.
    return [t for t in TIER_LABELS if t in found]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit-sha", required=True,
                    help="Commit the tests will run against. Coverage is only "
                         "inherited from runs of artifacts built from this exact sha.")
    ap.add_argument("--arch", required=True, help="Arch the tests RAN on")
    ap.add_argument("--test-type", default="",
                    help="Tier about to run; excluded from the output so a rerun "
                         "of the same tier is never suppressed by its own results.")
    ap.add_argument("--horizon-days", type=int,
                    default=int(os.getenv("SPYRE_REUSE_HORIZON_DAYS", "14")),
                    help="Ignore results older than this. Bounds how stale an "
                         "inherited pass can be.")
    ap.add_argument("--format", choices=["csv", "json"], default="csv")
    args = ap.parse_args()

    url = os.getenv("SPYRE_CH_URL", "").strip()
    db = os.getenv("SPYRE_CH_V2_DB", "").strip()
    if not url or not db:
        print("resolve_covered_tiers: SPYRE_CH_URL/SPYRE_CH_V2_DB unset -- "
              "running the full tier", file=sys.stderr)
        _emit([], args.format)
        return

    try:
        tiers = covered_tiers(
            url, os.getenv("SPYRE_CH_USER", "default"),
            os.getenv("SPYRE_CH_TOKEN", ""), db,
            args.commit_sha, args.arch, args.horizon_days,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"resolve_covered_tiers: lookup failed ({exc}) -- "
              f"running the full tier", file=sys.stderr)
        _emit([], args.format)
        return

    requested = args.test_type.strip()
    tiers = [t for t in tiers if t != requested]
    print(f"resolve_covered_tiers: already covered for {args.commit_sha[:12]} "
          f"[{args.arch}]: {tiers or 'nothing'}", file=sys.stderr)
    _emit(tiers, args.format)


def _emit(tiers: list, fmt: str) -> None:
    print(json.dumps(tiers) if fmt == "json" else ",".join(tiers))


if __name__ == "__main__":
    main()
