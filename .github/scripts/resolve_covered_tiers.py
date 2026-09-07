#!/usr/bin/env python3
"""Report which test tiers already have results for an artifact, for delta execution.

A run that needs `regression` when `integration` results already exist for the SAME
artifact only has to execute the configs regression adds. This script answers the
"what already exists" half; filter_configs.py --exclude-tiers does the selection.

Identity is what makes this safe, and it is structural rather than enforced here:
artifact_id embeds id12, the content-addressed digest, so different content is a
different artifact_id and cannot match. There is no way for these results to belong
to different bytes than the run about to execute.

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

# Membership is read from the declared label sets, never inferred from a ladder:
# `integration ⊂ regression` holds only for configs that declare both. Measured on
# prod, integration ⊄ trunk had 616 case-level violations, so ladder inference would
# silently drop real tests.

QUERY = """
SELECT DISTINCT tag
FROM {db}.test_case_runs AS r
INNER JOIN {db}.artifact_results AS ar USING (run_uid)
INNER JOIN {db}.test_cases AS c USING (test_case_id)
ARRAY JOIN c.tags AS tag
WHERE ar.artifact_id = {artifact_id}
  AND ar.arch = {arch}
  AND ar.state IN ('passed', 'failed')
  AND tag IN {tiers}
  AND r.ts >= now() - INTERVAL {horizon} DAY
""".strip()


def _quote(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def covered_tiers(url: str, user: str, token: str, db: str,
                  artifact_id: str, arch: str, horizon: int,
                  timeout: int = 20) -> list[str]:
    sql = QUERY.format(
        db=db,
        artifact_id=_quote(artifact_id),
        arch=_quote(arch),
        tiers="(" + ",".join(_quote(t) for t in TIER_LABELS) + ")",
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
    found = {line.strip() for line in body.splitlines() if line.strip()}
    return [t for t in TIER_LABELS if t in found]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-id", required=True,
                    help="Full v2 artifact_id: component|name|id12|arch")
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
            args.artifact_id, args.arch, args.horizon_days,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"resolve_covered_tiers: lookup failed ({exc}) -- "
              f"running the full tier", file=sys.stderr)
        _emit([], args.format)
        return

    requested = args.test_type.strip()
    tiers = [t for t in tiers if t != requested]
    print(f"resolve_covered_tiers: already covered for {args.artifact_id} "
          f"[{args.arch}]: {tiers or 'nothing'}", file=sys.stderr)
    _emit(tiers, args.format)


def _emit(tiers: list, fmt: str) -> None:
    print(json.dumps(tiers) if fmt == "json" else ",".join(tiers))


if __name__ == "__main__":
    main()
