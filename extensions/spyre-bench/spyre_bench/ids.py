# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""spyre-bench ids: the stored benchmark_id of each case, or with --inputs the ingest derivation inputs.

Engine-mode cases need no ingest package. Cases that declare `identity: {mode: ingest}` need `spyre-bench[ingest]`
for their id (exit 2 with the install hint otherwise); --inputs and --adapter never need it.
"""

import argparse
import json
import sys

from spyre_bench import engine, ingest_adapter


def rows(cat, names, inputs):
    out = []
    for c in cat.cases:
        if names and c["name"] not in names and c["benchmark"] not in names:
            continue
        a = cat.ingest_inputs(c)
        row = {"case": c["name"], "mode": "ingest" if a else "engine"}
        if inputs:
            row["inputs"] = a or {
                k: v for k, v in cat.identity(c).items() if k in ("component", "name", "tags", "disc")
            }
        else:
            try:
                row.update(cat.ids(c))
            except ingest_adapter.IngestUnavailable as x:
                row["error"] = str(x)
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spyre-bench ids")
    ap.add_argument("names", nargs="*", help="case or Benchmark names (default: every case)")
    ap.add_argument("--catalog")
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--central", action="append", default=[])
    ap.add_argument("--inputs", action="store_true", help="print the derivation inputs instead of the ids")
    ap.add_argument("--adapter", action="store_true", help="print the ingest adapter status and exit")
    ap.add_argument(
        "--allow-unverified", action="store_true", help="--adapter: exit 0 although a profile is unverified"
    )
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.adapter:
        info = ingest_adapter.info()
        status = {}
        for profile in engine.ID_PROFILES:
            try:
                status[profile] = engine.check_profile(profile)
            except ingest_adapter.IngestUnavailable as x:
                status[profile] = f"MISMATCH: {x}"
        print(json.dumps({**info, "profile_comparison": status}, indent=1))
        bad = ("MISMATCH",) if a.allow_unverified else ("MISMATCH", "unverified")
        return 0 if info["available"] and not any(v.startswith(bad) for v in status.values()) else 2
    if not a.catalog:
        ap.error("--catalog is required")
    cat = engine.Catalog([a.catalog], a.pack, a.central)
    if cat.errors:
        print("\n".join(cat.errors), file=sys.stderr)
        return 2
    known = {c["name"] for c in cat.cases} | {c["benchmark"] for c in cat.cases}
    unknown = sorted(set(a.names) - known)
    if unknown:
        print(f"error: no case or Benchmark named {unknown}", file=sys.stderr)
        return 2
    out = rows(cat, set(a.names), a.inputs)
    failed = [r for r in out if "error" in r]
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        for r in out:
            shown = "ERROR (see below)" if "error" in r else r.get("benchmark_id") or json.dumps(r.get("inputs"))
            print(f"{r['case']}\t{r['mode']}\t{shown}")
    if failed:
        print(f"error: {failed[0]['error']}", file=sys.stderr)
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
