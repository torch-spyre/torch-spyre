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

"""Validate spyre-bench catalogs: envelope + core schema, pack-typed blocks, cross-resource rules.

  spyre-bench validate CATALOG_DIR... [--pack DIR]... [--central DIR]... [--base-dir DIR] [--json]
--pack registers a pack at run time (packs not in packs/index.yaml). --central adds shared NodeClass/Environment
catalogs. --base-dir is the same tree at the merge base: a case whose benchmark_id changed under an unchanged case name
starts a new series, and a renamed Benchmark needs previous_names. Exit 0 valid, 2 invalid.
"""

import argparse
import json
import sys
from pathlib import Path

from spyre_bench import engine


def identity_changes(base, cur):
    """(warnings, notes): series that restart, and series that end."""
    old = {c["name"]: base.ids(c)["benchmark_id"] for c in base.cases}
    now = {b.name: b for b in cur.of("Benchmark")}
    renamed = {p: b.name for b in now.values() for p in b.spec.get("previous_names", [])}
    out = []
    for c in cur.cases:
        if c["name"] in old and old[c["name"]] != cur.ids(c)["benchmark_id"]:
            out.append(
                f"{c['where']}: identity changed under the same case name: a new series starts "
                f"(old benchmark_id {old[c['name']][:8]})"
            )
    notes = []
    for b in base.of("Benchmark"):
        if b.name in now:
            continue
        if b.name in renamed:
            out.append(
                f"Benchmark/{b.name} renamed to {renamed[b.name]}: benchmark_id changes (the name is hashed); "
                "history stays linked through previous_names"
            )
        else:
            notes.append(f"Benchmark/{b.name} removed or renamed without previous_names: its series ends")
    return out, notes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogs", nargs="+")
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--central", action="append", default=[])
    ap.add_argument("--base-dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if engine.Draft202012Validator is None:
        print("note: jsonschema not importable; using the stdlib schema-subset validator", file=sys.stderr)
    report = []
    for d in a.catalogs:
        cat = engine.Catalog([d], a.pack, a.central)
        warns = []
        if a.base_dir and not cat.errors:
            base = engine.Catalog([Path(a.base_dir) / Path(d).name], a.pack, a.central)
            changed, notes = identity_changes(base, cat)
            warns += changed + notes
        kinds = {}
        for doc in cat.docs:
            kinds[doc.kind] = kinds.get(doc.kind, 0) + 1
        report.append(
            {
                "catalog": d,
                "ok": not cat.errors,
                "kinds": kinds,
                "cases": len(cat.cases),
                "packs": {p: i["version"] for p, i in cat.reg.packs.items()},
                "errors": cat.errors,
                "warnings": warns,
            }
        )
    if a.json:
        print(json.dumps(report, indent=1))
    else:
        for r in report:
            print(
                f"{'PASS' if r['ok'] else 'FAIL'} {r['catalog']}: {sum(r['kinds'].values())} docs, {r['cases']} cases, "
                f"packs {r['packs']}"
            )
            for e in r["errors"]:
                print("  error: " + e)
            for w in r["warnings"]:
                print("  warning: " + w)
    return 0 if all(r["ok"] for r in report) else 2


if __name__ == "__main__":
    sys.exit(main())
