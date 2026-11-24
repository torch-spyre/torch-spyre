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

"""Show cases with every reference resolved: benchmark_id and its inputs, run-level props, the Environment request,
series and reuse keys, and the plan record. A status view; never written to source.

  spyre-bench explain NAME --catalog DIR [--pack DIR]... [--central DIR]... [--environment ENV] [--subject ARTIFACT_ID]
             [--set PATH=VALUE]... [--purpose P] [--dataset-meta-dir DIR | --dataset-store STORE] [--json]
NAME is a case name or a Benchmark name (all its cases). --set shows the ephemeral variant and its identity diff.
A dataset input is labelled by its slug; with --dataset-meta-dir or --dataset-store (a Store of the catalog) its
`<slug>/dataset.json` is read and the real name is shown only when that file says `public: true`.
"""

import argparse
import json
import sys

from spyre_bench import dataset_meta, engine


def leg_of(view):
    return {
        "arch": "x86_64",
        "devices": view["device_totals"],
        "nodes": view["nodes"],
        "reference_node": view["reference_node"],
    }


def dataset_labels(cat, a):
    """Labels over the store the flags name, or None (slugs only). Offline by default: nothing is read unasked."""
    if isinstance(getattr(a, "dataset_meta_dir", None), str):
        return dataset_meta.Labels(dataset_meta.DirSource(a.dataset_meta_dir))
    if isinstance(getattr(a, "dataset_store", None), str):
        store = cat.by.get(("Store", a.dataset_store))
        if store is None:
            raise dataset_meta.DatasetMetaError(f"--dataset-store {a.dataset_store!r}: no Store of that name")
        return dataset_meta.Labels(dataset_meta.source_for_store(store.spec))
    return None


def input_doc(cat, i, labels):
    ids, sel, props = cat.input_identity(i)
    doc = dict(zip(("asset_ids", "select", "props"), (ids, sel, props)), name=i["name"])
    if "dataset" in i:
        slug = cat.dataset_slug(i["dataset"])
        if labels:
            dataset_meta.check_assets(labels.meta(slug), cat.dataset_asset_specs(i["dataset"]))
        doc["dataset_label"] = labels.label(slug) if labels else slug
    return doc


def explain(cat, name, a):
    labels = dataset_labels(cat, a)
    cases = [c for c in cat.cases if name in (c["name"], c["benchmark"])]
    if not cases:
        raise SystemExit(f"no case or Benchmark named {name!r}")
    sets = getattr(a, "set", None) or []
    if sets:
        if len(cases) != 1:
            raise SystemExit("--set needs a single case name")
        variant, diff = cat.variant(cases[0]["name"], sets)
        cases = [variant]
    out = []
    for c in cases:
        ident = cat.identity(c)
        env_name = a.environment or c["environment"]
        view = cat.env_view(env_name) if env_name else None
        if view is None and env_name:
            raise SystemExit("\n".join(cat.errors))
        req = {"purpose": a.purpose, "environment": env_name, "selection": {"cases": [c["name"]], "type": c["type"]}}
        if a.subject:
            req["subject"] = {"artifact_id": a.subject}
        rec = None
        if not sets:
            try:
                rec = engine.plan(cat, req, leg_of(view) if view else {"arch": "x86_64"})["cases"][0]
            except (engine.PlanError, engine.IngestUnavailable) as x:
                rec = {"error": str(x)}
        req_c = cat.case_metrics(c)
        series = {
            k: view["labels"][k] for k in cat.project.spec.get("series_labels", []) if view and k in view["labels"]
        }
        ingest = cat.ingest_inputs(c)
        try:
            ids = cat.ids(c)
        except engine.IngestUnavailable as x:
            ids = {"benchmark_id": None, "id_unavailable": str(x)}
        doc = {
            "case": c["name"],
            "benchmark": c["benchmark"],
            "entry_file": c["entry_file"],
            "owner": c["owner"],
            **ids,
            "identity": ident,
            **({"ingest_identity": ingest} if ingest else {}),
            **({"est_hours": c["est_hours"]} if c.get("est_hours") is not None else {}),
            "run_props": cat.run_props(c),
            "system": {"name": c["system"], **c["sys"]},
            "workload": c["workload"],
            "load": c["load"],
            "metrics": {"required": req_c[0], "headline": req_c[1]},
            "profile": c["profile"],
            "run": c["run"],
            "inputs": [input_doc(cat, i, labels) for i in c["workload"].get("inputs", [])],
        }
        doc["validity"] = cat.validity_view(c, a.purpose, cat.effective("Environment", env_name) if env_name else None)
        if view:
            doc["environment"] = view
            doc["series_key"] = (
                engine.series_key(doc["benchmark_id"], series) if series and doc["benchmark_id"] else None
            )
            doc["reuse_key"] = {
                "benchmark_id": doc["benchmark_id"],
                "subject_id": a.subject or "<artifact id at plan time>",
                "environment": "env_id (exact, known after placement); label-key relaxation by Selection; an Analysis may relax env.id for compare",
            }
        if sets:
            doc["identity_diff"] = diff
        else:
            doc["plan"] = rec
        out.append(doc)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--central", action="append", default=[])
    ap.add_argument("--environment")
    ap.add_argument("--subject")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--purpose", default="investigate")
    ap.add_argument("--json", action="store_true")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dataset-meta-dir", help="directory holding <slug>/dataset.json")
    src.add_argument("--dataset-store", help="a Store of the catalog that holds <slug>/dataset.json")
    a = ap.parse_args(argv)
    cat = engine.Catalog([a.catalog], a.pack, a.central)
    if cat.errors:
        print("\n".join(cat.errors), file=sys.stderr)
        return 2
    bad = [s for s in a.set if "=" not in s]
    if bad:
        print(f"error: --set needs PATH=VALUE, got {bad[0]!r}", file=sys.stderr)
        return 2
    try:
        docs = explain(cat, a.name, a)
    except (KeyError, ValueError, TypeError, AttributeError, engine.PlanError, dataset_meta.DatasetMetaError) as x:
        print(f"error: {str(x).splitlines()[0] if str(x) else type(x).__name__}", file=sys.stderr)
        return 2
    print(
        json.dumps(docs, indent=1, default=str) if a.json else "---\n".join(engine.dump_yaml(d) for d in docs), end=""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
