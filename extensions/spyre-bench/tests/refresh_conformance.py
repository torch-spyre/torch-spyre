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

"""Rewrite golden/ingest-ids.tsv from the REAL ingest: producer rows and the benchmark_id column.

    PYTHONPATH=extensions/clickhouse-ingest python tests/refresh_conformance.py [--check]

Rows with a producer are produced by calling the ingest's own entry builders on tiny synthetic inputs (`producer_rows`),
so the table records what each producer writes. `probe` rows are edited by hand; their ids come from
`BenchmarkId.derive` directly. Needs clickhouse-connect (the producers import it). A changed id is a changed series:
review the diff.
"""

import json
import sys
from pathlib import Path

TABLE = Path(__file__).resolve().parent / "golden" / "ingest-ids.tsv"
HEADER = "example_case\tproducer\tcomponent\tname\ttags\tdisc_keys\tdisc\tbenchmark_id"
COMPONENT = "ingest-demo"


def producer_rows():
    """[(example_case, producer, entry)] from the ingest's own builders; needs clickhouse-connect."""
    from spyre_clickhouse_ingest import bundle, results, vllm

    def perf(**rec):
        return results._bench_entries([{"operation_name": rec.pop("name"), **rec}])[0]

    op = {"record_type": "op", "run_mode": "op_benchmark"}

    def vl(test, **extra):
        row = {
            "extra": json.dumps({"test_name": test, "device": "dev", **extra}),
            "metric": "avg_latency",
            "actual": 1.0,
        }
        return vllm.bench_entries([row])[0]

    return [
        (
            "ingest-demo/op_matmul_m4096-n4096-k4096",
            "perf-kernel-op",
            perf(name="op-matmul", input_shapes="4096_4096_4096", **op),
        ),
        (
            "ingest-demo/op_matmul_m1-n4096-k4096",
            "perf-kernel-op",
            perf(name="op-matmul", input_shapes="1_4096_4096", **op),
        ),
        (
            "ingest-demo/op_softmax_fp16",
            "perf-kernel-model",
            perf(
                name="softmax", record_type="model", config_name="fp16", input_shapes="1_512_4096", run_mode="inference"
            ),
        ),
        (
            "",
            "perf-kernel-kernel",
            perf(name="op-matmul", input_shapes="4096_4096_4096", kernel_name="fused_add@1", is_total=0, **op),
        ),
        (
            "",
            "perf-kernel-total",
            perf(name="op-matmul", input_shapes="4096_4096_4096", kernel_name="Total", is_total=1, **op),
        ),
        ("ingest-demo/op_latency_fp16", "vllm", vl("latency_demo_tp1_in128_out16")),
        ("", "vllm-sweep", vl("serving_demo_c8_n64_len2048_seqs16_tp1_in128_out16")),
        (
            "ingest-demo/op_generic_add",
            "generic",
            bundle.generic_entries(
                [
                    {
                        "name": "op-generic",
                        "tags": ["domain__demo-ops"],
                        "props": {"iterations": 128, "op": "add"},
                        "measurements": {"m": 1},
                    }
                ]
            )[0],
        ),
        ("", "vllm-sweep-model", vl("serving_demo_c8_n64_len2048_seqs16_tp1_in128_out16", model="demo-model")),
        (
            "",
            "generic-unsorted-props",
            bundle.generic_entries(
                [
                    {
                        "name": "op-generic",
                        "props": {"op": "add", "iterations": 128, "batch": 4},
                        "measurements": {"m": 1},
                    }
                ]
            )[0],
        ),
        (
            "",
            "generic-bool-none",
            bundle.generic_entries(
                [{"name": "op-generic", "props": {"flag": True, "gap": None}, "measurements": {"m": 1}}]
            )[0],
        ),
        (
            "",
            "manual-unsorted-props",
            bundle._bench_entry({"name": "op-generic", "props": {"z": 1, "a": 2}, "measurements": {"m": 1}}),
        ),
        ("", "generic-empty-props", bundle.generic_entries([{"name": "op-generic", "measurements": {"m": 1}}])[0]),
        (
            "",
            "manual",
            bundle._bench_entry(
                {
                    "name": "op-generic",
                    "props": {"iterations": 128, "op": "add"},
                    "identity": ["op"],
                    "measurements": {"m": 1},
                }
            ),
        ),
        (
            "",
            "manual-name-only",
            bundle._bench_entry(
                {"name": "op-generic", "props": {"op": "add"}, "identity": [], "measurements": {"m": 1}}
            ),
        ),
    ]


def row_of(example, producer, entry):
    keys = list(entry["disc_keys"])
    disc = {k: entry["disc"][k] for k in keys if k in entry["disc"]}
    tags = ",".join(entry["tags"])
    return [
        example,
        producer,
        COMPONENT,
        entry["name"],
        tags,
        ",".join(keys),
        json.dumps(disc, ensure_ascii=False, separators=(",", ":"), default=str),
        "",
    ]


def derive(cols):
    from spyre_clickhouse_ingest.identity import BenchmarkId

    _, _, component, name, tags, keys, disc, _ = cols
    return BenchmarkId.derive(
        component, name, tags.split(",") if tags else [], json.loads(disc), keys.split(",") if keys else []
    )


def build():
    probes = [ln.split("\t") for ln in TABLE.read_text().splitlines()[1:] if ln.split("\t")[1] == "probe"]
    rows = [row_of(*r) for r in producer_rows()] + probes
    for cols in rows:
        cols[-1] = derive(cols)
    return "\n".join([HEADER, *("\t".join(c) for c in rows)]) + "\n"


def main(argv):
    text = build()
    if "--check" in argv:
        print("current" if text == TABLE.read_text() else "stale")
        return 0 if text == TABLE.read_text() else 1
    TABLE.write_text(text)
    print(f"wrote {text.count(chr(10)) - 1} rows to {TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
