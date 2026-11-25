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

"""Repeated runs through the vLLM ingest: one row per (benchmark, repeat), one identity row.

The engine writes each repeat into its own `repeat-NNN` subdir of the case's raw dir. A flat
directory is the pre-repeat shape and must still ingest exactly as it did.
"""

import json

from spyre_clickhouse_ingest import vllm

TEST = "serve_granite33-8b_tp4_cics"


def _native(path, test_name=TEST, throughput=100.0):
    path.write_text(
        json.dumps(
            {
                "date": "2026-11-25",
                "model_id": "ibm-granite/granite-3.3-8b-instruct",
                "request_throughput": throughput,
                "mean_ttft_ms": 12.5,
                "num_prompts": 100,
            }
        )
    )


def _rows(results_dir):
    return vllm.extract_rows(str(results_dir), "main", "881a59d2", "", "0", "wf", 0)


def _extras(rows):
    return [json.loads(r["extra"]) for r in rows]


# ── discovery ──


def test_repeat_of_reads_the_containing_dir():
    assert vllm.repeat_of("/x/raw/repeat-003/serve_a.json") == "3"
    assert vllm.repeat_of("/x/raw/repeat-012/serve_a.json") == "12"
    assert vllm.repeat_of("/x/raw/serve_a.json") == ""
    # Not a repeat dir: a benchmark may legitimately sit under any other folder name.
    assert vllm.repeat_of("/x/raw/repeat-abc/serve_a.json") == ""
    assert vllm.repeat_of("/x/repeats/serve_a.json") == ""


def test_results_json_finds_both_levels(tmp_path):
    _native(tmp_path / f"{TEST}.json")
    (tmp_path / "repeat-001").mkdir()
    _native(tmp_path / "repeat-001" / f"{TEST}.json")
    (tmp_path / f"{TEST}.pytorch.json").write_text("[]")
    (tmp_path / "repeat-001" / f"{TEST}.host_resources.json").write_text("")

    native, pytorch, host = vllm.results_json(str(tmp_path))
    assert len(native) == 2, native
    assert len(pytorch) == 1 and len(host) == 1
    # A .pytorch.json / .host_resources.json is never also counted as native.
    assert not any(
        p.endswith((".pytorch.json", ".host_resources.json")) for p in native
    )


# ── extraction ──


def test_each_repeat_keeps_its_own_measurement(tmp_path):
    for n, tps in ((1, 100.0), (2, 110.0), (3, 105.0)):
        d = tmp_path / f"repeat-{n:03d}"
        d.mkdir()
        _native(d / f"{TEST}.json", throughput=tps)

    extras = _extras(_rows(tmp_path))
    got = {
        (e["repeat"], r["metric"]): r["actual"]
        for r, e in zip(_rows(tmp_path), extras)
        if r["metric"] == "request_throughput"
    }
    assert got == {
        ("1", "request_throughput"): 100.0,
        ("2", "request_throughput"): 110.0,
        ("3", "request_throughput"): 105.0,
    }


def test_a_flat_directory_carries_no_repeat(tmp_path):
    _native(tmp_path / f"{TEST}.json")
    assert all("repeat" not in e for e in _extras(_rows(tmp_path)))


# ── entries and row grouping ──


def test_entries_carry_the_repeat_but_never_hash_it(tmp_path):
    for n in (1, 2):
        d = tmp_path / f"repeat-{n:03d}"
        d.mkdir()
        _native(d / f"{TEST}.json", throughput=100.0 + n)

    entries = vllm.bench_entries(_rows(tmp_path))
    assert {e["repeat"] for e in entries} == {"1", "2"}
    # The repeat must stay out of the identity: disc_keys is positional, so a new key there
    # would re-key every vLLM benchmark already stored.
    for e in entries:
        assert "repeat" not in e["props"]
        assert "repeat" not in e["disc"]
        assert vllm.REPEAT_PROP not in e["props"]
    assert {tuple(e["disc_keys"]) for e in entries} == {vllm.BENCH_ID_KEYS}


def test_identity_is_the_same_with_and_without_repeats(tmp_path):
    """The whole point of repeats landing this way: stored history must not restart."""
    flat = tmp_path / "flat"
    flat.mkdir()
    _native(flat / f"{TEST}.json")

    nested = tmp_path / "nested"
    (nested / "repeat-001").mkdir(parents=True)
    _native(nested / "repeat-001" / f"{TEST}.json")

    def identity(d):
        return {
            (e["name"], tuple(sorted(e["props"].items())), tuple(e["disc_keys"]))
            for e in vllm.bench_entries(_rows(d))
        }

    assert identity(flat) == identity(nested)


def test_write_benchmarks_inserts_once_per_repeat(tmp_path, monkeypatch):
    for n in (1, 2, 3):
        d = tmp_path / f"repeat-{n:03d}"
        d.mkdir()
        _native(d / f"{TEST}.json", throughput=100.0 + n)

    calls = []
    monkeypatch.setattr(vllm, "tables_present", lambda *a, **k: True)
    monkeypatch.setattr(vllm, "benchmarks_already_ingested", lambda *a, **k: False)
    monkeypatch.setattr(
        vllm,
        "insert_benchmarks",
        lambda c, db, comp, rid, entries, **kw: (
            calls.append((kw.get("row_props") or {}).get(vllm.REPEAT_PROP)),
            len(entries),
        )[1],
    )
    vllm.write_benchmarks(None, "db", _rows(tmp_path), "rid")
    assert sorted(calls) == ["1", "2", "3"], calls


def test_write_benchmarks_unrepeated_stamps_no_repeat_prop(tmp_path, monkeypatch):
    """The regression guard: every other product's perf leg still takes the single-group path."""
    _native(tmp_path / f"{TEST}.json")

    calls = []
    monkeypatch.setattr(vllm, "tables_present", lambda *a, **k: True)
    monkeypatch.setattr(vllm, "benchmarks_already_ingested", lambda *a, **k: False)
    monkeypatch.setattr(
        vllm,
        "insert_benchmarks",
        lambda c, db, comp, rid, entries, **kw: (
            calls.append(kw.get("row_props") or {}),
            len(entries),
        )[1],
    )
    vllm.write_benchmarks(None, "db", _rows(tmp_path), "rid")
    assert len(calls) == 1
    assert vllm.REPEAT_PROP not in calls[0]
