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

"""Device-free tests for the Spyre provenance viewer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import torch_spyre.provenance as provenance
import torch_spyre.provenance_viewer as viewer
from torch_spyre.provenance import (
    load_provenance_document,
    ProvenanceReaderError,
)
from torch_spyre.provenance_viewer import (
    build_provenance_presentation,
    render_provenance_html,
    write_provenance_html,
)


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "provenance"
_SIDECAR = _FIXTURE_DIR / "valid_v1.json"
_DOM_CHECK = _FIXTURE_DIR / "viewer_dom_check.js"
_IDENTITY_KEY = "atqydvnuutl766na"
_EVENT_BASE = "spyre_kernel_v1_fused_linear_relu_atqydvnuutl766na"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_trace(path: Path) -> None:
    _write_json(
        path,
        {
            "traceEvents": [
                {
                    "name": _EVENT_BASE + "#2",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 10.5,
                    "pid": 3,
                    "tid": 4,
                    "args": {
                        "provenance_key": _IDENTITY_KEY,
                        "debug_handles": ["200"],
                        "correlation": 101,
                        "stream": 1,
                    },
                },
                {
                    "name": _EVENT_BASE + "#3",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 11.5,
                    "pid": 3,
                    "tid": 4,
                    "args": {"correlation": 102},
                },
                {"name": "unrelated_cpu_activity", "ph": "X"},
            ]
        },
    )


def _identity(presentation: dict) -> dict:
    return presentation["identities"][_IDENTITY_KEY]


def _panels(presentation: dict) -> dict[str, dict]:
    return {panel["id"]: panel for panel in _identity(presentation)["panels"]}


class TestValidatedSidecarLoader:
    def test_loads_and_validates_once(self, monkeypatch):
        calls = 0
        original = provenance._validate_document

        def counted(document):
            nonlocal calls
            calls += 1
            return original(document)

        monkeypatch.setattr(provenance, "_validate_document", counted)

        document = load_provenance_document(_SIDECAR)

        assert calls == 1
        assert document["schemaVersion"] == 1
        assert _IDENTITY_KEY in document["kernelIdentities"]

    @pytest.mark.parametrize(
        ("contents", "code"),
        [
            ("{", "schema-validation-failure"),
            ("[]", "schema-validation-failure"),
            ('{"schemaVersion":99}', "unsupported-schema-version"),
        ],
    )
    def test_rejects_invalid_documents(self, tmp_path, contents, code):
        path = tmp_path / "invalid.json"
        path.write_text(contents, encoding="utf-8")

        with pytest.raises(ProvenanceReaderError) as error:
            load_provenance_document(path)

        assert error.value.code == code


class TestPresentationModel:
    def test_deduplicates_source_and_aten_without_losing_multiplicity(self):
        source = {
            "file": "/workspace/model.py",
            "start_line": 17,
            "start_col": 0,
            "end_line": None,
            "end_col": None,
        }
        handles = {
            "1": {"source": source, "aten_op": "aten.add.Tensor"},
            "2": {"source": source, "aten_op": "aten.add.Tensor"},
        }

        source_rows = viewer._source_rows(handles)
        aten_rows = viewer._aten_rows(handles)

        assert len(source_rows) == len(aten_rows) == 1
        assert source_rows[0]["summary"] == "2 contributing handles"
        assert aten_rows[0]["summary"] == "2 contributing handles"
        assert source_rows[0]["refs"]["handleIds"] == ["1", "2"]
        assert aten_rows[0]["refs"]["handleIds"] == ["1", "2"]

    def test_builds_six_uniform_panel_collections(self):
        presentation = build_provenance_presentation(_SIDECAR)
        panels = _panels(presentation)

        assert presentation["presentationVersion"] == 1
        assert list(panels) == [
            "source",
            "aten",
            "pre-grad",
            "post-grad",
            "lower-ir",
            "opspec",
        ]
        assert [len(panel["rows"]) for panel in panels.values()] == [
            2,
            2,
            2,
            4,
            3,
            2,
        ]
        assert {row["label"] for row in panels["source"]["rows"]} == {
            "/workspace/model.py:17:0",
            "/workspace/model.py:18:0",
        }
        assert {row["label"] for row in panels["aten"]["rows"]} == {
            "aten.linear.default",
            "aten.relu.default",
        }
        assert {row["label"] for row in panels["pre-grad"]["rows"]} == {
            "linear",
            "relu",
        }
        assert {row["label"] for row in panels["post-grad"]["rows"]} == {
            "add",
            "mm",
            "permute",
            "relu",
        }

    def test_equal_fx_names_in_different_compiles_stay_scoped(self):
        occurrences = [
            {
                "compileId": compile_id,
                "occurrenceId": f"occurrence-{compile_id}",
                "registrations": [{"alias": "same_alias"}],
            }
            for compile_id in ("compile-a", "compile-b")
        ]
        projections = {
            compile_id: {
                "cppCodeToPost": {"same_alias": ["same_post"]},
                "postToPre": {"same_post": ["same_pre"]},
            }
            for compile_id in ("compile-a", "compile-b")
        }

        post_rows, pre_rows = viewer._fx_rows(
            occurrences,
            projections,
            {"1": {"ir_chain": []}},
            {"1": ["1"]},
            "ambiguous",
        )

        assert [row["label"] for row in post_rows] == ["same_post", "same_post"]
        assert [row["label"] for row in pre_rows] == ["same_pre", "same_pre"]
        assert len({row["id"] for row in post_rows}) == 2
        assert len({tuple(row["refs"]["postNodes"]) for row in post_rows}) == 2
        assert len({tuple(row["refs"]["preNodes"]) for row in pre_rows}) == 2

    def test_opspec_panel_contains_only_binding_granularity(self):
        panel = _panels(build_provenance_presentation(_SIDECAR))["opspec"]

        assert [row["label"] for row in panel["rows"]] == [
            "SpecPath [0]",
            "SpecPath [1, 0]",
        ]
        assert all(row["id"].startswith("opspec:") for row in panel["rows"])
        assert panel["scopeDetails"] == []
        serialized_rows = json.dumps(panel["rows"])
        assert "Finalized bundle" not in serialized_rows
        assert "Compile candidate" not in serialized_rows
        assert "sdsc_fused_linear_relu" not in serialized_rows
        assert "kernel" not in panel["description"].lower()
        assert "alias" not in panel["description"].lower()
        assert [row["summary"] for row in panel["rows"]] == [
            "Binding 0 | Handle 200",
            "Binding 1 | Handle 200",
        ]

    def test_lower_ir_keeps_complete_lineage_in_one_handle_row(self):
        panel = _panels(build_provenance_presentation(_SIDECAR))["lower-ir"]
        fused = next(row for row in panel["rows"] if row["label"] == "Handle 200")

        assert "IR: linear -> relu -> buf0" in fused["summary"]
        assert "Transforms: fusion by fuse_linear_relu" in fused["summary"]

    def test_evidence_strength_and_candidate_state_are_orthogonal(self):
        panels = _panels(build_provenance_presentation(_SIDECAR))

        assert {
            (row["evidenceStrength"], row["candidateState"])
            for row in panels["source"]["rows"]
        } == {("exact", "unique")}
        assert {row["candidateState"] for row in panels["post-grad"]["rows"]} == {
            "ambiguous"
        }
        assert any(
            row["evidenceStrength"] == "derived" for row in panels["post-grad"]["rows"]
        )
        assert all(
            row["evidenceStrength"] == "derived" for row in panels["pre-grad"]["rows"]
        )

    def test_rows_use_only_declared_typed_relationships(self):
        panels = _panels(build_provenance_presentation(_SIDECAR))
        expected = {"compileAliases", "handleIds", "postNodes", "preNodes"}

        for panel in panels.values():
            assert panel["description"]
            for row in panel["rows"]:
                assert "explanation" not in row
                assert set(row["refs"]) == expected
                assert all(
                    values == sorted(set(values)) for values in row["refs"].values()
                )
        first_binding = panels["opspec"]["rows"][0]
        assert first_binding["refs"]["handleIds"] == ["101", "102", "200"]

    def test_sidecar_only_mode_is_deterministic(self):
        first = build_provenance_presentation(_SIDECAR)
        second = build_provenance_presentation(_SIDECAR)

        assert first == second
        assert render_provenance_html(first) == render_provenance_html(second)
        assert first["inputs"]["kinetoTrace"]["status"] == "omitted"
        assert first["runSummary"] == {
            "resolvedObservations": 0,
            "displayedObservations": 0,
            "omittedResolvedObservations": 0,
            "unresolvedObservations": 0,
        }

    def test_sidecar_size_limit_fails_before_loading(self, monkeypatch):
        monkeypatch.setattr(viewer, "_MAX_SIDECAR_BYTES", _SIDECAR.stat().st_size - 1)

        with pytest.raises(ValueError) as error:
            build_provenance_presentation(_SIDECAR)

        assert error.value.code == "input-size-limit"

    def test_panel_rows_are_deterministically_capped(self, monkeypatch):
        monkeypatch.setattr(viewer, "_MAX_PANEL_ROWS", 1)

        presentation = build_provenance_presentation(_SIDECAR)
        panels = _panels(presentation)

        assert presentation["status"] == "partial"
        assert panels["source"]["totalRowCount"] == 2
        assert panels["source"]["displayedRowCount"] == 1
        assert panels["source"]["omittedRowCount"] == 1
        assert panels["source"]["rows"][0]["label"] == ("/workspace/model.py:17:0")
        assert "showing 1 of 2 rows" in panels["source"]["summary"]
        diagnostics = [
            item
            for item in presentation["diagnostics"]
            if item["code"] == "panel-rows-truncated"
        ]
        assert {item["details"]["panelId"] for item in diagnostics} == {
            "source",
            "aten",
            "pre-grad",
            "post-grad",
            "lower-ir",
            "opspec",
        }


class TestKinetoPairing:
    def test_retains_individual_runtime_observations(self, tmp_path):
        trace = tmp_path / "trace.json"
        _write_trace(trace)

        presentation = build_provenance_presentation(
            _SIDECAR,
            kineto_trace=trace,
        )

        event = next(
            item
            for item in presentation["events"]
            if item["identityKey"] == _IDENTITY_KEY
        )
        observations = event["observations"]
        assert [item["traceEventIndex"] for item in observations] == [0, 1]
        assert [item["commandStep"] for item in observations] == [2, 3]
        assert [item["durationUs"] for item in observations] == [10.5, 11.5]
        assert [item["keySource"] for item in observations] == [
            "both",
            "event-name",
        ]
        assert observations[0]["nativeDirectHandleIds"] == ["200"]
        assert presentation["inputs"]["kinetoTrace"]["pairing"] == {
            "candidateObservations": 2,
            "resolutionRate": 1.0,
            "resolvedObservations": 2,
            "status": "complete",
            "unresolvedObservations": 0,
        }

    def test_carrier_conflict_is_unresolved_and_loud(self, tmp_path):
        trace = tmp_path / "trace.json"
        _write_trace(trace)
        value = json.loads(trace.read_text(encoding="utf-8"))
        value["traceEvents"][0]["args"]["provenance_key"] = "aaaaaaaaaaaaaaaa"
        _write_json(trace, value)

        presentation = build_provenance_presentation(
            _SIDECAR,
            kineto_trace=trace,
        )

        assert presentation["status"] == "error"
        assert presentation["runSummary"]["unresolvedObservations"] == 1
        assert presentation["inputs"]["kinetoTrace"]["pairing"]["status"] == "error"
        assert "carrier-conflict" in {
            diagnostic["code"] for diagnostic in presentation["diagnostics"]
        }

    def test_malformed_optional_trace_fails_open(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text("{", encoding="utf-8")

        presentation = build_provenance_presentation(
            _SIDECAR,
            kineto_trace=trace,
        )

        assert presentation["status"] == "partial"
        assert presentation["inputs"]["kinetoTrace"]["status"] == "unavailable"
        assert presentation["events"]

    def test_native_handle_disagreement_does_not_replace_sidecar(self, tmp_path):
        trace = tmp_path / "trace.json"
        _write_trace(trace)
        value = json.loads(trace.read_text(encoding="utf-8"))
        value["traceEvents"][0]["args"]["debug_handles"] = ["101"]
        _write_json(trace, value)

        presentation = build_provenance_presentation(
            _SIDECAR,
            kineto_trace=trace,
        )

        assert presentation["runSummary"]["resolvedObservations"] == 2
        assert "native-handle-disagreement" in {
            diagnostic["code"] for diagnostic in presentation["diagnostics"]
        }
        assert _identity(presentation)["directHandleIds"] == ["200"]

    def test_runtime_occurrences_are_deterministically_capped(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(viewer, "_MAX_RUNTIME_OCCURRENCES", 1)
        trace = tmp_path / "trace.json"
        _write_trace(trace)

        presentation = build_provenance_presentation(
            _SIDECAR,
            kineto_trace=trace,
        )

        event = next(
            item
            for item in presentation["events"]
            if item["identityKey"] == _IDENTITY_KEY
        )
        assert [item["traceEventIndex"] for item in event["observations"]] == [0]
        assert event["observationCount"] == 2
        assert event["displayedObservationCount"] == 1
        assert event["omittedObservationCount"] == 1
        assert presentation["runSummary"] == {
            "resolvedObservations": 2,
            "displayedObservations": 1,
            "omittedResolvedObservations": 1,
            "unresolvedObservations": 0,
        }
        assert (
            presentation["inputs"]["kinetoTrace"]["pairing"]["resolvedObservations"]
            == 2
        )
        assert "runtime-occurrences-truncated" in {
            item["code"] for item in presentation["diagnostics"]
        }
        assert presentation["status"] == "partial"


class TestHtmlAndCli:
    def test_html_contains_only_viewer_interface(self):
        html = render_provenance_html(build_provenance_presentation(_SIDECAR))

        assert 'id="event-select"' in html
        assert 'id="observation-select"' in html
        assert "Python source locations" in html
        assert "Direct OpSpec bindings" in html
        assert "Advanced event lookup" not in html
        assert "Raw compiler text" not in html
        assert "Show complete attribution" not in html
        assert "compile-select" not in html
        assert "search-input" not in html
        assert "Selected runtime activity" not in html
        assert "--selection: #defbe6" in html
        assert "background: var(--selection)" in html
        assert "#0f62fe" not in html
        assert "#fff1b8" not in html
        assert "typed rewrite edges are unavailable" not in html.lower()

    def test_json_payload_is_script_safe(self):
        presentation = build_provenance_presentation(_SIDECAR)
        presentation["diagnostics"].append(
            {
                "code": "test",
                "severity": "warning",
                "message": "</script>&<!--\u2028",
            }
        )

        html = render_provenance_html(presentation)

        assert "</script>&<!--\u2028" not in html
        assert "\\u003c/script\\u003e\\u0026\\u003c!--\\u2028" in html
        assert "innerHTML" not in html
        assert "document.write" not in html
        assert "eval(" not in html
        assert "<script src=" not in html
        assert "<link " not in html
        assert "url(" not in html

    def test_atomic_writer_replaces_output(self, tmp_path):
        output = tmp_path / "viewer.html"
        output.write_text("old", encoding="utf-8")
        presentation = build_provenance_presentation(_SIDECAR)

        write_provenance_html(presentation, output)

        assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
        assert not list(tmp_path.glob(".viewer.html.*.tmp"))

    def test_documented_module_cli(self, tmp_path):
        output = tmp_path / "viewer.html"
        env = dict(os.environ)
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch_spyre.provenance_viewer",
                str(_SIDECAR),
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        assert completed.returncode == 0, completed.stderr
        assert output.is_file()
        help_result = subprocess.run(
            [sys.executable, "-m", "torch_spyre.provenance_viewer", "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert help_result.returncode == 0
        assert "--kineto-trace" in help_result.stdout
        assert "--compiler-artifacts" not in help_result.stdout

    def test_dom_interaction_gate(self, tmp_path):
        require_dom_gate = os.environ.get("SPYRE_REQUIRE_DOM_GATE") == "1"
        node = shutil.which("node")
        if node is None:
            if require_dom_gate:
                pytest.fail("SPYRE_REQUIRE_DOM_GATE=1 but Node.js is unavailable")
            pytest.skip("Node.js is unavailable")
        probe = subprocess.run(
            [
                node,
                "-e",
                (
                    'const version = require("jsdom/package.json").version;'
                    'process.exit(version.startsWith("30.") ? 0 : 2);'
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            message = (
                "jsdom 30 is unavailable; run "
                "'npm install --prefix /tmp/phase4a-jsdom --no-save "
                "jsdom@30.0.1' and set "
                "NODE_PATH=/tmp/phase4a-jsdom/node_modules"
            )
            if require_dom_gate:
                pytest.fail(f"SPYRE_REQUIRE_DOM_GATE=1 but {message}")
            pytest.skip(message)

        trace = tmp_path / "trace.json"
        output = tmp_path / "viewer.html"
        _write_trace(trace)
        write_provenance_html(
            build_provenance_presentation(_SIDECAR, kineto_trace=trace),
            output,
        )

        completed = subprocess.run(
            [node, str(_DOM_CHECK), str(output)],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert "Spyre provenance viewer DOM check passed" in completed.stdout
