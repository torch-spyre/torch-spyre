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

"""Writer classes for the bench tables of schema 81-83; ts/audit columns are left to their DEFAULTs
(result_invalidations takes an explicit ts: events of one id tie in argMax when they share it)."""

from .schema import Table


class Environments(Table):
    name = "environments"
    columns = ("name", "spec_digest", "labels", "devices", "host", "policy", "props")
    required = ("name", "spec_digest")


class NodeClasses(Table):
    name = "node_classes"
    columns = ("name", "spec_digest", "host_profile", "qualification", "props")
    required = ("name", "spec_digest")


class EnvironmentFingerprints(Table):
    name = "environment_fingerprints"
    columns = (
        "env_id",
        "fingerprint_version",
        "labels",
        "fingerprint",
        "members",
        "devices",
        "props",
    )
    required = ("env_id",)
    identity = "env_id"


class RunPlacements(Table):
    name = "run_placements"
    columns = (
        "run_id",
        "repeat",
        "started_at",
        "env_id",
        "placement_id",
        "nodes",
        "devices",
        "links",
        "clean",
        "props",
    )
    required = ("run_id", "env_id", "placement_id")


class EnvironmentQualifications(Table):
    name = "environment_qualifications"
    columns = (
        "node", "env_id", "probe", "outcome", "cv_pct", "limit_pct", "repeats", "checks",
        "evidence_run_id", "qualified_at", "expires_at", "props",
    )  # fmt: skip
    required = ("node", "env_id")


class ResultInvalidations(Table):
    name = "result_invalidations"
    columns = (
        "ts", "invalidation_id", "scope_kind", "run_id", "benchmark_id", "repeat", "node", "device_serial",
        "placement_id", "window_from", "window_to", "match", "match_expr", "matched_runs", "state",
        "reason_code", "reason", "actor", "evidence_url", "props",
    )  # fmt: skip
    required = ("invalidation_id", "scope_kind", "state", "reason_code", "actor")
    enums = (("state", frozenset({"suspect", "invalidated", "restored"})),)
