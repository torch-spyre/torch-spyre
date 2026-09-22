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

"""Pins the derived-identity contract with golden values.

Four independent producers must agree on these uuids -- this library, the Jenkins Groovy writer,
and the product-repo ingests. A drift does not raise; it silently stops rows joining.
"""

import uuid

import pytest
from spyre_clickhouse_ingest import (
    ID_NAMESPACE,
    artifact_id_for,
    benchmark_id_for,
    canonical_arch,
    component_of,
    base_artifact_id,
    gha_artifact_id,
    run_id_of,
    case_id_for,
)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_namespace_is_the_agreed_constant():
    # Changing this invalidates every id already written.
    assert str(ID_NAMESPACE) == "cb0af9bf-2858-5eab-9211-f51190531bf3"


def test_run_id_golden():
    assert run_id_of("gha", "12345", "amd64", "integration") == (
        "dab2a67f-14bf-53be-b6e4-fc9642086e47"
    )


def test_run_id_folds_arch_inside_the_hash():
    # The fold happens INSIDE the hash: normalising afterwards yields a different id.
    base = run_id_of("gha", "12345", "x86_64", "integration")
    for alias in ("amd64", "x86", "x86-64", "X86_64"):
        assert run_id_of("gha", "12345", alias, "integration") == base


def test_run_id_refuses_an_incomplete_key():
    # An empty field would hash to a real uuid shared by every other incomplete key.
    assert run_id_of("", "12345", "amd64", "integration") == ""
    assert run_id_of("gha", "", "amd64", "integration") == ""
    assert run_id_of("gha", "12345", "", "integration") == ""


def test_test_case_id_is_component_scoped():
    # component is a hash input, so a borrowed ingest must be told the real component.
    a = case_id_for("torch-spyre", "T", "test_x", [])
    b = case_id_for("hf-adapters", "T", "test_x", [])
    assert a and b and a != b


def test_test_case_id_sorts_tags():
    # tags are a SET; an order-sensitive hash would make two writers disagree.
    assert case_id_for("c", "T", "n", ["b", "a"]) == case_id_for(
        "c", "T", "n", ["a", "b"]
    )


def test_canonical_arch_aliases():
    for alias in ("amd64", "x86", "x86-64", "x86_64"):
        assert canonical_arch(alias) == "x86_64"


def test_component_prefers_the_explicit_override():
    assert component_of(_Args(component="hf-adapters")) == "hf-adapters"


def test_component_falls_back_when_absent_or_blank():
    from spyre_clickhouse_ingest import COMPONENT_DEFAULT

    assert component_of(_Args(component="")) == COMPONENT_DEFAULT
    assert component_of(_Args(component="   ")) == COMPONENT_DEFAULT
    assert component_of(_Args()) == COMPONENT_DEFAULT


def test_ids_are_uuid5_not_random():
    a = run_id_of("gha", "1", "amd64", "unit")
    b = run_id_of("gha", "1", "amd64", "unit")
    assert a == b
    assert uuid.UUID(a).version == 5


def test_component_default_is_caller_supplied():
    # Each repo has its own default. If the library baked one in, an importing repo would stamp
    # another product's name -- and component is a test_case_id hash input, so that mints a
    # different identity rather than merely mislabelling.
    assert component_of(_Args(component=""), "hf-adapters") == "hf-adapters"
    assert component_of(_Args(), "spyre-inference") == "spyre-inference"
    assert component_of(_Args(component="torch-spyre"), "hf-adapters") == "torch-spyre"


# ── artifact_id ─────────────────────────────────────────────────────────────────────────
# These goldens are REAL artifact_ids read from the live warehouse, not values this
# implementation produced. That is the point: three writers computed them (this library, an
# inlined python3 script in the Jenkins-side Groovy writer, and a product ingest's own copy),
# so pinning what is already STORED is what proves centralising the function re-keys nothing.

ARTIFACT_ID_GOLDENS = [
    # (component, artifact_name, id12, arch, expected)
    (
        "spyre-inference",
        "spyre-inference-minimal",
        "a6e224825fcf",
        "amd64",
        "1eef58cc-b88b-5597-97c6-6b27b8b4796f",
    ),
    (
        "spyre-inference",
        "spyre-inference-dev",
        "492504101aed",
        "amd64",
        "91b58b7a-0ead-5700-bf72-2e7257322ba6",
    ),
    (
        "hf-adapters",
        "hf-adapters-minimal",
        "be1d78c2cdfd",
        "amd64",
        "23aa0294-e2ff-5f48-8aca-ecd318b23a0e",
    ),
    # ppc64le is NOT folded, unlike amd64 -- covers both sides of canonical_arch.
    (
        "spyre-backend",
        "spyre-backend-minimal",
        "cb0f2bca6527",
        "ppc64le",
        "d9c4fe94-894f-50f3-85f8-e9272acfedb1",
    ),
]


@pytest.mark.parametrize("component,name,id12,arch,expected", ARTIFACT_ID_GOLDENS)
def test_artifact_id_golden(component, name, id12, arch, expected):
    assert artifact_id_for(component, name, id12, arch) == expected


def test_artifact_id_folds_arch_inside_the_hash():
    # An artifact labelled amd64 by Jenkins and x86_64 by GHA is ONE artifact.
    a = artifact_id_for(
        "spyre-inference", "spyre-inference-dev", "492504101aed", "amd64"
    )
    b = artifact_id_for(
        "spyre-inference", "spyre-inference-dev", "492504101aed", "x86_64"
    )
    assert a == b == "91b58b7a-0ead-5700-bf72-2e7257322ba6"


def test_artifact_id_refuses_an_incomplete_key():
    # A blank component or arch would hash to a real uuid that every incomplete artifact
    # shares -- worse than a blank, which a writer can detect and skip.
    assert artifact_id_for("", "n", "i", "amd64") == ""
    assert artifact_id_for("c", "n", "i", "") == ""
    # id12 is legitimately blank for a GHA-derived identity, so it is NOT required.
    assert artifact_id_for("c", "n", "", "amd64") != ""


def test_gha_artifact_id_is_install_order_independent():
    # Install order is incidental; an order-sensitive hash would mint a fresh identity for a
    # re-run of the same environment.
    a = gha_artifact_id("torch-spyre", "base:1", "pkg-b pkg-a", "amd64")
    b = gha_artifact_id("torch-spyre", "base:1", "pkg-a,pkg-b", "amd64")
    assert a == b != ""


def test_gha_artifact_id_lands_in_the_same_column_as_a_built_one():
    # It is an artifact_id, so it must be a uuid on the same derivation -- not a second,
    # GHA-shaped identity that artifact_results would need another column for.
    got = gha_artifact_id("torch-spyre", "base:1", "pkg-a", "amd64")
    assert uuid.UUID(got).version == 5


def test_base_artifact_id_reads_the_in_image_file(tmp_path):
    # The builder writes the id beside installed_rpms.txt; a test leg runs INSIDE the image
    # and reads it from there rather than inspecting its own OCI label over the network.
    f = tmp_path / "spyre_artifact_id.txt"
    f.write_text("  2B397099-6200-52FB-98C4-B603961A0582 \n")
    # Normalised, so a builder writing upper-case or a trailing newline still joins.
    assert base_artifact_id(str(f)) == "2b397099-6200-52fb-98c4-b603961a0582"


def test_base_artifact_id_is_blank_when_absent(tmp_path):
    # An image built before the label existed, or a standalone build with no orchestrator
    # node. Blank means "no base identity" -- callers must fall back to their own coordinate,
    # never to a defaulted hash, which every such leg would share.
    assert base_artifact_id(str(tmp_path / "nope.txt")) == ""
    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    assert base_artifact_id(str(empty)) == ""


def test_gha_artifact_id_chains_onto_a_real_embedded_id():
    # End-to-end shape: the id spyre-frameworks stamps into the image (verified against prod
    # for spyre-backend-dev/amd64) is what a GHA leg hashes its delta onto.
    embedded = "2b397099-6200-52fb-98c4-b603961a0582"
    got = gha_artifact_id("torch-spyre", embedded, "lxml clickhouse-connect", "amd64")
    assert uuid.UUID(got).version == 5
    # Distinct from the base: the leg IS a different artifact from the image it started on.
    assert got != embedded


# The vLLM perf writer's discriminator set. Positional: reordering re-keys every benchmark.
_VLLM_KEYS = ("record_type", "run_mode", "tensor_parallel", "input_len", "output_len")


def test_benchmark_id_golden():
    assert (
        benchmark_id_for(
            "spyre-inference",
            "serve_granite33-8b_tp1_in64_out64",
            [],
            {"run_mode": "serve", "tensor_parallel": "1"},
            _VLLM_KEYS,
        )
        == "3a681ed6-dc5a-517a-a1c6-bc3367ce815c"
    )
    # The hashed string this pins, spelled out so a drift is diagnosable without a debugger:
    # 'spyre-inference|serve_granite33-8b_tp1_in64_out64||record_type=,run_mode=serve,
    #  tensor_parallel=1,input_len=,output_len='
    assert uuid.uuid5(
        ID_NAMESPACE,
        "spyre-inference|serve_granite33-8b_tp1_in64_out64||"
        "record_type=,run_mode=serve,tensor_parallel=1,input_len=,output_len=",
    ) == uuid.UUID("3a681ed6-dc5a-517a-a1c6-bc3367ce815c")


def test_benchmark_id_normalises_component():
    # component is normalised too, else 'Torch-Spyre' is a different benchmark.
    assert benchmark_id_for("Torch-Spyre", "matmul", [], {}, ()) == benchmark_id_for(
        "torch-spyre", "matmul", [], {}, ()
    )


def test_benchmark_id_refuses_an_incomplete_key():
    # An empty field hashes to a real uuid, so all such benchmarks would share one id.
    assert benchmark_id_for("", "n", [], {}, ()) == ""
    assert benchmark_id_for("c", "", [], {}, ()) == ""


def test_benchmark_id_is_tag_order_independent():
    # tags are a set; an unsorted join makes two writers disagree about one benchmark.
    a = benchmark_id_for("c", "n", ["b", "a"], {}, ())
    b = benchmark_id_for("c", "n", ["a", "b", "a"], {}, ())
    assert a == b != ""


def test_backend_is_not_part_of_the_identity():
    # backend is the axis comparison pivots on; hashing it splits one comparison in two.
    cpu = benchmark_id_for("c", "n", [], {"backend": "cpu"}, ())
    spyre = benchmark_id_for("c", "n", [], {"backend": "spyre"}, ())
    assert cpu == spyre


def test_producers_with_different_disc_keys_cannot_collide():
    # component leads the hash, which is what lets each producer keep its own key set.
    kernel = benchmark_id_for(
        "torch-spyre", "x", [], {"record_type": "op"}, ("record_type", "config_name")
    )
    vllm = benchmark_id_for(
        "spyre-inference", "x", [], {"record_type": "op"}, _VLLM_KEYS
    )
    assert kernel != vllm


def test_disc_key_order_is_positional():
    # disc_keys is the key order in the hash, so alphabetising a producer's tuple re-keys
    # every benchmark -- pinned so that cannot pass silently.
    a = benchmark_id_for("c", "n", [], {"a": "1", "b": "2"}, ("a", "b"))
    b = benchmark_id_for("c", "n", [], {"a": "1", "b": "2"}, ("b", "a"))
    assert a != b
