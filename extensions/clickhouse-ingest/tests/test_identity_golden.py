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

from spyre_clickhouse_ingest import (
    V2_NAMESPACE,
    v2_canonical_arch,
    v2_component,
    v2_run_id,
    v2_test_case_id,
)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_namespace_is_the_agreed_constant():
    # Changing this invalidates every id already written.
    assert str(V2_NAMESPACE) == "cb0af9bf-2858-5eab-9211-f51190531bf3"


def test_run_id_golden():
    assert v2_run_id("gha", "12345", "amd64", "integration") == (
        "dab2a67f-14bf-53be-b6e4-fc9642086e47"
    )


def test_run_id_folds_arch_inside_the_hash():
    # The fold happens INSIDE the hash: normalising afterwards yields a different id.
    base = v2_run_id("gha", "12345", "x86_64", "integration")
    for alias in ("amd64", "x86", "x86-64", "X86_64"):
        assert v2_run_id("gha", "12345", alias, "integration") == base


def test_run_id_refuses_an_incomplete_key():
    # An empty field would hash to a real uuid shared by every other incomplete key.
    assert v2_run_id("", "12345", "amd64", "integration") == ""
    assert v2_run_id("gha", "", "amd64", "integration") == ""
    assert v2_run_id("gha", "12345", "", "integration") == ""


def test_test_case_id_is_component_scoped():
    # component is a hash input, so a borrowed ingest must be told the real component.
    a = v2_test_case_id("torch-spyre", "T", "test_x", [])
    b = v2_test_case_id("hf-adapters", "T", "test_x", [])
    assert a and b and a != b


def test_test_case_id_sorts_tags():
    # tags are a SET; an order-sensitive hash would make two writers disagree.
    assert v2_test_case_id("c", "T", "n", ["b", "a"]) == v2_test_case_id(
        "c", "T", "n", ["a", "b"]
    )


def test_canonical_arch_aliases():
    for alias in ("amd64", "x86", "x86-64", "x86_64"):
        assert v2_canonical_arch(alias) == "x86_64"


def test_component_prefers_the_explicit_override():
    assert v2_component(_Args(component="hf-adapters")) == "hf-adapters"


def test_component_falls_back_when_absent_or_blank():
    from spyre_clickhouse_ingest import V2_COMPONENT_DEFAULT

    assert v2_component(_Args(component="")) == V2_COMPONENT_DEFAULT
    assert v2_component(_Args(component="   ")) == V2_COMPONENT_DEFAULT
    assert v2_component(_Args()) == V2_COMPONENT_DEFAULT


def test_ids_are_uuid5_not_random():
    a = v2_run_id("gha", "1", "amd64", "unit")
    b = v2_run_id("gha", "1", "amd64", "unit")
    assert a == b
    assert uuid.UUID(a).version == 5


def test_component_default_is_caller_supplied():
    # Each repo has its own default. If the library baked one in, an importing repo would stamp
    # another product's name -- and component is a test_case_id hash input, so that mints a
    # different identity rather than merely mislabelling.
    assert v2_component(_Args(component=""), "hf-adapters") == "hf-adapters"
    assert v2_component(_Args(), "spyre-inference") == "spyre-inference"
    assert v2_component(_Args(component="torch-spyre"), "hf-adapters") == "torch-spyre"
