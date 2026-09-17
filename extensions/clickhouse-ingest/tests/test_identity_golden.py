"""Pins the derived-identity contract with GOLDEN values.

These uuids are written by four independent producers (this library, the Jenkins Groovy writer,
and two product-repo ingests). If any of them drifts, rows stop joining and the symptom is not an
error -- it is silently missing data that reads as "no tests ran". A golden test is the only thing
that catches a normalisation change before it ships.
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
    # Every producer derives from this. Changing it invalidates every id ever written.
    assert str(V2_NAMESPACE) == "cb0af9bf-2858-5eab-9211-f51190531bf3"


def test_run_id_golden():
    assert v2_run_id("gha", "12345", "amd64", "integration") == (
        "dab2a67f-14bf-53be-b6e4-fc9642086e47"
    )


def test_run_id_folds_arch_inside_the_hash():
    # amd64/x86/x86-64 are one arch. The fold happens INSIDE the hash, so a producer that
    # normalises afterwards would derive a different id for the same run.
    base = v2_run_id("gha", "12345", "x86_64", "integration")
    for alias in ("amd64", "x86", "x86-64", "X86_64"):
        assert v2_run_id("gha", "12345", alias, "integration") == base


def test_run_id_refuses_an_incomplete_key():
    # An empty field still hashes to a real, stable uuid that every other such case shares --
    # so the guard returns "" instead of minting a collision magnet.
    assert v2_run_id("", "12345", "amd64", "integration") == ""
    assert v2_run_id("gha", "", "amd64", "integration") == ""
    assert v2_run_id("gha", "12345", "", "integration") == ""


def test_test_case_id_is_component_scoped():
    # component is a hash input, so the same test under two components is two identities.
    # That is why a borrowed ingest MUST be told the real component (--component).
    a = v2_test_case_id("torch-spyre", "T", "test_x", [])
    b = v2_test_case_id("hf-adapters", "T", "test_x", [])
    assert a and b and a != b


def test_test_case_id_sorts_tags():
    # tags are a SET; source order is incidental. An order-sensitive hash would make two
    # writers disagree about the same test.
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
