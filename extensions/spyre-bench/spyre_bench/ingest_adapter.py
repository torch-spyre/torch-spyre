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

"""The only module that imports the ingest: the stored benchmark_id comes from its derivation (contract in docs/README.md)."""

import importlib
import importlib.metadata
import importlib.util

CONTRACT_VERSION = 1
INGEST_MODULE = "spyre_clickhouse_ingest.identity"
INGEST_DIST = "spyre-clickhouse-ingest"
INVALIDATION_MODULE = "spyre_clickhouse_ingest.bench_invalidation"
# Known answers, each element pinning one property of the ingest rule: tags are stripped, lower-cased, deduplicated,
# sorted, raw (no alias or namespace handling, empty dropped, non-str stringified); values only strip and lower-case
# (no NFC/NFKC/casefold, inner whitespace kept, bool/float/None via str, no truncation); keys keep case and padding, an
# absent key hashes empty, duplicates and order are kept; component and name are stripped and lower-cased untruncated;
# an all-empty disc part still hashes. It cannot prove every property (docs/README.md says which it does not cover).
_LONG = "Ab" * 125
PROBES = (
    (
        (
            " Probe-Comp ",
            " " + _LONG + " ",
            ["", " Zed ", "alpha", "ZED", "Cadence__Nightly", "nightly", "Refcoverage__X", "trail_", 5],
            {
                "Zeta": "ß  Ａb ",
                "a": " \t Q\n",
                " pad ": "e\u0301 x\ny",
                "flag": True,
                "ratio": 2.0,
                "none": None,
                "long": "L" * 300,
            },
            ["Zeta", "m", "a", " pad ", "flag", "ratio", "none", "long", "Zeta"],
        ),
        "021b6409-b7c4-5a62-b5a9-4d8ce576f359",
    ),
    (("c", "n", [], {}, ["x"]), "e813e541-091c-5a3d-ad65-906e381a7ab8"),
)
HINT = (
    "install the ingest from a checkout: uv pip install 'extensions/spyre-bench[ingest]' or uv run --extra ingest "
    "(the extra is not installable from a package index); or vendor INGEST_MODULE (docs/README.md)"
)
PROFILE_SOURCES = {
    "perf-kernel": ("spyre_clickhouse_ingest.results", "_V2_BENCH_ID_KEYS"),
    "vllm": ("spyre_clickhouse_ingest.vllm", "BENCH_ID_KEYS"),
    "vllm-sweep": ("spyre_clickhouse_ingest.vllm", "SWEEP_ID_KEYS"),
}

OFFLINE_MODULE = "spyre_clickhouse_ingest.offline"
# The offline module's bench support: absent in an ingest that predates bench bundles.
OFFLINE_NEEDS = ("bench_inputs", "content_digest", "receipt_id")
UPGRADE_HINT = "upgrade the ingest first (a kit sealed by a newer ingest than the receiving one is refused)"

_state = {}


class IngestUnavailable(RuntimeError):
    """The ingest derivation is missing or no longer matches the probe; the message says which."""


def _fail(message):
    _state["error"] = message
    raise IngestUnavailable(message)


def _load():
    if "derive" in _state:
        return _state["derive"]
    if "error" in _state:
        raise IngestUnavailable(_state["error"])
    try:
        mod = importlib.import_module(INGEST_MODULE)
        derive, namespace = mod.BenchmarkId.derive, mod.ID_NAMESPACE
    except Exception as e:  # noqa: BLE001 - any import-time failure means the derivation is unusable
        _fail(f"{INGEST_MODULE} is not importable ({type(e).__name__}: {e}); {HINT}")
    for args, expected in PROBES:
        try:
            got = str(derive(*args))
        except Exception as e:  # noqa: BLE001 - a derivation that raises on the probe has changed
            got = f"a {type(e).__name__} ({e})"
        if got != expected:
            _fail(
                f"{INGEST_MODULE} derives {got} for a known-answer probe, expected {expected}: the ingest rule "
                f"changed; update PROBES and the conformance table deliberately"
            )
    _state.update(derive=derive, namespace=str(namespace))
    return derive


def offline():
    """The ingest's stdlib-only sealing and upload module (it reads bundle.schema.json beside itself, so the whole
    package is needed; a vendored identity module is not enough)."""
    try:
        mod = importlib.import_module(OFFLINE_MODULE)
    except Exception as e:  # noqa: BLE001 - any import-time failure means the module is unusable
        raise IngestUnavailable(f"{OFFLINE_MODULE} is not importable ({type(e).__name__}: {e}); {HINT}") from e
    missing = [n for n in OFFLINE_NEEDS if not hasattr(mod, n)]
    if missing:
        raise IngestUnavailable(
            f"{OFFLINE_MODULE} ({getattr(mod, '__file__', 'unknown path')}) lacks {missing}: it predates bench bundles; {UPGRADE_HINT}"
        )
    return mod


def reset():
    """Forget the cached import (tests)."""
    _state.clear()


def available():
    try:
        _load()
    except IngestUnavailable:
        return False
    return True


def benchmark_id(component, name, tags, disc, disc_keys):
    return str(_load()(component, name, list(tags or []), dict(disc or {}), list(disc_keys or [])))


def profile_status(profile):
    """('keys', tuple) from the ingest's own constant; ('absent', reason) when its producer module does not exist (a
    vendored identity module); ('unverified', reason) when it exists but cannot be imported or read."""
    module, name = PROFILE_SOURCES[profile]
    try:
        if importlib.util.find_spec(module) is None:
            return "absent", f"{module} does not exist"
    except ImportError as e:  # the parent package itself is not there
        return "absent", str(e)
    except ValueError:  # a module object without a spec is present: import it as is
        pass
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        return "unverified", f"producer modules not importable: {type(e).__name__}: {e}"
    try:
        return "keys", tuple(getattr(mod, name))
    except AttributeError as e:
        raise IngestUnavailable(f"{module} no longer has {name}: the ingest rule changed ({e})") from e


def profile_keys(profile):
    """The ingest's key list for a profile, or None when it cannot be read (see profile_status)."""
    kind, value = profile_status(profile)
    return value if kind == "keys" else None


def invalidation():
    """The ingest's invalidation module (preview, write rules, bundle kind), or IngestUnavailable with how to install it."""
    try:
        return importlib.import_module(INVALIDATION_MODULE)
    except Exception as e:  # noqa: BLE001 - any import-time failure means the module is unusable
        raise IngestUnavailable(f"{INVALIDATION_MODULE} is not importable ({type(e).__name__}: {e}); {HINT}") from e


def info():
    """Status for `spyre-bench ids --adapter`: availability, module, distribution version, error."""
    ok = available()
    try:
        version = importlib.metadata.version(INGEST_DIST)
    except importlib.metadata.PackageNotFoundError:
        version = "not installed as a distribution"
    return {
        "contract": CONTRACT_VERSION,
        "module": INGEST_MODULE,
        "available": ok,
        "distribution": INGEST_DIST,
        "version": version if ok else None,
        "namespace": _state.get("namespace"),
        "error": None if ok else _state["error"],
    }
