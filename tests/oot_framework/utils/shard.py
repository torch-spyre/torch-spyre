"""Split one suite's collected tests across parallel CI jobs.

A suite config can set `test_suite_config.shards: N`; filter_configs.py then emits N matrix
entries for it, and each job runs with SPYRE_TEST_SHARD_ID=k (0-based) and
SPYRE_TEST_NUM_SHARDS=N. conftest.py calls `select_shard` after collection, so every job keeps a
disjoint slice and together they run exactly the tests the unsharded suite would.

Assignment is deterministic, so every job computes the same partition without coordinating:
  - With a durations file (SPYRE_TEST_DURATIONS, JSON {nodeid: seconds}), tests are placed
    longest-first onto the least-loaded shard. Unknown tests get the median known duration.
  - Without one, each test goes to sha1(nodeid) % N.
Ties in the greedy pass break on nodeid, so the result never depends on collection order.
"""
import gzip
import hashlib
import json
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

ENV_SHARD_ID = "SPYRE_TEST_SHARD_ID"
ENV_NUM_SHARDS = "SPYRE_TEST_NUM_SHARDS"
ENV_DURATIONS = "SPYRE_TEST_DURATIONS"


def shard_from_env(environ=os.environ) -> Optional[Tuple[int, int]]:
    """(shard_id, num_shards) from the environment, or None when this job is not sharded.

    Raises ValueError on a malformed or out-of-range pair: a job that silently ran every test
    (or none) would hide the misconfiguration behind a green check.
    """
    raw_n = (environ.get(ENV_NUM_SHARDS) or "").strip()
    raw_k = (environ.get(ENV_SHARD_ID) or "").strip()
    if not raw_n and not raw_k:
        return None
    if not raw_n or not raw_k:
        raise ValueError(f"{ENV_SHARD_ID} and {ENV_NUM_SHARDS} must be set together "
                         f"(got {raw_k!r}, {raw_n!r})")
    n, k = int(raw_n), int(raw_k)
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"shard {k} out of range for {n} shard(s)")
    return (k, n) if n > 1 else None


def load_durations(path: Optional[str]) -> Dict[str, float]:
    """{key: seconds} from a JSON file (gzipped if it ends in .gz); {} when unset or unreadable.

    Keys are `duration_key` strings or nodeids.
    """
    if not path:
        return {}
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float)) and v >= 0}


def assign(nodeids: Sequence[str], num_shards: int,
           durations: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """nodeid -> shard index, the same on every job for the same inputs."""
    if not durations:
        return {n: int(hashlib.sha1(n.encode("utf-8")).hexdigest(), 16) % num_shards
                for n in nodeids}
    known = [durations[n] for n in nodeids if n in durations]
    default = statistics.median(known) if known else 1.0
    weight = {n: durations.get(n, default) for n in nodeids}
    loads = [0.0] * num_shards
    out: Dict[str, int] = {}
    for n in sorted(nodeids, key=lambda x: (-weight[x], x)):
        k = min(range(num_shards), key=lambda i: (loads[i], i))
        out[n] = k
        loads[k] += weight[n]
    return out


_WRAPPER_SUFFIX = "__oot_wrapper"


def duration_key(item) -> str:
    """`<file stem>::[<Class>::]<test name>`, the same for a collected item and its JUnit case.

    A nodeid depends on how run_test.sh invoked pytest (its cwd, xdist, torchrun, and the generated
    `<stem>__oot_wrapper.py` module), so durations recorded from one run's JUnit XML would not match
    another run's nodeids. The stem (wrapper suffix dropped), class and name are the same everywhere.
    """
    path = getattr(item, "path", None) or getattr(item, "fspath", None)
    stem = os.path.splitext(os.path.basename(str(path)))[0] if path else ""
    if stem.endswith(_WRAPPER_SUFFIX):
        stem = stem[: -len(_WRAPPER_SUFFIX)]
    cls = getattr(item, "cls", None)
    parts = [stem] + ([cls.__name__] if cls is not None else []) + [getattr(item, "name", "")]
    return "::".join(parts)


def select_shard(items: List, shard: Tuple[int, int], durations: Dict[str, float]):
    """(kept, deselected) items for shard `shard` = (k, N).

    A test's duration is looked up by `duration_key`, then by its nodeid. The partition itself is
    keyed by nodeid, so every test still lands in exactly one shard whatever the durations say.
    """
    k, n = shard
    by_nodeid = {}
    for i in items:
        d = durations.get(duration_key(i), durations.get(i.nodeid))
        if d is not None:
            by_nodeid[i.nodeid] = d
    where = assign([i.nodeid for i in items], n, by_nodeid)
    kept = [i for i in items if where[i.nodeid] == k]
    dropped = [i for i in items if where[i.nodeid] != k]
    return kept, dropped
