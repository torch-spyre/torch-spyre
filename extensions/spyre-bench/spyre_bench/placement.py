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

"""Observed CPU placement: compare the affinity of every thread in a process tree with the requested CPU set.

Linux only (/proc, /sys); anywhere else, or when any part of the tree cannot be read, the result is `unknown`, never
`true`. The tree is the root's descendants plus every process sharing its process group, session or cgroup, so a
re-parented worker is still judged. When the root sits in the root cgroup (a private cgroup namespace) a cgroup cannot
tie a worker to the root, so any process outside the tree that was re-parented to pid 1 makes the result `unknown`.
Verification covers affinity masks only: a cgroup cpuset narrower than the requested set is not distinguished. The
output keys are the run.exec.* run properties (strings, as stored).
"""

import sys
from pathlib import Path

import regex as re

POLICIES = ("none", "device_local", "per_device", "explicit")
MAX_CPU = 16383
LIST = re.compile(r"[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*")
ROOT_CGROUPS = ("", "0::/")


def parse_cpu_list(text):
    """'0-3,8' -> frozenset({0, 1, 2, 3, 8}). Raises ValueError on a malformed, empty, reversed or oversized list."""
    text = str(text)
    if not LIST.fullmatch(text):
        raise ValueError(f"bad CPU list {text!r}")
    out = set()
    for part in text.split(","):
        lo, dash, hi = part.partition("-")
        a, b = int(lo), int(hi) if dash else int(lo)
        if b < a or b > MAX_CPU:
            raise ValueError(f"bad CPU range {part!r} (ids run 0-{MAX_CPU}, low to high)")
        out.update(range(a, b + 1))
    return frozenset(out)


def format_cpu_list(cpus):
    """Canonical range string: sorted, merged, no spaces."""
    runs, last = [], None
    for c in sorted(cpus):
        if last is not None and c == last + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
        last = c
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _linux(root):
    return str(root) != "/proc" or sys.platform.startswith("linux")


def _stat(proc, pid):
    """(ppid, pgrp, session, starttime) from /proc/<pid>/stat; comm may contain spaces and parens."""
    stat = (proc / str(pid) / "stat").read_text()
    f = stat[stat.rindex(")") + 1 :].split()
    if len(f) < 20:
        raise ValueError(f"short stat for pid {pid}")
    return int(f[1]), int(f[2]), int(f[3]), int(f[19])  # ppid, pgrp, session, starttime (field 22)


def _tree(proc, root_pid, snap, cgroup):
    root = snap[root_pid]
    kids = {}
    for pid, (ppid, _, _, _) in snap.items():
        kids.setdefault(ppid, []).append(pid)
    members, todo = set(), [root_pid]
    while todo:
        pid = todo.pop()
        if pid not in members:
            members.add(pid)
            todo += kids.get(pid, [])
    members |= {pid for pid, s in snap.items() if s[1] == root[1] or s[2] == root[2]}
    if cgroup in ROOT_CGROUPS:
        if any(s[0] == 1 and pid not in members and pid != 1 for pid, s in snap.items()):
            return None
        return members
    for pid in snap.keys() - members:
        try:
            same = (proc / str(pid) / "cgroup").read_text().strip() == cgroup
        except FileNotFoundError:
            continue
        if same:
            members.add(pid)
    return members


def read_affinities(root_pid, proc="/proc", expected_starttime=None):
    """[(pid, tid, frozenset(cpus))] for the process tree of root_pid, or None unless every member was read.

    A process that exits mid-scan is skipped; any other unreadable or malformed entry, an empty mask, a root whose
    start time changed during the scan or differs from expected_starttime (the value recorded at spawn: pid reuse),
    or, in the root cgroup, a re-parented process outside the tree, makes the whole observation None.
    """
    proc = Path(proc)
    root_pid = int(root_pid)
    if not _linux(proc):
        return None
    try:
        started = _stat(proc, root_pid)[3]
        if expected_starttime is not None and int(expected_starttime) != started:
            return None
        cgroup = ""
        try:
            cgroup = (proc / str(root_pid) / "cgroup").read_text().strip()
        except FileNotFoundError:
            pass
        snap = {}
        for d in proc.iterdir():
            if d.name.isdigit():
                try:
                    snap[int(d.name)] = _stat(proc, d.name)
                except FileNotFoundError:
                    continue
        if root_pid not in snap:
            return None
        out = []
        members = _tree(proc, root_pid, snap, cgroup)
        if members is None:
            return None
        for pid in sorted(members):
            try:
                tasks = sorted((proc / str(pid) / "task").iterdir())
            except FileNotFoundError:
                continue
            for t in tasks:
                try:
                    lines = (t / "status").read_text().splitlines()
                except FileNotFoundError:
                    continue
                masks = [x.split(":", 1)[1] for x in lines if x.startswith("Cpus_allowed_list:")]
                if len(masks) != 1:
                    return None
                out.append((pid, int(t.name), parse_cpu_list(masks[0].strip())))
        return out if out and _stat(proc, root_pid)[3] == started else None
    except (OSError, ValueError):
        return None


def numa_nodes(value):
    """NUMA node ids from a device's `numa_node`: an int, a list, or the fingerprint string (comma-separated integers,
    one per device in device order). None when any entry is malformed."""
    try:
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        if isinstance(value, str):
            return [int(v.strip()) for v in value.split(",")]
        return [int(value)]
    except (TypeError, ValueError):
        return None


def node_cpus(numa_node, sys_root="/sys"):
    """CPUs of one NUMA node from sysfs, or None."""
    try:
        n = int(numa_node)
        if n < 0:
            return None
        return parse_cpu_list((Path(sys_root) / f"devices/system/node/node{n}/cpulist").read_text().strip())
    except (OSError, ValueError, TypeError):
        return None


def resolve_requested(request, device_numa=(), sys_root="/sys"):
    """(allowed_sets, source): the CPU sets a thread may be inside (any one of them).

    (None, 'unresolved') when any requested device's local CPUs are unknown: a partial set would flag valid placements.
    """
    policy = (request or {}).get("policy", "none")
    if policy == "none":
        return None, "none"
    if policy == "explicit":
        try:
            return [parse_cpu_list(request["cpus"])], "explicit"
        except (KeyError, ValueError):
            return None, "unresolved"
    nodes = numa_nodes(device_numa) if device_numa not in ((), [], "", None) else None
    sets = [node_cpus(n, sys_root) for n in nodes or ()]
    if not sets or any(s is None for s in sets):
        return None, "unresolved"
    return ([frozenset().union(*sets)] if policy == "device_local" else sets), "device_numa"


def verify_pinning(request, pid, device_numa=(), proc="/proc", sys_root="/sys", expected_starttime=None):
    """run.exec.* properties for the process tree at pid under the requested cpu_pinning block.

    expected_starttime: the root's /proc starttime recorded at spawn; a different value (pid reuse) gives unknown.
    """
    policy = (request or {}).get("policy", "none")
    allowed, source = resolve_requested(request, device_numa, sys_root)
    seen = read_affinities(pid, proc, expected_starttime)
    props = {"run.exec.cpu_pinning": policy, "run.exec.cpu_set_source": source}
    if seen:
        props["run.exec.cpu_set"] = format_cpu_list(frozenset().union(*(c for _, _, c in seen)))
    if allowed is None or not seen:
        return {**props, "run.exec.pinning_verified": "unknown"}
    bad = sum(1 for _, _, c in seen if not any(c <= a for a in allowed))
    return {
        **props,
        "run.exec.pinning_verified": "false" if bad else "true",
        "run.exec.pinning_violations": str(bad),
    }


def validity_facts(request, props):
    """validity_of() facts from the request and the run.exec.* props of one run or of every repeat.

    The worst repeat wins: false, then unknown (a repeat without the property counts as unknown), then true.
    """
    reps = list(props) if isinstance(props, (list, tuple)) else [props]
    seen = [(p or {}).get("run.exec.pinning_verified", "unknown") for p in reps] or ["unknown"]
    worst = "false" if "false" in seen else "true" if all(v == "true" for v in seen) else "unknown"
    return {"pinning_required": bool((request or {}).get("require_verified")), "pinning_verified": worst}
