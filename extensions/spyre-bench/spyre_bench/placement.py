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

A request with per-device lists (`cpus_by_device`) is judged per device: every thread must be inside one list, or, with
`layout: ranks`, inside the list of its own rank (the LOCAL_RANK of its process environment) and a process without a
rank (the launcher agent) inside their union. Processes still in the launcher stage (the pinning tool or the rank
wrapper before it execs the command) are not judged; a sample that sees only those is skipped by the caller.
"""

import os
import sys
from pathlib import Path

import regex as re

POLICIES = ("none", "device_local", "per_device", "explicit", "platform")
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
        # The observer and its ancestors are no orphans, though in a container they are direct children of pid 1.
        own, pid = set(), os.getpid()
        while pid in snap and pid not in own:
            own.add(pid)
            pid = snap[pid][0]
        if any(s[0] == 1 and p not in members and p != 1 and p not in own for p, s in snap.items()):
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


def starttime(pid, proc="/proc"):
    """The /proc start time of a process (to detect pid reuse), or None when unreadable."""
    try:
        return _stat(Path(proc), pid)[3]
    except (OSError, ValueError):
        return None


def launcher_stage(proc, pid, markers=()):
    """True while a process is still the engine's own pinning tool or rank wrapper.

    markers: wrapper paths, and launcher prefixes joined with NUL; a benchmark that runs taskset/numactl itself is not
    the launcher stage and is judged by its masks like any other process.
    """
    try:
        argv = [a for a in (proc / str(pid) / "cmdline").read_bytes().decode(errors="replace").split("\0") if a]
    except OSError:
        return False
    joined = "\0".join(argv) + "\0"
    return bool(argv) and any(joined.startswith(m) if "\0" in m else m in argv[:3] for m in markers)


def read_affinities(root_pid, proc="/proc", expected_starttime=None, markers=None):
    """[(pid, tid, frozenset(cpus))] for the process tree of root_pid, or None unless every member was read.

    A process that exits mid-scan is skipped; any other unreadable or malformed entry, an empty mask, a root whose
    start time changed during the scan or differs from expected_starttime (the value recorded at spawn: pid reuse),
    or, in the root cgroup, a re-parented process outside the tree, makes the whole observation None. With `markers`
    (a collection of wrapper paths; may be empty) processes in the launcher stage are left out, and a tree made only of
    them gives [] (nothing to judge yet) rather than None.
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
        out, staged = [], 0
        members = _tree(proc, root_pid, snap, cgroup)
        if members is None:
            return None
        for pid in sorted(members):
            if markers is not None and launcher_stage(proc, pid, markers):
                staged += 1
                continue
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
        if not out and staged:
            return []
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
    """CPUs of one NUMA node from sysfs, or None. A kernel without NUMA (no node directory) has node 0 = all online."""
    try:
        n = int(numa_node)
        if n < 0:
            return None
        root = Path(sys_root) / "devices/system"
        if n == 0 and not (root / "node").is_dir():
            return parse_cpu_list((root / "cpu/online").read_text().strip())
        return parse_cpu_list((root / f"node/node{n}/cpulist").read_text().strip())
    except (OSError, ValueError, TypeError):
        return None


def resolve_requested(request, device_numa=(), sys_root="/sys"):
    """(allowed_sets, source): the CPU sets a thread may be inside (any one of them).

    (None, 'unresolved') when any requested device's local CPUs are unknown: a partial set would flag valid placements.
    """
    policy = (request or {}).get("policy", "none")
    if policy == "none":
        return None, "none"
    if policy == "platform":
        return None, "platform"  # pinned by the platform in-process: recorded, never judged
    if (request or {}).get("unresolved"):
        return None, "unresolved"
    if policy == "explicit":
        try:
            return [parse_cpu_list(request["cpus"])], "explicit"
        except (KeyError, ValueError):
            return None, "unresolved"
    if policy == "per_device" and request.get("cpus_by_device"):
        try:
            return [parse_cpu_list(x) for x in request["cpus_by_device"]], "explicit"
        except ValueError:
            return None, "unresolved"
    nodes = numa_nodes(device_numa) if device_numa not in ((), [], "", None) else None
    sets = [node_cpus(n, sys_root) for n in nodes or ()]
    if not sets or any(s is None for s in sets):
        return None, "unresolved"
    return ([frozenset().union(*sets)] if policy == "device_local" else sets), "device_numa"


def rank_of(proc, pid):
    """LOCAL_RANK from a process's start-time environment: an int, None when absent, or "gone" when it exited.

    Raises OSError for an unreadable environment (another user's process), which makes the observation unknown.
    """
    try:
        env = (Path(proc) / str(pid) / "environ").read_bytes().split(b"\0")
    except FileNotFoundError:
        return "gone"
    for item in env:
        if item.startswith(b"LOCAL_RANK="):
            value = item[len(b"LOCAL_RANK=") :]
            return int(value) if value.isdigit() else None
    return None


def rank_violations(seen, lists, proc):
    """Threads outside their rank's list (ranks layout), or None when a rank could not be read."""
    union = frozenset().union(*lists)
    bad, ranks = 0, {}
    for pid, _, cpus in seen:
        try:
            if pid not in ranks:
                ranks[pid] = rank_of(proc, pid)
        except OSError:
            return None
        rank = ranks[pid]
        if rank == "gone":
            continue
        allowed = union if rank is None else lists[rank] if rank < len(lists) else frozenset()
        bad += not cpus <= allowed
    return bad


def verify_pinning(request, pid, device_numa=(), proc="/proc", sys_root="/sys", expected_starttime=None, markers=None):
    """run.exec.* properties for the process tree at pid under the requested cpu_pinning block, or None while every
    process of the tree is still in the launcher stage (see read_affinities; only with `markers`).

    expected_starttime: the root's /proc starttime recorded at spawn; a different value (pid reuse) gives unknown.
    """
    policy = (request or {}).get("policy", "none")
    allowed, source = resolve_requested(request, device_numa, sys_root)
    seen = read_affinities(pid, proc, expected_starttime, markers)
    if seen == []:
        return None
    props = {"run.exec.cpu_pinning": policy, "run.exec.cpu_set_source": source}
    if seen:
        props["run.exec.cpu_set"] = format_cpu_list(frozenset().union(*(c for _, _, c in seen)))
    if allowed is None or not seen:
        return {**props, "run.exec.pinning_verified": "unknown"}
    if (request or {}).get("layout") == "ranks" and policy == "per_device":
        bad = rank_violations(seen, allowed, Path(proc))
        if bad is None:
            return {**props, "run.exec.pinning_verified": "unknown"}
    else:
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
