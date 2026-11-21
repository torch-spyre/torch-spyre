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

"""CPU pinning plans: host topology + devices + policy -> CPU lists and a launch recipe.

Stdlib only, so a launcher host without the engine's dependencies can import it. Pure functions apart from
`read_topology` and `device_numa_from_pci`, the only readers of /sys (Linux). Topology and NUMA nodes come from the
HOST: a scheduler's resource slice may report -1 for a device's node.

  assign(topology, devices, Policy(...))        deterministic per-device CPU lists (the `per_device` generator)
  resolve(request, topology, device_numa)       a host.cpu_pinning request -> PinPlan (what to apply, what to verify)

  python -m spyre_bench.pinning_plan --devices 0:0,1:1 --cores-per-device 4 [--topology topo.json] [--layout ranks]
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

MAX_CPU = 16383
SPREADS = ("pack", "spread", "round_robin")
SMTS = ("primary_only", "siblings", "both")
SPREAD_SCOPES = ("all", "socket")
LAYOUTS = ("tree", "ranks")
LAUNCHERS = ("taskset", "numactl")
POLICIES = ("none", "device_local", "per_device", "explicit", "platform")
ASSIGN_KEYS = ("cores_per_device", "spread", "smt", "reserve", "reserve_cpus", "spread_scope", "llc_align")


class PinningError(ValueError):
    """The request cannot be satisfied."""


class BadTopology(PinningError):
    pass


class DeviceNumaUnknown(PinningError):
    pass


class NotEnoughCores(PinningError):
    pass


def parse_cpu_list(text):
    """'0-3,8' -> frozenset({0, 1, 2, 3, 8}). Raises ValueError on a malformed, empty, reversed or oversized list."""
    text = str(text).strip()
    out = set()
    for part in text.split(","):
        lo, dash, hi = part.partition("-")
        if not lo.isascii() or not lo.isdigit() or (dash and (not hi.isascii() or not hi.isdigit())):
            raise ValueError(f"bad CPU list {text!r}")
        a, b = int(lo), int(hi) if dash else int(lo)
        if b < a or b > MAX_CPU:
            raise ValueError(f"bad CPU range {part!r} (ids run 0-{MAX_CPU}, low to high)")
        out.update(range(a, b + 1))
    return frozenset(out)


def format_cpu_list(cpus):
    """Canonical range string: sorted, merged, no spaces."""
    runs = []
    for c in sorted(set(cpus)):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


@dataclass(frozen=True)
class Topology:
    numa_cpus: dict  # node -> frozenset of online CPUs
    smt_groups: tuple = ()  # SMT sibling sets; CPUs in none are single-thread cores
    llc_groups: tuple = ()
    sub_numa: bool = False
    node_socket: dict = field(default_factory=dict)  # node -> socket id (default: one socket)

    @classmethod
    def from_dict(cls, d):
        try:
            nodes = {
                int(k): parse_cpu_list(v) if isinstance(v, str) else frozenset(int(x) for x in v)
                for k, v in d["numa_nodes"].items()
            }
        except (KeyError, ValueError, AttributeError, TypeError) as e:
            raise BadTopology(f"topology needs numa_nodes {{node: cpu list}}: {e}") from e
        if not nodes or any(not v for v in nodes.values()):
            raise BadTopology("every NUMA node in the topology must have CPUs")
        owner = {}
        for n, cpus in nodes.items():
            for c in cpus:
                if c in owner:
                    raise BadTopology(f"cpu {c} is in NUMA nodes {owner[c]} and {n}")
                owner[c] = n
        smt = tuple(tuple(sorted(int(c) for c in g)) for g in d.get("smt_groups", ()))
        for g in smt:
            if None in {owner.get(c) for c in g} or len({owner[c] for c in g}) != 1:
                raise BadTopology(f"SMT group {list(g)} is not inside one known NUMA node")
        llc = tuple(
            parse_cpu_list(g) if isinstance(g, str) else frozenset(int(x) for x in g) for g in d.get("llc_groups", ())
        )
        socket = {int(k): int(v) for k, v in d.get("node_socket", {}).items()}
        return cls(nodes, smt, llc, bool(d.get("sub_numa", False)), socket)

    def to_dict(self):
        return {
            "numa_nodes": {str(n): format_cpu_list(c) for n, c in sorted(self.numa_cpus.items())},
            "smt_groups": [list(g) for g in self.smt_groups],
            "llc_groups": [format_cpu_list(g) for g in self.llc_groups],
            "sub_numa": self.sub_numa,
            "node_socket": {str(k): v for k, v in sorted(self.node_socket.items())},
        }

    def cores(self, node):
        """Physical cores of one node ordered by lowest CPU; a core is the sorted tuple of its hardware threads."""
        cpus = self.numa_cpus[node]
        grouped, cores = set(), []
        for g in self.smt_groups:
            if g[0] in cpus:
                cores.append(g)
                grouped.update(g)
        cores.extend((c,) for c in cpus if c not in grouped)
        return sorted(cores, key=lambda g: g[0])

    def llc_index(self, cpu):
        return next((i for i, g in enumerate(self.llc_groups) if cpu in g), None)

    def nodes_of(self, cpus):
        return sorted(n for n, c in self.numa_cpus.items() if c & set(cpus))


@dataclass(frozen=True)
class Device:
    id: str
    numa: int | None


@dataclass(frozen=True)
class Policy:
    """The generator's knobs. Every CPU count is in physical cores; `smt` decides which hardware threads they bring."""

    cores_per_device: int
    spread: str = (
        "pack"  # pack: own node, consecutive; spread: equal share of every node; round_robin: own node, strided
    )
    smt: str = "primary_only"  # first thread of each core | only the other threads | all threads
    reserve: int = 0  # lowest-numbered cores held back per NUMA node for the system and the engine
    reserve_cpus: str = ""  # extra CPU list kept out; a core touching it is removed whole
    spread_scope: str = "all"  # spread only: every node, or only the device's socket
    llc_align: bool = True  # pack only: start the next device on a last-level-cache boundary when cores allow

    def validate(self):
        if isinstance(self.cores_per_device, bool) or not isinstance(self.cores_per_device, int):
            raise PinningError("cores_per_device must be an integer >= 1")
        if self.cores_per_device < 1:
            raise PinningError("cores_per_device must be an integer >= 1")
        for name, val, allowed in (
            ("spread", self.spread, SPREADS),
            ("smt", self.smt, SMTS),
            ("spread_scope", self.spread_scope, SPREAD_SCOPES),
        ):
            if val not in allowed:
                raise PinningError(f"{name} must be one of {allowed}, got {val!r}")
        if isinstance(self.reserve, bool) or not isinstance(self.reserve, int) or self.reserve < 0:
            raise PinningError("reserve must be an integer >= 0")
        if not isinstance(self.llc_align, bool):
            raise PinningError("llc_align must be true or false")
        if self.reserve_cpus:
            try:
                parse_cpu_list(self.reserve_cpus)
            except ValueError as e:
                raise PinningError(f"reserve_cpus: {e}") from e

    @classmethod
    def from_dict(cls, d):
        extra = set(d) - set(ASSIGN_KEYS)
        if extra:
            raise PinningError(f"unknown assign keys {sorted(extra)}")
        if "cores_per_device" not in d:
            raise PinningError("assign needs cores_per_device")
        p = cls(**d)
        p.validate()
        return p


@dataclass(frozen=True)
class DeviceAssignment:
    id: str
    numa: int
    cpus: tuple
    by_node: dict

    @property
    def cpu_list(self):
        return format_cpu_list(self.cpus)


@dataclass(frozen=True)
class Assignment:
    devices: tuple
    policy: Policy
    topology: Topology

    @property
    def union(self):
        return format_cpu_list(c for d in self.devices for c in d.cpus)

    def to_dict(self):
        return {
            "policy": dict(self.policy.__dict__),
            "union": self.union,
            "devices": [
                {
                    "id": d.id,
                    "numa": d.numa,
                    "cpu_list": d.cpu_list,
                    "cpu_count": len(d.cpus),
                    "by_node": {str(k): v for k, v in sorted(d.by_node.items())},
                }
                for d in self.devices
            ],
        }


def _pool(topo, node, policy):
    held = parse_cpu_list(policy.reserve_cpus) if policy.reserve_cpus else frozenset()
    return [c for c in topo.cores(node)[policy.reserve :] if not held.intersection(c)]


def _threads(core, smt):
    return core[:1] if smt == "primary_only" else core[1:] if smt == "siblings" else core


def _take(cores, smt, dev, node):
    if smt == "siblings" and any(len(core) < 2 for core in cores):
        raise NotEnoughCores(f"device {dev}: smt=siblings but node {node} has cores without SMT siblings")
    return tuple(c for core in cores for c in _threads(core, smt))


def _next_llc_boundary(topo, pool, cursor):
    """Index of the first pool core at or after `cursor` that starts a new LLC group, or None."""
    if cursor >= len(pool):
        return None
    cur = topo.llc_index(pool[cursor - 1][0])
    return next((i for i in range(cursor, len(pool)) if topo.llc_index(pool[i][0]) != cur), None)


def assign(topo, devices, policy):
    """Deterministic: devices are served in the given order, cores of a node in ascending CPU order; sets never overlap."""
    policy.validate()
    if not devices:
        raise PinningError("no devices given")
    if len({d.id for d in devices}) != len(devices):
        raise PinningError("duplicate device ids")
    for d in devices:
        if d.numa is None or d.numa < 0:
            raise DeviceNumaUnknown(
                f"device {d.id}: NUMA node unknown ({d.numa}); read it from the host (PCI numa_node)"
            )
        if d.numa not in topo.numa_cpus:
            raise DeviceNumaUnknown(
                f"device {d.id}: NUMA node {d.numa} is not in the topology {sorted(topo.numa_cpus)}"
            )
    want = policy.cores_per_device
    pools = {n: _pool(topo, n, policy) for n in topo.numa_cpus}
    cursor = {n: 0 for n in topo.numa_cpus}
    mates = {}
    for d in devices:
        mates.setdefault(d.numa, []).append(d.id)

    def grab(node, count, dev, idxs=None):
        pool = pools[node]
        if idxs is None:
            idxs = list(range(cursor[node], min(cursor[node] + count, len(pool))))
            if len(idxs) < count:
                raise NotEnoughCores(
                    f"device {dev}: needs {count} cores on NUMA node {node} but only {len(pool) - cursor[node]} "
                    f"are left ({len(pool)} usable after reserve)"
                )
            cursor[node] += count
        return _take([pool[i] for i in idxs], policy.smt, dev, node)

    out, served = [], set()
    for d in devices:
        node = d.numa
        if policy.spread == "pack":
            if policy.llc_align and topo.llc_groups and cursor[node] > 0:
                nxt = _next_llc_boundary(topo, pools[node], cursor[node])
                left = len([x for x in mates[node] if x not in served])
                if nxt is not None and len(pools[node]) - nxt >= left * want:
                    cursor[node] = nxt
            cpus = grab(node, want, d.id)
            per = {node: len(cpus)}
        elif policy.spread == "round_robin":
            k, m = mates[node].index(d.id), len(mates[node])
            idxs = list(range(k, len(pools[node]), m))[:want]
            if len(idxs) < want:
                raise NotEnoughCores(
                    f"device {d.id}: round_robin over {m} devices on NUMA node {node} gives only {len(idxs)} of "
                    f"{want} cores ({len(pools[node])} usable after reserve)"
                )
            cpus = grab(node, want, d.id, idxs)
            per = {node: len(cpus)}
        else:
            scope = sorted(topo.numa_cpus)
            if policy.spread_scope == "socket":
                sock = topo.node_socket.get(node, 0)
                scope = [n for n in scope if topo.node_socket.get(n, 0) == sock]
            start = scope.index(node)
            order = scope[start:] + scope[:start]  # the local node first: it gets the remainder
            base, extra = divmod(want, len(order))
            got, per = [], {}
            for i, n in enumerate(order):
                count = base + (1 if i < extra else 0)
                if count:
                    part = grab(n, count, d.id)
                    got.extend(part)
                    per[n] = len(part)
            cpus = tuple(got)
        served.add(d.id)
        out.append(DeviceAssignment(d.id, node, tuple(sorted(cpus)), per))
    owner = {}
    for da in out:
        for c in da.cpus:
            if c in owner:
                raise PinningError(f"internal: cpu {c} given to devices {owner[c]} and {da.id}")
            owner[c] = da.id
    return Assignment(tuple(out), policy, topo)


# ---- request -> plan ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PinPlan:
    """What a host.cpu_pinning request means on this host.

    `source` is the run.exec.cpu_set_source value: none, explicit (a list given or generated), device_numa (the
    union of the devices' NUMA nodes) or unresolved (the request could not be turned into CPUs here).
    """

    policy: str
    source: str
    union: str | None = None
    by_device: tuple | None = None  # one CPU list per device, in device order
    layout: str = "tree"
    launcher: str = "taskset"
    require_verified: bool = False
    apply: bool = False  # False: nothing to launch under (none, unresolved, or per_device left to the stack)
    reason: str = ""  # why a plan is unresolved
    nodes: tuple = ()  # NUMA nodes the union touches, when the topology is known
    nodes_by_device: tuple | None = None

    def observer_request(self):
        """The request block the observer (placement.verify_pinning) judges the process tree against."""
        req = {"policy": self.policy, "require_verified": self.require_verified}
        if self.source == "unresolved":
            return {**req, "unresolved": True}
        if self.by_device is not None:
            # layout tree pins the whole tree to the union, so that is the one set a thread must be inside
            lists = list(self.by_device) if self.layout == "ranks" else [self.union]
            req.update(cpus_by_device=lists, layout=self.layout)
        elif self.policy == "explicit" and self.union:
            req["cpus"] = self.union
        return req

    def prefix(self, tool):
        """argv words to put before the benchmark command; tool is the launcher's path."""
        if not self.apply or not self.union:
            return []
        if self.launcher == "numactl":
            return [tool, f"--physcpubind={self.union}"] + ([f"--membind={_nodes(self.nodes)}"] if self.nodes else [])
        return [tool, "-c", self.union]

    def stage_prefixes(self, tool):
        """NUL-joined argv prefixes (with a trailing NUL) of the engine's own launcher processes, for the observer: the
        outer prefix and, in a ranks layout, each rank's tool before it execs the command."""
        words = [self.prefix(tool)]
        if self.layout == "ranks" and self.by_device:
            for cpus in self.by_device:
                words.append([tool, f"--physcpubind={cpus}"] if self.launcher == "numactl" else [tool, "-c", cpus])
        return tuple("\0".join(w) + "\0" for w in words if w)

    def wrapper(self, tool):
        """Text of the per-rank wrapper: pins by LOCAL_RANK (layout ranks), else passes the command through."""
        lines = ["#!/usr/bin/env bash", "set -eu"]
        if not (self.apply and self.layout == "ranks" and self.by_device):
            return "\n".join(lines + ['exec "$@"']) + "\n"
        lines.append('case "${LOCAL_RANK:?LOCAL_RANK not set}" in')
        for i, cpus in enumerate(self.by_device):
            nodes = self.nodes_by_device[i] if self.nodes_by_device else ()
            lines.append(f'  {i}) CPUS="{cpus}"; MEM="{_nodes(nodes)}";;')
        lines += ['  *) echo "no CPU set for LOCAL_RANK=$LOCAL_RANK" >&2; exit 2;;', "esac"]
        if self.launcher == "numactl":
            lines.append(
                f'if [ -n "$MEM" ]; then exec {shlex.quote(tool)} --physcpubind="$CPUS" --membind="$MEM" "$@"; fi'
            )
            lines.append(f'exec {shlex.quote(tool)} --physcpubind="$CPUS" "$@"')
        else:
            lines.append(f'exec {shlex.quote(tool)} -c "$CPUS" "$@"')
        return "\n".join(lines) + "\n"


def _nodes(nodes):
    return ",".join(str(n) for n in nodes)


def _device_nodes(device_numa):
    """[int] from a list or the fingerprint string; None when empty or any entry is unknown (negative, malformed)."""
    try:
        if isinstance(device_numa, str):
            device_numa = [v.strip() for v in device_numa.split(",")] if device_numa.strip() else []
        nodes = [int(v) for v in device_numa] if isinstance(device_numa, (list, tuple)) else [int(device_numa)]
    except (TypeError, ValueError):
        return None
    return nodes if nodes and all(n >= 0 for n in nodes) else None


def _unresolved(req, reason):
    return PinPlan(
        req["policy"],
        "unresolved",
        layout=req.get("layout", "tree"),
        launcher=req.get("launcher", "taskset"),
        require_verified=bool(req.get("require_verified")),
        reason=reason,
    )


def resolve(request, topology=None, device_numa=()):
    """PinPlan for a host.cpu_pinning request. topology: Topology or None (unreadable); device_numa: the host's NUMA
    node per device (list or the fingerprint's comma-separated string).

    Never raises for an unmet host condition (that is `unresolved`, with the reason); raises PinningError for a request
    that is inconsistent in itself.
    """
    req = request or {}
    policy = req.get("policy", "none")
    if policy not in POLICIES:
        raise PinningError(f"unknown cpu_pinning policy {policy!r}")
    layout, launcher = req.get("layout", "tree"), req.get("launcher", "taskset")
    if layout not in LAYOUTS or launcher not in LAUNCHERS:
        raise PinningError(f"layout must be one of {LAYOUTS} and launcher one of {LAUNCHERS}")
    required = bool(req.get("require_verified"))
    if policy == "none":
        return PinPlan("none", "none", require_verified=required)
    if policy == "platform":
        return PinPlan("platform", "platform", require_verified=required)  # the stack pins itself: nothing to launch
    common = {"layout": layout, "launcher": launcher, "require_verified": required}
    nodes = _device_nodes(device_numa)

    def known(cpus):
        return tuple(topology.nodes_of(parse_cpu_list(cpus))) if topology else ()

    if policy == "explicit":
        try:
            cpus = format_cpu_list(parse_cpu_list(req["cpus"]))
        except (KeyError, ValueError) as e:
            raise PinningError(f"explicit needs a valid cpus list: {e}") from e
        return PinPlan("explicit", "explicit", cpus, apply=True, nodes=known(cpus), **common)
    if policy == "device_local":
        if topology is None:
            return _unresolved(req, "host topology unreadable")
        if nodes is None or any(n not in topology.numa_cpus for n in nodes):
            return _unresolved(req, "device NUMA node unknown on this host")
        cpus = format_cpu_list(frozenset().union(*(topology.numa_cpus[n] for n in nodes)))
        return PinPlan("device_local", "device_numa", cpus, apply=True, nodes=tuple(sorted(set(nodes))), **common)
    lists = req.get("cpus_by_device")
    gen = req.get("assign")
    if lists and gen:
        raise PinningError("per_device takes cpus_by_device or assign, not both")
    if lists:
        try:
            by = tuple(format_cpu_list(parse_cpu_list(x)) for x in lists)
        except ValueError as e:
            raise PinningError(f"cpus_by_device: {e}") from e
        if nodes is not None and len(nodes) != len(by):
            raise PinningError(f"cpus_by_device has {len(by)} lists but the host reports {len(nodes)} device(s)")
    elif gen:
        if topology is None:
            return _unresolved(req, "host topology unreadable")
        if nodes is None:
            return _unresolved(req, "device NUMA node unknown on this host")
        devices = [Device(str(i), n) for i, n in enumerate(nodes)]
        try:
            asg = assign(topology, devices, Policy.from_dict(dict(gen)))
        except DeviceNumaUnknown as e:
            return _unresolved(req, str(e))
        by = tuple(d.cpu_list for d in asg.devices)
    else:
        if layout == "ranks":
            raise PinningError("layout ranks needs cpus_by_device or assign")
        return PinPlan("per_device", "device_numa", apply=False, **common)  # the stack pins; the observer verifies
    union = format_cpu_list(frozenset().union(*(parse_cpu_list(x) for x in by)))
    per_dev_nodes = tuple(known(x) for x in by) if topology else None
    return PinPlan(
        "per_device",
        "explicit",
        union,
        by,
        apply=True,
        nodes=known(union),
        nodes_by_device=per_dev_nodes,
        **common,
    )


# ---- host readers (Linux only) -----------------------------------------------------------------------------------


def _need_linux(sys_root):
    if str(sys_root) == "/sys" and not sys.platform.startswith("linux"):
        raise PinningError("reading host topology needs Linux (/sys); pass a topology JSON instead")


def read_topology(sys_root="/sys"):
    """Topology from /sys/devices/system/{node,cpu}. `sub_numa` is a heuristic (more CPU-bearing NUMA nodes than
    sockets); offline CPUs and memory-only nodes are dropped. Raises BadTopology / PinningError when unreadable."""
    _need_linux(sys_root)
    root = Path(sys_root) / "devices/system"
    try:
        online = parse_cpu_list((root / "cpu/online").read_text())
    except (OSError, ValueError) as e:
        raise BadTopology(f"cannot read {root / 'cpu/online'}: {e}") from e
    nodes = {}
    for p in sorted((root / "node").glob("node[0-9]*")):  # a kernel without NUMA has no node directory: one node
        try:
            cpus = parse_cpu_list((p / "cpulist").read_text()) & online
        except (OSError, ValueError):
            continue  # a memory-only node has an empty cpulist
        if cpus:
            nodes[int(p.name[4:])] = cpus
    if not nodes and not (root / "node").is_dir():
        nodes = {0: online}
    if not nodes:
        raise BadTopology(f"no NUMA nodes with CPUs under {root / 'node'}")
    smt, llc, package = set(), set(), {}
    for c in sorted(online):
        cd = root / f"cpu/cpu{c}"
        try:
            sib = parse_cpu_list((cd / "topology/thread_siblings_list").read_text()) & online
            if len(sib) > 1:
                smt.add(tuple(sorted(sib)))
        except (OSError, ValueError):
            pass
        try:
            package[c] = int((cd / "topology/physical_package_id").read_text())
        except (OSError, ValueError):
            pass
        best = None
        for idx in cd.glob("cache/index*"):
            try:
                level = int((idx / "level").read_text())
                if best is None or level > best[0]:
                    best = (level, parse_cpu_list((idx / "shared_cpu_list").read_text()) & online)
            except (OSError, ValueError):
                pass
        if best:
            llc.add(best[1])
    node_socket = {
        n: min(package[c] for c in cpus if c in package) for n, cpus in nodes.items() if any(c in package for c in cpus)
    }
    sockets = len(set(node_socket.values())) or 1
    return Topology(nodes, tuple(sorted(smt)), tuple(sorted(llc, key=min)), len(nodes) > sockets, node_socket)


def device_numa_from_pci(addr, sys_root="/sys"):
    """NUMA node of a PCI device (`0000:1a:00.0`); DeviceNumaUnknown on -1 or unreadable."""
    _need_linux(sys_root)
    path = Path(sys_root) / "bus/pci/devices" / addr / "numa_node"
    try:
        n = int(path.read_text())
    except (OSError, ValueError) as e:
        raise DeviceNumaUnknown(f"device {addr}: cannot read {path}: {e}") from e
    if n < 0:
        raise DeviceNumaUnknown(f"device {addr}: sysfs numa_node is {n} (unknown); pass the node explicitly")
    return n


def parse_devices(spec, sys_root="/sys"):
    """'0:0,1:1' (id:numa), '0:pci=0000:1a:00.0' (node from sysfs) or '0:?' (unknown; assign() rejects it)."""
    out = []
    for item in [s for s in spec.split(",") if s]:
        ident, _, rest = item.partition(":")
        if not rest:
            raise PinningError(f"device {item!r}: use id:numa")
        if rest.startswith("pci="):
            out.append(Device(ident, device_numa_from_pci(rest[4:], sys_root)))
        elif rest in ("?", "-1"):
            out.append(Device(ident, None))
        else:
            try:
                out.append(Device(ident, int(rest)))
            except ValueError as e:
                raise PinningError(f"device {item!r}: numa must be an integer") from e
    return out


# ---- CLI ---------------------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spyre_bench.pinning_plan", description=__doc__.split("\n")[0])
    ap.add_argument("--topology", help="topology JSON file; omit to read /sys (Linux)")
    ap.add_argument("--sys-root", default="/sys")
    ap.add_argument("--devices", required=True, help="id:numa[,...]; id:pci=ADDR reads the node from sysfs")
    ap.add_argument("--cores-per-device", type=int, required=True)
    ap.add_argument("--spread", choices=SPREADS)
    ap.add_argument("--smt", choices=SMTS)
    ap.add_argument("--reserve", type=int)
    ap.add_argument("--reserve-cpus")
    ap.add_argument("--spread-scope", choices=SPREAD_SCOPES)
    ap.add_argument("--no-llc-align", action="store_true")
    ap.add_argument("--layout", choices=LAYOUTS, default="tree")
    ap.add_argument("--launcher", choices=LAUNCHERS, default="taskset")
    ap.add_argument("--format", choices=("json", "wrapper", "cpulist"), default="json")
    ns = ap.parse_args(argv)
    knobs = {
        "cores_per_device": ns.cores_per_device,
        "spread": ns.spread,
        "smt": ns.smt,
        "reserve": ns.reserve,
        "reserve_cpus": ns.reserve_cpus,
        "spread_scope": ns.spread_scope,
        "llc_align": False if ns.no_llc_align else None,
    }
    try:
        topo = (
            Topology.from_dict(json.loads(Path(ns.topology).read_text())) if ns.topology else read_topology(ns.sys_root)
        )
        policy = Policy.from_dict({k: v for k, v in knobs.items() if v is not None})
        a = assign(topo, parse_devices(ns.devices, ns.sys_root), policy)
        plan = resolve(
            {"policy": "per_device", "assign": dict(policy.__dict__), "layout": ns.layout, "launcher": ns.launcher},
            topo,
            [d.numa for d in a.devices],
        )
    except (PinningError, OSError, json.JSONDecodeError) as e:
        print(f"pinning_plan: {e}", file=sys.stderr)
        return 2
    if ns.format == "cpulist":
        print(plan.union if ns.layout == "tree" else "\n".join(plan.by_device))
    elif ns.format == "wrapper":
        sys.stdout.write(plan.wrapper(ns.launcher))
    else:
        print(json.dumps({"assignment": a.to_dict(), "plan": plan.__dict__}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
