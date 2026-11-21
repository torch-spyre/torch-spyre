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

"""CPU pinning: the plan generator, the launcher, the observer with per-device lists, and the run integration.
python -m pytest -q -p no:cacheprovider tests/test_pinning.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from spyre_bench import analysis, engine, pinning_plan as pp, pinning_run, placement
from spyre_bench import plan as plan_cmd
from spyre_bench import run as run_cmd
from test_engine import fake_proc, fake_sys

EXT = Path(__file__).resolve().parent.parent
DEMO = EXT / "examples" / "pinning-demo"
TMP = Path(tempfile.mkdtemp(prefix="spyre-bench-pin-tests-"))
LINUX = sys.platform.startswith("linux") and shutil.which("taskset") is not None


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def two_socket(smt=True):
    """2 sockets x 36 cores, 2 NUMA nodes; the SMT sibling of cpu N is N+72."""
    d = {
        "numa_nodes": {"0": "0-35,72-107", "1": "36-71,108-143"} if smt else {"0": "0-35", "1": "36-71"},
        "node_socket": {"0": 0, "1": 1},
    }
    if smt:
        d["smt_groups"] = [[i, i + 72] for i in range(72)]
    return pp.Topology.from_dict(d)


def snc_one_socket():
    """1 socket x 48 cores, SMT (N, N+96), four sub-NUMA nodes with one LLC slice each."""
    nodes = {str(n): f"{n * 12}-{n * 12 + 11},{96 + n * 12}-{96 + n * 12 + 11}" for n in range(4)}
    return pp.Topology.from_dict(
        {
            "numa_nodes": nodes,
            "node_socket": {str(n): 0 for n in range(4)},
            "sub_numa": True,
            "smt_groups": [[i, i + 96] for i in range(48)],
            "llc_groups": [nodes[str(n)] for n in range(4)],
        }
    )


def devs(spec):
    return pp.parse_devices(spec)


def lists(a):
    return {d.id: d.cpu_list for d in a.devices}


class Assign(unittest.TestCase):
    def test_pack_one_node_and_two_nodes(self):
        self.assertEqual(lists(pp.assign(two_socket(), devs("0:0,1:0"), pp.Policy(4))), {"0": "0-3", "1": "4-7"})
        a = pp.assign(two_socket(), devs("0:0,1:1"), pp.Policy(8))
        self.assertEqual(lists(a), {"0": "0-7", "1": "36-43"})

    def test_spread_two_card_tp(self):
        a = pp.assign(two_socket(), devs("0:0,1:1"), pp.Policy(4, "spread"))
        self.assertEqual(lists(a), {"0": "0-1,36-37", "1": "2-3,38-39"})
        self.assertEqual(a.devices[0].by_node, {0: 2, 1: 2})
        self.assertEqual(a.union, "0-3,36-39")

    def test_spread_remainder_goes_to_the_local_node_and_scope_socket(self):
        a = pp.assign(two_socket(), devs("0:0,1:1"), pp.Policy(5, "spread"))
        self.assertEqual((a.devices[0].by_node, a.devices[1].by_node), ({0: 3, 1: 2}, {1: 3, 0: 2}))
        b = pp.assign(two_socket(), devs("0:0"), pp.Policy(4, "spread", spread_scope="socket"))
        self.assertEqual(b.devices[0].by_node, {0: 4})

    def test_smt_policies(self):
        t = two_socket()
        got = {m: pp.assign(t, devs("0:0"), pp.Policy(4, smt=m)).devices[0].cpu_list for m in pp.SMTS}
        self.assertEqual(got, {"primary_only": "0-3", "siblings": "72-75", "both": "0-3,72-75"})
        with self.assertRaises(pp.NotEnoughCores):
            pp.assign(two_socket(smt=False), devs("0:0"), pp.Policy(2, smt="siblings"))
        self.assertEqual(pp.assign(two_socket(smt=False), devs("0:0"), pp.Policy(3, smt="both")).union, "0-2")

    def test_reserve(self):
        t = two_socket()
        self.assertEqual(lists(pp.assign(t, devs("0:0,1:1"), pp.Policy(2, reserve=2))), {"0": "2-3", "1": "38-39"})
        self.assertEqual(pp.assign(t, devs("0:0"), pp.Policy(2, reserve_cpus="0,73")).union, "2-3")

    def test_round_robin(self):
        a = pp.assign(two_socket(), devs("0:0,1:0"), pp.Policy(3, "round_robin"))
        self.assertEqual(lists(a), {"0": "0,2,4", "1": "1,3,5"})

    def test_llc_alignment(self):
        topo = pp.Topology.from_dict({"numa_nodes": {"0": "0-15"}, "llc_groups": ["0-7", "8-15"]})
        self.assertEqual(lists(pp.assign(topo, devs("0:0,1:0"), pp.Policy(5))), {"0": "0-4", "1": "8-12"})
        self.assertEqual(
            lists(pp.assign(topo, devs("0:0,1:0"), pp.Policy(5, llc_align=False))), {"0": "0-4", "1": "5-9"}
        )
        three = pp.assign(topo, devs("0:0,1:0,2:0"), pp.Policy(5))
        self.assertEqual(lists(three), {"0": "0-4", "1": "5-9", "2": "10-14"})

    def test_sub_numa_spread(self):
        a = pp.assign(snc_one_socket(), devs("0:0"), pp.Policy(8, "spread"))
        self.assertEqual(a.devices[0].by_node, {0: 2, 1: 2, 2: 2, 3: 2})

    def test_errors(self):
        t = two_socket()
        with self.assertRaisesRegex(pp.NotEnoughCores, "NUMA node 0"):
            pp.assign(t, devs("0:0,1:0"), pp.Policy(20))
        for spec in ("0:?", "0:-1"):
            with self.assertRaisesRegex(pp.DeviceNumaUnknown, "unknown"):
                pp.assign(t, devs(spec), pp.Policy(2))
        with self.assertRaisesRegex(pp.DeviceNumaUnknown, "not in the topology"):
            pp.assign(t, devs("0:5"), pp.Policy(2))
        with self.assertRaisesRegex(pp.PinningError, "duplicate"):
            pp.assign(t, devs("0:0,0:1"), pp.Policy(2))
        for kw in ({"cores_per_device": 0}, {"cores_per_device": 2, "spread": "x"}, {"cores_per_device": True}):
            with self.assertRaises(pp.PinningError):
                pp.assign(t, devs("0:0"), pp.Policy(**kw))
        with self.assertRaisesRegex(pp.PinningError, "unknown assign keys"):
            pp.Policy.from_dict({"cores_per_device": 2, "bogus": 1})
        with self.assertRaises(pp.BadTopology):
            pp.Topology.from_dict({"numa_nodes": {"0": "0-3", "1": "3-5"}})

    def test_deterministic_and_disjoint(self):
        t = two_socket()
        runs = {json.dumps(pp.assign(t, devs("0:0,1:1,2:0,3:1"), pp.Policy(4, "spread")).to_dict()) for _ in range(3)}
        self.assertEqual(len(runs), 1)
        cpus = [c for d in pp.assign(t, devs("0:0,1:1,2:0,3:1"), pp.Policy(4, "spread")).devices for c in d.cpus]
        self.assertEqual(len(cpus), len(set(cpus)))

    def test_cpu_list_round_trip(self):
        for s in ("0-3,8", "5", "0-3,36-39", "1,3,5"):
            self.assertEqual(pp.format_cpu_list(pp.parse_cpu_list(s)), s)
        for bad in ("", "3-1", "a", "1,,2", "1-", "99999"):
            with self.assertRaises(ValueError):
                pp.parse_cpu_list(bad)
        self.assertEqual(pp.parse_cpu_list("0-2,5"), placement.parse_cpu_list("0-2,5"))

    def test_topology_dict_round_trip(self):
        for t in (two_socket(), snc_one_socket()):
            self.assertEqual(pp.Topology.from_dict(t.to_dict()), t)


def make_sysfs(root, nodes, online, pci=None, smt=True):
    s = root / "devices/system"
    (s / "cpu").mkdir(parents=True)
    (s / "cpu/online").write_text(online + "\n")
    for n, cl in nodes.items():
        (s / f"node/node{n}").mkdir(parents=True)
        (s / f"node/node{n}/cpulist").write_text(cl + "\n")
    for c in pp.parse_cpu_list(online):
        cd = s / f"cpu/cpu{c}"
        (cd / "topology").mkdir(parents=True)
        (cd / "topology/thread_siblings_list").write_text((f"{c % 8},{c % 8 + 8}" if smt else str(c)) + "\n")
        (cd / "topology/physical_package_id").write_text(f"{(c % 8) // 4}\n")
        (cd / "cache/index0").mkdir(parents=True)
        (cd / "cache/index0/level").write_text("3\n")
        (cd / "cache/index0/shared_cpu_list").write_text(("0-3,8-11" if c % 8 < 4 else "4-7,12-14") + "\n")
    for addr, val in (pci or {}).items():
        (root / "bus/pci/devices" / addr).mkdir(parents=True)
        (root / "bus/pci/devices" / addr / "numa_node").write_text(f"{val}\n")


class Readers(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=TMP))
        make_sysfs(
            self.root,
            {0: "0-3,8-11", 1: "4-7,12-15", 2: ""},
            "0-14",
            pci={"0000:1a:00.0": 0, "0000:3b:00.0": 1, "0000:5e:00.0": -1},
        )

    def test_read_topology(self):
        t = pp.read_topology(self.root)
        self.assertEqual(sorted(t.numa_cpus), [0, 1])
        self.assertEqual(pp.format_cpu_list(t.numa_cpus[1]), "4-7,12-14")
        self.assertIn((0, 8), t.smt_groups)
        self.assertEqual(t.node_socket, {0: 0, 1: 1})
        self.assertFalse(t.sub_numa)
        a = pp.assign(t, [pp.Device("0", 0), pp.Device("1", 1)], pp.Policy(2))
        self.assertEqual(lists(a), {"0": "0-1", "1": "4-5"})

    def test_no_numa_directory_is_one_node(self):
        bare = Path(tempfile.mkdtemp(dir=TMP))
        (bare / "devices/system/cpu").mkdir(parents=True)
        (bare / "devices/system/cpu/online").write_text("0-7\n")
        self.assertEqual(pp.format_cpu_list(pp.read_topology(bare).numa_cpus[0]), "0-7")
        self.assertEqual(placement.node_cpus(0, bare), frozenset(range(8)))
        self.assertIsNone(placement.node_cpus(1, bare))

    def test_unreadable_topology(self):
        with self.assertRaises(pp.BadTopology):
            pp.read_topology(TMP / "nope")

    def test_pci_numa(self):
        self.assertEqual(pp.device_numa_from_pci("0000:3b:00.0", self.root), 1)
        for addr in ("0000:5e:00.0", "0000:ff:00.0"):
            with self.assertRaises(pp.DeviceNumaUnknown):
                pp.device_numa_from_pci(addr, self.root)
        self.assertEqual([d.numa for d in pp.parse_devices("0:pci=0000:1a:00.0,1:pci=0000:3b:00.0", self.root)], [0, 1])

    def test_linux_guard(self):
        with mock.patch.object(pp.sys, "platform", "darwin"):
            with self.assertRaisesRegex(pp.PinningError, "Linux"):
                pp.read_topology()

    def test_cli(self):
        topo = self.root / "topo.json"
        topo.write_text(json.dumps(two_socket().to_dict()))
        base = ["--topology", str(topo), "--devices", "0:0,1:1", "--cores-per-device", "4"]
        for argv, expect in (
            (base + ["--format", "cpulist"], "0-3,36-39\n"),
            (base + ["--layout", "ranks", "--format", "cpulist"], "0-3\n36-39\n"),
        ):
            with mock.patch("sys.stdout") as out:
                self.assertEqual(pp.main(argv), 0)
            self.assertEqual("".join(c.args[0] for c in out.write.call_args_list), expect)
        self.assertEqual(pp.main(["--topology", str(topo), "--devices", "0:?", "--cores-per-device", "4"]), 2)
        self.assertEqual(pp.main(base + ["--format", "json", "--spread", "spread", "--layout", "ranks"]), 0)


class Resolve(unittest.TestCase):
    def test_none_explicit_device_local(self):
        self.assertEqual((pp.resolve(None).policy, pp.resolve({"policy": "none"}).source), ("none", "none"))
        p = pp.resolve({"policy": "explicit", "cpus": "3,0-1"})
        self.assertEqual((p.union, p.source, p.apply), ("0-1,3", "explicit", True))
        d = pp.resolve({"policy": "device_local"}, two_socket(), "1")
        self.assertEqual((d.union, d.source, d.nodes), ("36-71,108-143", "device_numa", (1,)))
        both = pp.resolve({"policy": "device_local"}, two_socket(), [0, 1])
        self.assertEqual(both.union, "0-143")

    def test_unresolved_is_a_plan_not_an_error(self):
        for numa in ("", "-1", "0,-1", "9"):
            p = pp.resolve({"policy": "device_local"}, two_socket(), numa)
            self.assertEqual((p.source, p.apply), ("unresolved", False), numa)
        self.assertEqual(pp.resolve({"policy": "device_local"}, None, "0").source, "unresolved")
        self.assertEqual(
            pp.resolve({"policy": "per_device", "assign": {"cores_per_device": 2}}, None, "0").source, "unresolved"
        )

    def test_per_device_generated_and_listed(self):
        g = pp.resolve(
            {"policy": "per_device", "assign": {"cores_per_device": 4, "spread": "spread"}}, two_socket(), "0,1"
        )
        self.assertEqual((g.by_device, g.union, g.source), (("0-1,36-37", "2-3,38-39"), "0-3,36-39", "explicit"))
        self.assertEqual(g.nodes_by_device, ((0, 1), (0, 1)))
        lst = pp.resolve({"policy": "per_device", "cpus_by_device": ["0-3", "8-11"]}, None, "0,1")
        self.assertEqual((lst.by_device, lst.apply), (("0-3", "8-11"), True))
        stack = pp.resolve({"policy": "per_device"}, two_socket(), "0,1")
        self.assertEqual((stack.apply, stack.source, stack.by_device), (False, "device_numa", None))

    def test_inconsistent_requests_raise(self):
        for req, numa in (
            ({"policy": "per_device", "cpus_by_device": ["0"], "assign": {"cores_per_device": 1}}, "0"),
            ({"policy": "per_device", "cpus_by_device": ["0"]}, "0,1"),
            ({"policy": "per_device", "layout": "ranks"}, "0"),
            ({"policy": "bogus"}, ""),
            ({"policy": "explicit", "cpus": "0", "launcher": "x"}, ""),
        ):
            with self.assertRaises(pp.PinningError, msg=str(req)):
                pp.resolve(req, two_socket(), numa)

    def test_observer_request_carries_the_lists(self):
        g = pp.resolve(
            {"policy": "per_device", "cpus_by_device": ["0-1", "4-5"], "layout": "ranks", "require_verified": True}
        )
        tree = pp.resolve({"policy": "per_device", "cpus_by_device": ["0-1", "4-5"]})
        self.assertEqual(tree.observer_request()["cpus_by_device"], ["0-1,4-5"])
        self.assertEqual(
            g.observer_request(),
            {"policy": "per_device", "require_verified": True, "cpus_by_device": ["0-1", "4-5"], "layout": "ranks"},
        )
        self.assertEqual(pp.resolve({"policy": "explicit", "cpus": "0-3"}).observer_request()["cpus"], "0-3")

    def test_golden_launcher_commands(self):
        t = two_socket()
        tree = pp.resolve({"policy": "explicit", "cpus": "0-3"}, t)
        self.assertEqual(tree.prefix("/usr/bin/taskset"), ["/usr/bin/taskset", "-c", "0-3"])
        num = pp.resolve({"policy": "device_local", "launcher": "numactl"}, t, "1")
        self.assertEqual(
            num.prefix("/usr/bin/numactl"), ["/usr/bin/numactl", "--physcpubind=36-71,108-143", "--membind=1"]
        )
        self.assertEqual(pp.resolve({"policy": "none"}).prefix("taskset"), [])
        self.assertEqual(pp.resolve({"policy": "device_local"}, t, "").prefix("taskset"), [])

    def test_stage_prefixes_cover_the_rank_tools_exactly(self):
        p = pp.resolve({"policy": "per_device", "cpus_by_device": ["0-1", "4-5"], "layout": "ranks"})
        want = tuple("\0".join(["/t", "-c", c]) + "\0" for c in ("0-1,4-5", "0-1", "4-5"))
        self.assertEqual(p.stage_prefixes("/t"), want)
        for bad in (
            {"policy": "explicit"},
            {"policy": "explicit", "cpus": "x"},
            {"policy": "per_device", "cpus_by_device": ["x"]},
        ):
            with self.assertRaises(pp.PinningError):
                pp.resolve(bad)

    def test_golden_rank_wrapper(self):
        p = pp.resolve(
            {"policy": "per_device", "assign": {"cores_per_device": 2}, "layout": "ranks"}, two_socket(), "0,1"
        )
        self.assertEqual(
            p.wrapper("/usr/bin/taskset"),
            "#!/usr/bin/env bash\nset -eu\n"
            'case "${LOCAL_RANK:?LOCAL_RANK not set}" in\n'
            '  0) CPUS="0-1"; MEM="0";;\n'
            '  1) CPUS="36-37"; MEM="1";;\n'
            '  *) echo "no CPU set for LOCAL_RANK=$LOCAL_RANK" >&2; exit 2;;\n'
            "esac\n"
            'exec /usr/bin/taskset -c "$CPUS" "$@"\n',
        )
        self.assertEqual(pp.resolve({"policy": "none"}).wrapper("x"), '#!/usr/bin/env bash\nset -eu\nexec "$@"\n')
        n = pp.resolve(
            {"policy": "per_device", "cpus_by_device": ["0-1", "4-5"], "layout": "ranks", "launcher": "numactl"}
        )
        self.assertIn('--physcpubind="$CPUS"', n.wrapper("numactl"))


class Observer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=TMP))
        self.proc, self.sysd = self.tmp / "proc", self.tmp / "sys"
        self.proc.mkdir()
        fake_sys(self.sysd, {0: "0-3", 1: "4-7"})

    def env(self, pid, rank=None, cmdline=None):
        d = self.proc / str(pid)
        (d / "environ").write_bytes(
            b"\0".join([b"A=1"] + ([f"LOCAL_RANK={rank}".encode()] if rank is not None else [])) + b"\0"
        )
        if cmdline:
            (d / "cmdline").write_bytes(b"\0".join(c.encode() for c in cmdline) + b"\0")

    def verify(self, req, pid=100, numa=(), **kw):
        return placement.verify_pinning(req, pid, numa, self.proc, self.sysd, **kw)

    def test_spread_lists_verify_as_per_device(self):
        fake_proc(self.proc, {100: (5, "run", {100: "0-1,4-5", 101: "2-3,6-7"})})
        req = {"policy": "per_device", "cpus_by_device": ["0-1,4-5", "2-3,6-7"]}
        r = self.verify(req)
        self.assertEqual((r["run.exec.pinning_verified"], r["run.exec.cpu_set_source"]), ("true", "explicit"))
        fake_proc(self.proc, {100: (5, "run", {100: "0-1,4-5", 101: "0-3"})})
        self.assertEqual(self.verify(req)["run.exec.pinning_violations"], "1")

    def test_ranks_judged_by_their_own_list(self):
        fake_proc(
            self.proc,
            {
                100: (5, "agent", {100: "0-3"}, {"pgrp": 100}),
                200: (100, "r0", {200: "0-1"}, {"pgrp": 100}),
                201: (100, "r1", {201: "2-3"}, {"pgrp": 100}),
            },
        )
        self.env(100)
        self.env(200, 0)
        self.env(201, 1)
        req = {"policy": "per_device", "cpus_by_device": ["0-1", "2-3"], "layout": "ranks"}
        self.assertEqual(self.verify(req)["run.exec.pinning_verified"], "true")
        self.env(201, 0)  # rank 1's process claims rank 0: its mask 2-3 is outside list 0-1
        r = self.verify(req)
        self.assertEqual((r["run.exec.pinning_verified"], r["run.exec.pinning_violations"]), ("false", "1"))
        self.env(201, 5)  # a rank with no list
        self.assertEqual(self.verify(req)["run.exec.pinning_verified"], "false")
        self.env(201, 1)
        fake_proc(self.proc, {100: (5, "agent", {100: "0-7"}, {"pgrp": 100})})  # the agent outside the union
        self.assertEqual(self.verify(req)["run.exec.pinning_verified"], "false")

    def test_unreadable_rank_environment_is_unknown(self):
        fake_proc(self.proc, {100: (5, "r", {100: "0-1"})})
        (self.proc / "100" / "environ").write_bytes(b"")
        with mock.patch("pathlib.Path.read_bytes", side_effect=PermissionError):
            r = self.verify({"policy": "per_device", "cpus_by_device": ["0-1"], "layout": "ranks"})
        self.assertEqual(r["run.exec.pinning_verified"], "unknown")

    def test_launcher_stage_is_not_judged_and_not_a_sample(self):
        fake_proc(self.proc, {100: (5, "taskset", {100: "0-7"})})
        self.env(100, cmdline=["/usr/bin/taskset", "-c", "0-1", "python"])
        ours = "\0".join(["/usr/bin/taskset", "-c", "0-1"]) + "\0"
        self.assertIsNone(self.verify({"policy": "explicit", "cpus": "0-1"}, markers=(ours,)))
        self.env(100, cmdline=["/usr/bin/taskset", "-c", "2-3", "python"])  # the benchmark's own taskset is judged
        self.assertEqual(
            self.verify({"policy": "explicit", "cpus": "0-1"}, markers=(ours,))["run.exec.pinning_verified"], "false"
        )
        self.env(100, cmdline=["/usr/bin/taskset", "-c", "0-1", "python"])
        self.assertEqual(self.verify({"policy": "explicit", "cpus": "0-1"})["run.exec.pinning_verified"], "false")
        self.env(100, cmdline=["bash", "/x/pin-wrapper.sh", "python"])
        self.assertIsNone(self.verify({"policy": "explicit", "cpus": "0-1"}, markers=("/x/pin-wrapper.sh",)))
        self.env(100, cmdline=["python", "bench.py"])
        self.assertEqual(
            self.verify({"policy": "explicit", "cpus": "0-1"}, markers=())["run.exec.pinning_verified"], "false"
        )

    def test_starttime_and_observer_pid(self):
        fake_proc(self.proc, {100: (5, "r", {100: "0-1"}, {"start": 77})})
        self.assertEqual(placement.starttime(100, self.proc), 77)
        self.assertIsNone(placement.starttime(999, self.proc))
        r = self.verify({"policy": "explicit", "cpus": "0-1"}, expected_starttime=78)
        self.assertEqual(r["run.exec.pinning_verified"], "unknown")
        # a direct child of pid 1 that is the observer itself is not an orphan
        me = os.getpid()
        fake_proc(
            self.proc,
            {100: (1, "r", {100: "0-1"}), me: (4000, "engine", {me: "0-7"}), 4000: (1, "make", {4000: "0-7"})},
        )
        r = self.verify({"policy": "explicit", "cpus": "0-1"})
        self.assertEqual(r["run.exec.pinning_verified"], "true")
        fake_proc(self.proc, {5000: (1, "stray", {5000: "0-7"})})  # a real orphan still makes the tree unknowable
        self.assertEqual(self.verify({"policy": "explicit", "cpus": "0-1"})["run.exec.pinning_verified"], "unknown")


class Planning(unittest.TestCase):
    def args(self, *argv):
        return plan_cmd.parser().parse_args(argv)

    def test_build_pinning(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(plan_cmd.build_pinning(self.args()))
            got = plan_cmd.build_pinning(
                self.args(
                    "--cpu-pinning",
                    "per_device",
                    "--pin-opt",
                    "cores_per_device=4",
                    "--pin-opt",
                    "spread=spread",
                    "--pin-opt",
                    "layout=ranks",
                    "--require-pinning",
                )
            )
            self.assertEqual(
                got,
                {
                    "policy": "per_device",
                    "layout": "ranks",
                    "assign": {"cores_per_device": 4, "spread": "spread"},
                    "require_verified": True,
                },
            )
            lists_ = plan_cmd.build_pinning(
                self.args("--cpu-pinning", "per_device", "--pin-opt", "cpus_by_device=0-3,8;4-7")
            )
            self.assertEqual(lists_["cpus_by_device"], ["0-3,8", "4-7"])
            for bad in (
                ("--cpu-set", "0-3"),
                ("--cpu-pinning", "explicit", "--pin-opt", "nope=1"),
                ("--cpu-pinning", "per_device", "--pin-opt", "reserve=x"),
            ):
                with self.assertRaises(plan_cmd.UsageError):
                    plan_cmd.build_pinning(self.args(*bad))
        with mock.patch.dict(
            os.environ, {"BENCH_CPU_PINNING": "explicit", "BENCH_CPU_SET": "0-3", "BENCH_REQUIRE_PINNING": "1"}
        ):
            self.assertEqual(
                plan_cmd.build_pinning(self.args()), {"policy": "explicit", "cpus": "0-3", "require_verified": True}
            )

    def plan(self, *argv):
        a = plan_cmd.parser().parse_args(
            ["--catalog", str(DEMO), "--purpose", "on-demand", "--arch", "x86_64", "--devices", "spyre/pf=2", *argv]
        )
        cat = engine.Catalog([str(DEMO)], [], [])
        req = plan_cmd.build_request(a)
        return plan_cmd.make_plan(
            cat,
            req,
            plan_cmd.build_leg(a),
            mock.Mock(stored_results=lambda *_: [], invalidations=lambda *_: []),
            mock.ANY if False else __import__("datetime").datetime(2026, 11, 16),
        )

    def test_request_replaces_the_environment_block_and_labels(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            doc = self.plan("--cpu-pinning", "explicit", "--cpu-set", "0-3")
            rec = doc["cases"][0]
            self.assertEqual(rec["pinning"], {"policy": "explicit", "cpus": "0-3"})
            self.assertEqual(rec["labels"]["env.label.cpu_pinning"], "explicit")
            self.assertEqual(
                self.plan("--environment", "node-a-device-local")["cases"][0]["pinning"]["policy"], "device_local"
            )
            with self.assertRaises(engine.PlanError):
                self.plan("--cpu-pinning", "explicit")
            with self.assertRaises(engine.PlanError):
                self.plan("--cpu-pinning", "per_device", "--pin-opt", "layout=ranks")

    def test_schema_accepts_the_new_fields(self):
        from jsonschema import Draft202012Validator

        schema = json.loads((EXT / "spyre_bench/core/schemas/environment.schema.json").read_text())
        block = (
            schema["properties"]["spec"]["properties"]["host"]["properties"]["cpu_pinning"]
            if "properties" in schema.get("properties", {}).get("spec", {})
            else None
        )
        self.assertIsNotNone(block)
        for key in ("cpus_by_device", "assign", "layout", "launcher"):
            self.assertIn(key, block["properties"])
        Draft202012Validator.check_schema(schema)

    def test_pin_problems(self):
        ok = {"policy": "per_device", "assign": {"cores_per_device": 2}, "layout": "ranks"}
        self.assertEqual(engine.pin_problems(ok, 2), [])
        self.assertTrue(engine.pin_problems({"policy": "explicit", "cpus": "0", "assign": {"cores_per_device": 1}}, 1))
        self.assertTrue(engine.pin_problems({"policy": "per_device", "assign": {"cores_per_device": 0}}, 1))
        self.assertTrue(engine.pin_problems({"policy": "none", "launcher": "taskset"}, 1))
        self.assertTrue(engine.pin_problems({"policy": "per_device", "cpus_by_device": ["x"]}, 1))
        self.assertTrue(engine.pin_problems({"policy": "per_device", "cpus_by_device": "0-3"}, 1))


class PlatformPolicy(unittest.TestCase):
    def test_platform_is_recorded_and_never_judged(self):
        p = pp.resolve({"policy": "platform"})
        self.assertEqual((p.policy, p.source, p.apply, p.prefix("taskset")), ("platform", "platform", False, []))
        proc = Path(tempfile.mkdtemp(dir=TMP))
        fake_proc(proc, {100: (5, "run", {100: "0-7"}), 101: (100, "w", {101: "0-1"})})
        r = placement.verify_pinning(p.observer_request(), 100, (), proc, proc)
        self.assertEqual(
            (r["run.exec.cpu_pinning"], r["run.exec.cpu_set_source"], r["run.exec.pinning_verified"]),
            ("platform", "platform", "unknown"),
        )
        self.assertNotIn("run.exec.pinning_violations", r)

    def test_platform_cannot_be_required_verified(self):
        self.assertTrue(engine.pin_problems({"policy": "platform", "require_verified": True}, 1))
        self.assertEqual(engine.pin_problems({"policy": "platform"}, 0), [])
        self.assertEqual(engine.PIN_LABEL, "env.label.cpu_pinning")

    def test_platform_gets_its_own_series_label(self):
        cat = engine.Catalog([str(DEMO)], [], [])
        env = engine.overlay_pinning(cat.effective("Environment", "node-a-unpinned"), {"policy": "platform"})
        self.assertEqual(cat.env_labels(env)[engine.PIN_LABEL], "platform")
        self.assertNotEqual(
            engine.request_digest(cat, "node-a-unpinned", {"policy": "platform"}),
            engine.request_digest(cat, "node-a-unpinned", None),
        )
        self.assertEqual(
            engine.request_digest(cat, "node-a-unpinned", None), cat.digest("Environment", "node-a-unpinned")
        )

    def test_a_benchmark_that_pins_itself_is_judged_by_its_masks(self):
        proc = Path(tempfile.mkdtemp(dir=TMP))
        fake_proc(proc, {100: (5, "taskset", {100: "0-1"})})
        (proc / "100" / "cmdline").write_bytes(b"taskset\0-c\0000-1\0python\0")
        ours = "\0".join(["/usr/bin/taskset", "-c", "2-3"]) + "\0"
        r = placement.verify_pinning({"policy": "explicit", "cpus": "2-3"}, 100, (), proc, proc, markers=(ours,))
        self.assertEqual(r["run.exec.pinning_verified"], "false")


class Prepare(unittest.TestCase):
    def test_not_applied_records_unresolved_and_a_note(self):
        with mock.patch.object(pinning_run.sys, "platform", "darwin"):
            p = pinning_run.prepare({"policy": "explicit", "cpus": "0-1", "require_verified": True})
        self.assertFalse(p.launches)
        self.assertEqual(p.wrap(["x"], TMP, False), ["x"])
        self.assertEqual(
            p.recorded(None),
            {
                "run.exec.cpu_pinning": "explicit",
                "run.exec.cpu_set_source": "unresolved",
                "run.exec.pinning_verified": "unknown",
            },
        )
        self.assertTrue(p.notes)

    def test_missing_tool_is_unknown_not_a_violation(self):
        with mock.patch.object(pinning_run.sys, "platform", "linux"):
            p = pinning_run.prepare(
                {"policy": "explicit", "cpus": "0-3", "require_verified": True}, which=lambda t: None
            )
        self.assertFalse(p.launches)
        self.assertEqual(p.observer_request(), {"policy": "explicit", "require_verified": True, "unresolved": True})
        proc = Path(tempfile.mkdtemp(dir=TMP))
        fake_proc(proc, {100: (5, "run", {100: "0-7"})})
        r = placement.verify_pinning(p.observer_request(), 100, (), proc, proc)
        self.assertEqual((r["run.exec.pinning_verified"], r["run.exec.cpu_set_source"]), ("unknown", "unresolved"))

    def test_golden_wrap_with_a_fake_tool(self):
        with mock.patch.object(pinning_run.sys, "platform", "linux"):
            p = pinning_run.prepare({"policy": "explicit", "cpus": "0-1"}, which=lambda t: f"/usr/bin/{t}")
        self.assertEqual(p.wrap(["python", "b.py"], TMP, False), ["/usr/bin/taskset", "-c", "0-1", "python", "b.py"])
        self.assertEqual(p.recorded(None)["run.exec.cpu_set"], "0-1")
        d = Path(tempfile.mkdtemp(dir=TMP))
        p.wrap(["x"], d, True)
        self.assertTrue(os.access(d / "pin-wrapper.sh", os.X_OK))

    def test_none_records_the_policy_for_compare(self):
        p = pinning_run.prepare(None)
        self.assertFalse(p.observes)
        self.assertEqual(p.recorded(None), {"run.exec.cpu_pinning": "none", "run.exec.cpu_set_source": "none"})
        self.assertTrue(pinning_run.prepare({"policy": "none", "require_verified": True}).observes)


@unittest.skipUnless(LINUX, "needs Linux with taskset: real pinning and the /proc observer")
class EndToEnd(unittest.TestCase):
    """Real taskset, real /proc, the stub benchmark; the topology is described by a fake sysfs (two nodes)."""

    @classmethod
    def setUpClass(cls):
        cls.cpus = sorted(os.sched_getaffinity(0))
        if len(cls.cpus) < 2:
            raise unittest.SkipTest("needs at least 2 CPUs")
        half = len(cls.cpus) // 2
        cls.lo, cls.hi = cls.cpus[:half], cls.cpus[half:]
        cls.sysd = TMP / "e2e-sys"
        fake_sys(cls.sysd, {0: pp.format_cpu_list(cls.lo), 1: pp.format_cpu_list(cls.hi)})
        (cls.sysd / "devices/system/cpu").mkdir(parents=True)
        (cls.sysd / "devices/system/cpu/online").write_text(pp.format_cpu_list(cls.cpus) + "\n")
        cls.env = TMP / "e2e-env.json"
        cls.env.write_text(json.dumps({"node": "n1", "fingerprint": {"env.fp.device.numa_node": "0,1"}}))

    def run_policy(self, suite, *opts, name):
        out = TMP / f"e2e-{name}"
        shutil.rmtree(out, ignore_errors=True)
        plan = TMP / f"e2e-{name}.json"
        with mock.patch.dict(
            os.environ, {k: v for k, v in os.environ.items() if not k.startswith("BENCH_")}, clear=True
        ):
            self.assertEqual(
                plan_cmd.main(
                    [
                        "--catalog",
                        str(DEMO),
                        "--purpose",
                        "on-demand",
                        "--arch",
                        "x86_64",
                        "--suite",
                        suite,
                        "--devices",
                        "spyre/pf=2",
                        "--set",
                        "run.repeats=1",
                        "--out",
                        str(plan),
                        *opts,
                    ]
                ),
                0,
            )
            code = run_cmd.main(
                [
                    "--plan",
                    str(plan),
                    "--catalog",
                    str(DEMO),
                    "--workdir",
                    str(DEMO),
                    "--env-file",
                    str(self.env),
                    "--sys-root",
                    str(self.sysd),
                    "--out",
                    str(out),
                ]
            )
        statuses = [json.loads(p.read_text()) for p in sorted(out.glob("cases/*/repeat-*/status.json"))]
        return code, statuses

    def test_explicit_is_applied_and_verified(self):
        want = pp.format_cpu_list(self.lo)
        code, st = self.run_policy(
            "decode", "--cpu-pinning", "explicit", "--cpu-set", want, "--require-pinning", name="explicit"
        )
        self.assertEqual(code, 0)
        for s in st:
            self.assertEqual(s["argv"][1:3], ["-c", want])
            self.assertEqual(
                (s["pinning"]["run.exec.pinning_verified"], s["pinning"]["run.exec.cpu_set"]), ("true", want)
            )

    def test_missing_tool_runs_unpinned_and_unknown(self):
        want = pp.format_cpu_list(self.lo)
        with mock.patch("spyre_bench.pinning_run.shutil.which", return_value=None):
            _, st = self.run_policy(
                "decode", "--cpu-pinning", "explicit", "--cpu-set", want, "--require-pinning", name="notool"
            )
        self.assertEqual({s["pinning"]["run.exec.pinning_verified"] for s in st}, {"unknown"})
        self.assertEqual({s["pinning"]["run.exec.cpu_set_source"] for s in st}, {"unresolved"})
        self.assertTrue(all(s["argv"][0] != "taskset" for s in st))

    def test_device_local_second_device_node(self):
        _, st = self.run_policy("decode", "--cpu-pinning", "device_local", "--require-pinning", name="local")
        union = pp.format_cpu_list(self.cpus)
        self.assertTrue(all(s["pinning"]["run.exec.cpu_set_source"] == "device_numa" for s in st))
        self.assertEqual({s["pinning"]["run.exec.cpu_set"] for s in st}, {union})

    def test_per_device_ranks_pins_each_rank_and_verifies(self):
        code, st = self.run_policy(
            "tp2",
            "--cpu-pinning",
            "per_device",
            "--pin-opt",
            "cores_per_device=1",
            "--pin-opt",
            "layout=ranks",
            "--require-pinning",
            name="ranks",
        )
        self.assertEqual(code, 0)
        for s in st:
            self.assertEqual(s["pinning"]["run.exec.pinning_verified"], "true", s)
            self.assertEqual(s["pinning"]["run.exec.pinning_violations"], "0")

    def test_per_device_tree_layout_with_spread(self):
        _, st = self.run_policy(
            "decode",
            "--cpu-pinning",
            "per_device",
            "--pin-opt",
            "cores_per_device=2",
            "--pin-opt",
            "spread=spread",
            "--require-pinning",
            name="spread",
        )
        self.assertEqual({s["pinning"]["run.exec.pinning_verified"] for s in st}, {"true"})


class Analyze(unittest.TestCase):
    def test_load_results_reads_files_and_run_directories(self):
        out = Path(tempfile.mkdtemp(dir=TMP))
        (out / "cases/a").mkdir(parents=True)
        (out / "cases/a/result.json").write_text('{"x": 1}')
        one, many = out / "one.json", out / "many.json"
        one.write_text('{"y": 2}')
        many.write_text('[{"z": 3}, {"w": 4}]')
        self.assertEqual(analysis.load_results([out, one, many]), [{"x": 1}, {"y": 2}, {"z": 3}, {"w": 4}])


if __name__ == "__main__":
    unittest.main()
