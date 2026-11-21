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

"""Randomised write / fail / retry / publish / retract / duplicate / rollback / retire sequences on real chdb, with promote
flipped on retries and two keys; the invariants of tests/test_bench_reference_chdb.py hold after every step."""

import random
import pytest
from test_bench_reference_chdb import (
    Chdb,
    Flaky,
    doc,
    wctx,
    reference,
    KEYS,
    check_invariants,
)


def tagrows(c):
    return c.rows(
        "SELECT tag, toString(artifact_id) a FROM artifact_tags WHERE tag_family='reference'"
    )


def pubset(c):
    return {
        r["artifact_id"]
        for r in c.rows(
            "SELECT artifact_id FROM v_result_uploads WHERE state='published' AND NOT is_round"
        )
    }


def run(seed, steps=25):
    rnd = random.Random(seed)
    c = Flaky()
    clock = [0]

    def tick():
        clock[0] += 1
        return f"2026-11-01 {clock[0] // 3600:02d}:{clock[0] // 60 % 60:02d}:{clock[0] % 60:02d}.000"

    orig = c.insert

    def insert(table, rows, column_names=None, database=None, **kw):
        if (
            table in ("artifact_tags", "artifacts")
            and column_names
            and "audit_timestamp" not in column_names
        ):
            column_names = [*column_names, "audit_timestamp"] + (
                ["ts"] if table == "artifact_tags" else []
            )
            rows = [
                [*r, tick()] + ([tick()[:19]] if table == "artifact_tags" else [])
                for r in rows
            ]
        return orig(table, rows, column_names=column_names, database=database, **kw)

    c.insert = insert
    ups, trail, n = {}, [], [0]
    contents = [doc(10.0 + i, key=KEYS[i % 2]) for i in range(5)]
    try:
        for _ in range(steps):
            op = rnd.choice(
                [
                    "new",
                    "new",
                    "retry",
                    "publish",
                    "publish",
                    "retract",
                    "dup",
                    "rollback",
                    "retire",
                    "rollback_any",
                ]
            )
            trail.append(op)
            if op == "new":
                i = rnd.randrange(5)
                n[0] += 1
                up = f"01a13151-2096-79bf-bd9c-311b8c8b{n[0]:04d}"
                ups[up] = dict(
                    doc=contents[i],
                    done=False,
                    pub=False,
                    ret=False,
                    promote=rnd.random() < 0.6,
                    first=None,
                )
                c.event(contents[i], up, "pending", "amy", tick())
                if rnd.random() < 0.6:
                    c.fail[
                        rnd.choice(
                            [
                                "artifacts",
                                "artifact_refs",
                                "benchmark_runs",
                                "artifact_tags",
                                "artifact_results",
                            ]
                        )
                    ] = 1
                op = "retry"
            if op == "retry":
                todo = [u for u, v in ups.items() if not v["done"] and not v["ret"]]
                if todo:
                    u = rnd.choice(todo)
                    v = ups[u]
                    d = v["doc"]
                    aid = reference.identity(d).artifact_id
                    pub = pubset(c)
                    before = tagrows(c)
                    # newest published tagged version above this one
                    higher = (
                        c.rows(
                            f"SELECT count() n FROM artifact_tags t JOIN artifacts a ON a.artifact_id=t.artifact_id WHERE t.tag='ref-current' AND t.props['ref.key']='{d['key']}' AND toString(t.artifact_id) IN ({','.join(repr(p) for p in pub) or chr(39) * 2}) AND toUInt32OrZero(a.props['ref.version']) > toUInt32OrZero((SELECT any(props['ref.version']) FROM artifacts WHERE artifact_id=toUUID('{aid}')))"
                        )[0]["n"]
                        if True
                        else 0
                    )
                    exists = c.rows(
                        f"SELECT count() n FROM artifacts WHERE artifact_id=toUUID('{aid}')"
                    )[0]["n"]
                    try:
                        out = reference.write(
                            c,
                            Chdb.DB,
                            d,
                            wctx(
                                u,
                                promote=v["promote"]
                                if rnd.random() < 0.8
                                else not v["promote"],
                            ),
                        )
                        v["done"] = out["status"] in (
                            "ingested",
                            "resumed",
                            "duplicate",
                        )
                        after = tagrows(c)
                        new = after[len(before) :] if len(after) >= len(before) else []
                        if exists and higher and any(t["a"] == aid for t in new):
                            raise AssertionError(
                                (
                                    "resumed older write took slot from newer published",
                                    trail,
                                )
                            )
                    except reference.ReferenceError as err:
                        if not err.retry:
                            v["ret"] = True
                    c.fail.clear()
            elif op == "publish":
                todo = [
                    u
                    for u, v in ups.items()
                    if v["done"] and not v["pub"] and not v["ret"]
                ]
                if todo:
                    u = rnd.choice(todo)
                    c.event(ups[u]["doc"], u, "published", "bob", tick())
                    ups[u]["pub"] = True
            elif op == "foreign":
                from test_bench_reference_chdb import claim

                claim(c, rnd.choice(KEYS), rnd.randrange(1, 5))
            elif op == "retract":
                todo = [u for u, v in ups.items() if v["pub"] and not v["ret"]]
                if todo:
                    u = rnd.choice(todo)
                    c.event(ups[u]["doc"], u, "retracted", "amy", tick())
                    ups[u]["ret"] = True
            elif op == "dup":
                done = [v["doc"] for v in ups.values() if v["done"]]
                if done:
                    n[0] += 1
                    up = f"01a13151-2096-79bf-bd9c-311b8c8b{n[0]:04d}"
                    d = rnd.choice(done)
                    c.event(d, up, "pending", "cy", tick())
                    c.event(d, up, "retracted", "cy", tick())
            elif op in ("rollback", "retire", "rollback_any"):
                pool = [
                    v
                    for v in ups.values()
                    if (
                        v["done"] if op == "rollback_any" else v["pub"] and not v["ret"]
                    )
                ]
                if pool:
                    d = rnd.choice(pool)["doc"]
                    reference.set_tag(
                        c,
                        Chdb.DB,
                        reference.TAG_RETIRED
                        if op == "retire"
                        else reference.TAG_CURRENT,
                        reference.identity(d).artifact_id,
                        d["key"],
                        "root",
                    )
            check_invariants(c, list(trail))
            # S3: label/terms on value rows equals the artifact's
            bad = c.rows(
                "SELECT count() n FROM benchmark_runs r JOIN v_artifacts a ON r.props['ref.key']=a.props['ref.key'] AND r.props['ref.version']=a.props['ref.version'] WHERE r.props['run.ref_only']='1' AND r.props['ref.label'] != a.props['ref.label']"
            )[0]["n"]
            assert bad == 0, ("label split", trail)
    finally:
        c.s.close()


@pytest.mark.parametrize("seed", range(40))
def test_random_interleavings_with_promote_flips_two_keys_and_rollbacks(seed):
    run(seed, 20)
