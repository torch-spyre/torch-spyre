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

"""spyre-bench dataset: add a dataset to the store and the catalog in one command, list them, check them.

  spyre-bench dataset add PATH --display-name NAME [--public --license L] [--slug slot-NN] [--store datasets]
                          [--subset NAME=GLOB]... [--role R] [--benchmarks-dir D] [--apply]
  spyre-bench dataset list  [--benchmarks-dir D] [--remote] [--json]
  spyre-bench dataset check [--benchmarks-dir D]

`add` is a dry run unless --apply. The slug is chosen from the store listing, the display name goes only into the
store's dataset.json (never into a repo file) and is never printed for a private dataset. See docs/dataset-add.md.
"""

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import regex as re
import yaml

from spyre_bench import cliutil, engine
from spyre_bench import dataset_meta as dm
from spyre_bench import dataset_store as ds

DATASETS_YAML, ASSETS_YAML, STORES_YAML = "datasets.yaml", "assets-datasets.yaml", "stores.yaml"
MANIFEST = "SHA256SUMS"
SCHEMA_VERSION = 1
MAX_BYTES, MAX_FILES, MAX_DEPTH, MAX_NAME = 20 << 30, 10000, 8, 120
ARCHIVE_MEMBERS = 200000
RESERVED = {dm.FILE.lower(), MANIFEST.lower()}
SUFFIXES = {
    ".jsonl",
    ".json",
    ".csv",
    ".tsv",
    ".parquet",
    ".txt",
    ".arrow",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
}
ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


class AddError(Exception):
    """A refusal: bad input, a collision, or a mismatch. Exit 2."""


@dataclass
class Plan:
    kind: str  # file | directory
    items: list  # ds.Item, sorted by rel
    total: int
    sha256: str  # the dataset.json digest: the file's, or the manifest's for a directory
    fmt: str = ""


def check_display_name(name):
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_NAME or not name.isprintable():
        raise AddError(f"--display-name must be 1-{MAX_NAME} printable characters")
    return name.strip()


def check_slug_arg(slug):
    m = ds.SLOT.fullmatch(slug or "")
    if not m or not 1 <= int(m[1]) < ds.SLOT_LIMIT:
        raise AddError(f"--slug must be slot-NN with NN from 01 to {ds.SLOT_LIMIT - 1}")
    return slug


def safe_member(name):
    p = PurePosixPath(name.replace("\\", "/"))
    return not (p.is_absolute() or ".." in p.parts or (p.parts and p.parts[0].endswith(":")))


def check_archive(path, label):
    """Refuse an archive a later extraction could escape from: absolute or `..` members, links, devices."""
    low = path.name.lower()
    try:
        if low.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                infos = z.infolist()
                if len(infos) > ARCHIVE_MEMBERS:
                    raise AddError(f"{label}: too many archive members")
                for i in infos:
                    mode = i.external_attr >> 16
                    if not safe_member(i.filename) or (mode and stat.S_IFMT(mode) not in (stat.S_IFREG, stat.S_IFDIR)):
                        raise AddError(f"{label}: archive member is unsafe (path escape or link)")
        elif low.endswith(ARCHIVES):
            with tarfile.open(path, "r:*") as t:
                for n, m in enumerate(t, 1):
                    if n > ARCHIVE_MEMBERS:
                        raise AddError(f"{label}: too many archive members")
                    if not safe_member(m.name) or not (m.isreg() or m.isdir()):
                        raise AddError(f"{label}: archive member is unsafe (path escape, link or device)")
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError):
        raise AddError(f"{label}: not a readable archive") from None


def neutral_name(src_name):
    """data + the known suffixes: the source file name may carry the dataset's real name."""
    keep = []
    for sfx in reversed([s.lower() for s in Path(src_name).suffixes][-2:]):
        if sfx not in SUFFIXES:
            break
        keep.insert(0, sfx)
    return "data" + "".join(keep)


def scan(path, max_bytes=MAX_BYTES):
    """Hash a file or a directory tree (deterministic order), refusing symlinks, odd names and oversize input."""
    src = Path(os.path.expanduser(path))
    if src.is_symlink():
        raise AddError("PATH is a symlink; give the real path")
    if src.is_file():
        entries = [(neutral_name(src.name), src)]
        kind = "file"
    elif src.is_dir():
        entries, kind = [], "directory"
        stack = [(src, "", 0)]
        while stack:
            d, prefix, depth = stack.pop()
            if depth > MAX_DEPTH:
                raise AddError(f"directory is nested deeper than {MAX_DEPTH} levels")
            with os.scandir(d) as it:
                children = sorted(it, key=lambda e: e.name)
            for e in children:
                rel = f"{prefix}{e.name}"
                if e.is_symlink():
                    raise AddError(f"refused: a symlink at depth {depth}")
                if not ds.SEGMENT.fullmatch(e.name):
                    raise AddError(f"refused: a name at depth {depth} is outside [A-Za-z0-9._-] or starts with '.'")
                if e.is_dir(follow_symlinks=False):
                    stack.append((Path(e.path), rel + "/", depth + 1))
                elif e.is_file(follow_symlinks=False):
                    entries.append((rel, Path(e.path)))
                else:
                    raise AddError(f"refused: an entry at depth {depth} is not a regular file")
                if len(entries) > MAX_FILES:
                    raise AddError(f"more than {MAX_FILES} files")
        if not entries:
            raise AddError("the directory holds no files")
        if any(r.lower() in RESERVED for r, _ in entries):
            raise AddError(f"refused: a top-level file is named {dm.FILE} or {MANIFEST} (any case)")
    else:
        raise AddError("PATH is not a file or directory")
    items, total = [], 0
    for n, (rel, p) in enumerate(sorted(entries), 1):
        try:
            if total + os.stat(p).st_size > max_bytes:
                raise AddError(f"input exceeds the size cap of {max_bytes} bytes (--max-bytes)")
        except OSError:
            raise AddError(f"cannot read entry #{n}") from None
        if p.name.lower().endswith(ARCHIVES):
            check_archive(p, rel if kind == "directory" else "archive")
        try:
            sha, size = ds.sha256_file(p)
        except (OSError, ds.StoreError) as e:
            raise AddError(f"cannot read entry #{n} ({type(e).__name__})") from None
        total += size
        if total > max_bytes:
            raise AddError(f"input exceeds the size cap of {max_bytes} bytes (--max-bytes)")
        items.append(ds.Item(rel, size, sha, path=p))
    if kind == "file":
        return Plan(kind, items, total, items[0].sha256, items[0].rel.partition(".")[2])
    manifest = "".join(f"{i.sha256}  {i.rel}\n" for i in items).encode()
    msha = hashlib.sha256(manifest).hexdigest()
    items.append(ds.Item(MANIFEST, len(manifest), msha, data=manifest))
    return Plan(kind, items, total, msha)


def created_by(env):
    who = "".join(
        c for c in (env.get("BENCH_CREATED_BY") or env.get("USER") or env.get("LOGNAME") or "") if c.isprintable()
    )
    return who[:64] or "unknown"


def meta_doc(slug, a, plan, env, stamp):
    return (
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "slug": slug,
                "display_name": a.display_name,
                "public": bool(a.public),
                "license": a.license,
                "sha256": plan.sha256,
                "size_bytes": plan.total,
                "files": len([i for i in plan.items if i.path is not None]),
                "created_by": created_by(env),
                "created_at": stamp,
            },
            indent=1,
        )
        + "\n"
    )


# ------------------------------------------------------------------ YAML text
def q(v):
    """A YAML flow scalar: bare when plainly safe, else a JSON string (keeps digests and globs quoted)."""
    plain = re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", v) and isinstance(yaml.safe_load(v), str)
    return v if plain else json.dumps(v)


def store_doc(name):
    var = "".join(c if c.isalnum() else "_" for c in name.upper())
    return (
        f"apiVersion: bench/v1alpha1\nkind: Store\nmetadata: {{name: {name}}}\nspec:\n  type: core/artifactory\n"
        f"  endpoint: {{env: STORE_BASE_URL}}\n  root: {{env: STORE_{var}_PATH}}\n"
        "  auth: {mode: token, refs: [{env: STORE_TOKEN}]}\n  visibility: internal\n"
    )


def asset_doc(slug, store, plan, a, public_store):
    if plan.kind == "file":
        i = plan.items[0]
        integ = f"{{files: [{{path: {i.rel}, sha256: {json.dumps(i.sha256)}, size: {i.size}}}]}}"
    else:
        integ = f"{{manifest: {{path: {MANIFEST}, sha256: {json.dumps(plan.sha256)}}}}}"
    lines = [
        "apiVersion: bench/v1alpha1",
        "kind: Asset",
        f"metadata: {{name: {slug}-files}}",
        "spec:",
        "  type: fileset",
    ]
    if plan.fmt:
        lines.append(f"  format: {plan.fmt}")
    lines += [
        '  version: "1"',
        f"  license: {q(a.license)}",
        f"  visibility: {'public' if a.public and public_store else 'internal'}",
        f"  source: {{store: {store}, slug: {slug}}}",
        f"  integrity: {integ}",
    ]
    return "\n".join(lines) + "\n"


def dataset_doc(slug, a):
    spec = f"asset: {slug}-files, role: {a.role}"
    if a.subsets:
        spec += ", subsets: {" + ", ".join(f"{q(n)}: {{files: [{q(g)}]}}" for n, g in a.subsets.items()) + "}"
    return f"apiVersion: bench/v1alpha1\nkind: Dataset\nmetadata: {{name: {slug}}}\nspec: {{{spec}}}\n"


def parse_subsets(raw):
    out = {}
    for s in raw or []:
        name, eq, glob = s.partition("=")
        if not eq or not re.fullmatch(r"[a-z0-9]([a-z0-9._@-]{0,126}[a-z0-9])?", name) or not glob or "\n" in glob:
            raise AddError("--subset must be NAME=GLOB")
        if not safe_member(glob):
            raise AddError("--subset GLOB must be a relative pattern without '..'")
        out[name] = glob
    return out


def append_docs(path, texts):
    """Append YAML documents, creating the file; temp file then rename."""
    body = path.read_text(encoding="utf-8") if path.exists() else ""
    for t in texts:
        if body.strip():
            body = re.sub(r"(\n---[ \t]*)+\n*\Z", "\n", body)
            body = body.rstrip("\n") + "\n---\n" + t
        else:
            body = t
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".dataset-add-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def pending(cat, wants):
    """[(file, text)] of the documents the catalog lacks; an existing one must equal ours or the add is refused."""
    out = []
    for fname, kind, name, text in wants:
        have = cat.by.get((kind, name))
        if have is None:
            out.append((fname, text))
        elif have.spec != yaml.safe_load(text)["spec"]:
            raise AddError(f"{kind}/{name} already exists with a different definition; edit it by hand")
    return out


def validate_copy(bdir, central, packs, extra):
    """Apply the pending documents to a throwaway copy of the catalog and report its validation errors."""
    tmp = Path(tempfile.mkdtemp(prefix="dataset-add-"))
    try:
        work = tmp / Path(bdir).resolve().name
        shutil.copytree(bdir, work)
        for fname, text in extra:
            append_docs(work / fname, [text])
        return [e.replace(str(work), str(bdir)) for e in engine.Catalog([work], packs, central).errors]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def find_existing(store, taken, plan):
    """The slot (and its metadata) already holding this exact data, if any."""
    for s in sorted(t for t in taken if ds.SLOT.fullmatch(t)):
        text = store.meta_text(s)
        if text:
            try:
                m = dm.parse(text, s)
            except dm.DatasetMetaError:
                continue
            if m.sha256 == plan.sha256:
                return s, m
    return None, None


def same_data(store, slug, plan):
    for i in plan.items:
        same = (
            store.read(slug, i.rel) == i.data
            if i.data is not None
            else store.file_info(slug, i.rel) == (i.sha256, i.size)
        )
        if not same:
            return False
    return True


# ------------------------------------------------------------------ verbs
def open_catalog(a):
    bdir = Path(cliutil.pick(a.benchmarks_dir, "BENCH_DIR", "benchmarks"))
    if not bdir.is_dir():
        raise AddError(f"{bdir} is not a directory (--benchmarks-dir)")
    cat = engine.Catalog([bdir], cliutil.paths(a.pack, "BENCH_PACK"), cliutil.paths(a.central, "BENCH_CENTRAL"))
    if cat.errors:
        raise AddError("the catalog is invalid:\n" + "\n".join(cat.errors))
    return bdir, cat


def repo_slugs(cat):
    out = {cat.dataset_slug(d.name) for d in cat.of("Dataset")}
    out |= {d.spec["source"]["slug"] for d in cat.of("Asset") if (d.spec.get("source") or {}).get("slug")}
    return out


def cmd_add(a, env, urlopen, out):
    a.display_name = check_display_name(a.display_name)
    a.subsets = parse_subsets(a.subset)
    if a.slug:
        check_slug_arg(a.slug)
    if not re.fullmatch(r"[a-z0-9]([a-z0-9._@-]{0,126}[a-z0-9])?", a.store):
        raise AddError("--store must be a Store name (lowercase letters, digits, . _ @ -)")
    if a.public and not a.license:
        raise AddError("--public needs --license")
    a.license = a.license or "internal"
    bdir, cat = open_catalog(a)
    plan = scan(a.path, a.max_bytes)

    def say(line):
        print(line, file=out)

    say(f"source: {plan.kind}, {len([i for i in plan.items if i.path is not None])} file(s), {plan.total} bytes")
    say(f"sha256: {plan.sha256}")

    store_missing = ("Store", a.store) not in cat.by
    store_text = store_doc(a.store) if store_missing else None
    spec = yaml.safe_load(store_text)["spec"] if store_missing else cat.by[("Store", a.store)].spec
    store = ds.store_from_spec(spec, env, urlopen, write=a.apply)

    taken = store.slugs()
    in_repo = repo_slugs(cat)
    found, found_meta = find_existing(store, taken, plan)
    if a.slug:
        text = store.meta_text(a.slug)
        if text is None:
            slug, reuse = a.slug, False
            if a.slug in in_repo:
                say(f"note: {a.slug} is in the catalog but not yet in the store")
        else:
            m = None
            try:
                m = dm.parse(text, a.slug) if text else None
            except dm.DatasetMetaError:
                pass
            if m is None or m.sha256 != plan.sha256:
                raise AddError(f"{a.slug} is taken by different data or an unreadable entry")
            slug, reuse, found_meta = a.slug, True, m
    elif found:
        slug, reuse = found, True
    else:
        slug, reuse = None, False
    if reuse:
        if (found_meta.display_name, found_meta.public, found_meta.license) != (
            a.display_name,
            bool(a.public),
            a.license,
        ):
            raise AddError(f"{slug} already holds this data with different metadata; nothing was changed")
        if not same_data(store, slug, plan):
            raise AddError(f"{slug}: the stored files differ from the source; run dataset check")
    preview = slug or ds.next_slug(taken | in_repo)
    say(f"slug: {preview} ({'already in the store, identical' if reuse else 'new'})")
    if a.public:
        say(f"public: yes ({a.display_name})")
    else:
        say("public: no (the name stays in the store)")

    def wants(s):
        return [
            (ASSETS_YAML, "Asset", f"{s}-files", asset_doc(s, a.store, plan, a, spec.get("visibility") == "public")),
            (DATASETS_YAML, "Dataset", s, dataset_doc(s, a)),
        ]

    def plan_yaml(s):
        docs = pending(cat, wants(s))
        return ([(STORES_YAML, store_text)] if store_missing else []) + docs

    todo = plan_yaml(preview)
    errs = validate_copy(bdir, a_central(a), a_packs(a), todo)
    if errs:
        raise AddError("the result would not validate:\n" + "\n".join(errs))
    say(f"store: {len(plan.items)} item(s) under {preview}/ plus {dm.FILE}")
    if not a.apply:
        for f, _ in todo:
            say(f"would write: {bdir / f}")
    say("preflight validate: PASS")
    if not a.apply:
        say("dry run: nothing was written; pass --apply")
        return 0

    if not reuse:
        stamp = cliutil.utc_stamp()
        held = in_repo - {slug} if a.slug else in_repo
        slug = store.publish(plan.items, lambda s: meta_doc(s, a, plan, env, stamp), slug, taken=held)
        say(f"stored: {slug}")
        if slug != preview:
            todo = plan_yaml(slug)
            errs = validate_copy(bdir, a_central(a), a_packs(a), todo)
            if errs:
                raise AddError(f"{slug} is stored but the result would not validate:\n" + "\n".join(errs))
    by_file = {}
    for f, t in todo:
        by_file.setdefault(f, []).append(t)
    snap = {f: (bdir / f).read_text(encoding="utf-8") if (bdir / f).exists() else None for f in by_file}
    try:
        for f, texts in by_file.items():
            append_docs(bdir / f, texts)
            say(f"wrote: {bdir / f}")
        errs = engine.Catalog([bdir], a_packs(a), a_central(a)).errors
        if errs:
            raise AddError("validate FAILED after the write (files restored):\n" + "\n".join(errs))
    except BaseException:
        for f, old in snap.items():
            if old is None:
                (bdir / f).unlink(missing_ok=True)
            else:
                (bdir / f).write_text(old, encoding="utf-8")
        raise
    say(f"validate: PASS ({bdir})")
    say("Add this to a Benchmark or Workload:")
    say("  inputs:")
    say("    - name: data")
    say(f"      dataset: {slug}")
    if a.subsets:
        say(f"# optional subsets: {', '.join(a.subsets)}")
    return 0


def a_central(a):
    return cliutil.paths(a.central, "BENCH_CENTRAL")


def a_packs(a):
    return cliutil.paths(a.pack, "BENCH_PACK")


def datasets_in(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "dataset" and isinstance(v, str):
                yield v
            else:
                yield from datasets_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from datasets_in(v)


def used_by(cat):
    wl = {d.name: set(datasets_in(d.spec)) for d in cat.of("Workload")}
    out = {}
    for b in cat.of("Benchmark"):
        names = set(datasets_in(b.spec))
        if isinstance(b.spec.get("workload"), str):
            names |= wl.get(b.spec["workload"], set())
        for n in names:
            out.setdefault(n, set()).add(b.name)
    return out


def asset_digest(specs):
    for s in specs:
        integ = s.get("integrity") or {}
        if (integ.get("manifest") or {}).get("sha256"):
            return integ["manifest"]["sha256"]
        files = integ.get("files") or []
        if files:
            return files[0]["sha256"]
    return ""


def asset_size(specs):
    sizes = [f.get("size") for s in specs for f in (s.get("integrity") or {}).get("files") or []]
    return sum(sizes) if sizes and all(isinstance(x, int) for x in sizes) else None


def store_of(cat, name, env, urlopen, cache):
    spec = cat.dataset_store(name)
    if spec is None:
        return None
    key = json.dumps(spec, sort_keys=True)
    if key not in cache:
        cache[key] = ds.store_from_spec(spec, env, urlopen)
    return cache[key]


def cmd_list(a, env, urlopen, out):
    bdir, cat = open_catalog(a)
    users, cache, rows = used_by(cat), {}, []
    for d in sorted(cat.of("Dataset"), key=lambda d: d.name):
        slug, specs = cat.dataset_slug(d.name), cat.dataset_asset_specs(d.name)
        row = {
            "slug": slug,
            "public": None,
            "sha256": asset_digest(specs),
            "size": asset_size(specs),
            "used_by": sorted(users.get(d.name, ())),
        }
        if a.remote:
            store = store_of(cat, d.name, env, urlopen, cache)
            text = store.meta_text(slug) if store else None
            row["in_store"] = bool(text)
            try:
                m = dm.parse(text, slug) if text else None
            except dm.DatasetMetaError:
                m = None
            if m:
                row["public"] = m.public
                row["sha256"] = row["sha256"] or m.sha256
                if row["size"] is None:
                    row["size"] = json.loads(text).get("size_bytes")
        rows.append(row)
    if a.json:
        print(json.dumps(rows, indent=1), file=out)
        return 0
    print(f"{'SLUG':<10} {'PUBLIC':<7} {'SHA256':<13} {'SIZE':>12}  USED-BY", file=out)
    for r in rows:
        pub = "-" if r["public"] is None else ("yes" if r["public"] else "no")
        sha = (r["sha256"] or "-")[:12]
        size = "-" if r["size"] is None else str(r["size"])
        miss = "  [not in store]" if a.remote and not r.get("in_store") else ""
        print(f"{r['slug']:<10} {pub:<7} {sha:<13} {size:>12}  {','.join(r['used_by']) or '-'}{miss}", file=out)
    return 0


def verify_asset(store, spec, slug):
    """Error strings (slug and item numbers only) for one Asset's bytes in the store."""
    src, integ, errs = spec.get("source") or {}, spec.get("integrity") or {}, []
    loc = src.get("slug") or src.get("path")
    if not (src.get("store") and loc):
        return errs
    for n, f in enumerate(integ.get("files") or [], 1):
        info = store.file_info(loc, f["path"])
        if info is None or info[0] != f["sha256"] or ("size" in f and info[1] != f["size"]):
            errs.append(f"{slug}: file #{n} is missing or differs from its declared digest/size")
    man = integ.get("manifest")
    if man:
        raw = store.read(loc, man["path"])
        if raw is None or hashlib.sha256(raw).hexdigest() != man["sha256"]:
            errs.append(f"{slug}: the manifest is missing or differs from its declared digest")
        else:
            for n, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                sha, _, rel = line.partition("  ")
                try:
                    info = store.file_info(loc, rel) if rel else None
                except ds.StoreError:
                    info = None
                if info is None or info[0] != sha:
                    errs.append(f"{slug}: manifest entry #{n} is missing or differs")
    return errs


def cmd_check(a, env, urlopen, out):
    bdir, cat = open_catalog(a)
    cache, errs, checked = {}, [], 0
    for d in sorted(cat.of("Dataset"), key=lambda d: d.name):
        slug, specs = cat.dataset_slug(d.name), cat.dataset_asset_specs(d.name)
        try:
            store = store_of(cat, d.name, env, urlopen, cache)
        except ds.StoreError as e:
            errs.append(f"{slug}: {e}")
            continue
        if store is None:
            continue
        checked += 1
        try:
            text = store.meta_text(slug)
            if not text:
                errs.append(f"{slug}: {dm.FILE} is {'missing' if text is None else 'absent from an existing folder'}")
                continue
            m = dm.parse(text, slug)
            dm.check_assets(m, specs)
            for s in specs:
                errs += verify_asset(store, s, slug)
        except (dm.DatasetMetaError, ds.StoreError) as e:
            errs.append(str(e) if str(e).startswith(slug) else f"{slug}: {e}")
    for e in errs:
        print(f"error: {e}", file=sys.stderr)
    print(f"{'FAIL' if errs else 'PASS'}: {checked} dataset(s) checked", file=out)
    return 2 if errs else 0


def build_parser():
    ap = argparse.ArgumentParser(prog="spyre-bench dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--benchmarks-dir", help="the catalog (BENCH_DIR, default benchmarks)")
        p.add_argument("--central", action="append", default=[], help="shared definitions (BENCH_CENTRAL)")
        p.add_argument("--pack", action="append", default=[], help="extra pack directory (BENCH_PACK)")

    p = sub.add_parser("add", help="stage a file or directory in the store and define its Dataset and Asset")
    p.add_argument("path")
    p.add_argument("--display-name", required=True, help="stored in the store only; printed only when --public")
    p.add_argument("--public", action="store_true", help="the display name may be shown (needs --license)")
    p.add_argument("--license", default="")
    p.add_argument("--slug", help="slot-NN; default the next free slot from the store listing")
    p.add_argument("--store", default="datasets", help="Store name in the catalog (created if missing)")
    p.add_argument("--subset", action="append", default=[], metavar="NAME=GLOB")
    p.add_argument("--role", default="trace", choices=["trace", "eval", "calibration", "shapes", "fixture", "schema"])
    p.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    p.add_argument("--apply", action="store_true", help="write; without it only print the plan")
    common(p)
    p = sub.add_parser("list", help="the catalog's datasets: slug, public flag, digest, size, users")
    p.add_argument("--remote", action="store_true", help="also read each dataset.json from its store")
    p.add_argument("--json", action="store_true")
    common(p)
    p = sub.add_parser("check", help="re-verify digests and metadata against the store and the catalog")
    common(p)
    return ap


def main(argv=None, env=None, urlopen=None, out=None):
    env = os.environ if env is None else env
    urlopen = urlopen or dm._open
    out = out or sys.stdout
    ap = build_parser()
    fns = {"add": cmd_add, "list": cmd_list, "check": cmd_check}

    def run(a):
        try:
            return fns[a.cmd](a, env, urlopen, out)
        except ds.Collision as e:
            raise cliutil.UsageError(str(e)) from None
        except (AddError, dm.DatasetMetaError) as e:
            raise cliutil.UsageError(str(e)) from None
        except ds.StoreError as e:
            print(f"error: {e}", file=sys.stderr)
            return cliutil.EXIT_STORE

    return cliutil.run_main(run, ap, argv)


if __name__ == "__main__":
    sys.exit(main())
