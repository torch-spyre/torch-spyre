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

"""Read and write access to a dataset store: a directory (core/filesystem) or an Artifactory-style https endpoint.

A dataset lives at `<root>/<slug>/` as data files plus `dataset.json`. `publish` makes it appear complete or not at all
on a directory store (staged, then renamed into place under a lock); on https it uploads data first and `dataset.json`
last. Messages carry slugs and item numbers, never URLs, tokens or file contents.
"""

import hashlib
import http.client
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import regex as re

from spyre_bench import dataset_meta as dm

try:
    import fcntl
except ImportError:  # not POSIX: no cross-process lock, the rename check still refuses a taken slug
    fcntl = None

SLOT = re.compile(r"^slot-([0-9]{2})$")
SLOT_LIMIT = 90
SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
CLAIM = ".claim-"
READ_CAP = 16 * 1024 * 1024
CHUNK = 1 << 20


class StoreError(RuntimeError):
    """The store could not be read or written (unreachable, HTTP error, disk)."""


class Collision(StoreError):
    """The slug is taken or another writer is racing for it. A refusal, not an outage."""


@dataclass
class Item:
    """One file to store: from `path` (re-hashed while copied) or from `data` bytes."""

    rel: str
    size: int
    sha256: str
    path: Path = None
    data: bytes = None


def check_rel(rel):
    parts = str(rel).split("/")
    if not parts or any(not SEGMENT.fullmatch(p) or p in (".", "..") for p in parts):
        raise StoreError("a stored path is not a plain relative path")
    return rel


def next_slug(taken):
    """The slot after the highest taken one below the limit; freed numbers are never reused."""
    used = [int(m[1]) for s in taken if (m := SLOT.match(s)) and int(m[1]) < SLOT_LIMIT]
    n = max(used, default=0) + 1
    if n >= SLOT_LIMIT:
        raise Collision(f"no free slot below slot-{SLOT_LIMIT}")
    return f"slot-{n:02d}"


def open_regular(path):
    """A read-only fd of a regular file, never through a symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise StoreError("a source entry is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def sha256_file(path):
    h, n = hashlib.sha256(), 0
    with os.fdopen(open_regular(path), "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def write_meta_file(path, text):
    """Write then rename, so a reader sees the whole file or none."""
    tmp = os.path.join(os.path.dirname(path), ".dataset.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def copy_verified(item, dst):
    """Copy a source file and prove the bytes are the ones that were hashed."""
    if item.data is not None:
        Path(dst).write_bytes(item.data)
        return
    h, n = hashlib.sha256(), 0
    with os.fdopen(open_regular(item.path), "rb") as src, open(dst, "wb") as out:
        while chunk := src.read(CHUNK):
            h.update(chunk)
            n += len(chunk)
            out.write(chunk)
    if (h.hexdigest(), n) != (item.sha256, item.size):
        raise StoreError("a source file changed while it was being added")


class DirStore:
    kind = "directory"

    def __init__(self, root):
        self.root = Path(root)

    def _need_root(self):
        if not self.root.is_dir():
            raise StoreError("the dataset store path is not an existing directory")

    def slugs(self):
        self._need_root()
        return {p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")}

    def meta_text(self, slug):
        """The dataset.json text; None when the slug is absent, "" when it exists without the file."""
        d = self.root / dm.check_slug(slug)
        if not d.is_dir():
            return None
        if not (d / dm.FILE).is_file():
            return ""
        try:
            return (d / dm.FILE).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise StoreError(f"{slug}/{dm.FILE}: cannot read it ({type(e).__name__})") from None

    def _file(self, loc, rel):
        parts = [*str(loc).split("/"), *str(rel).split("/")]
        for p in parts:
            check_rel(p)
        return self.root.joinpath(*parts)

    def file_info(self, loc, rel):
        p = self._file(loc, rel)
        if p.is_symlink() or not p.is_file():
            return None
        try:
            return sha256_file(p)
        except OSError as e:
            raise StoreError(f"{loc}: cannot read a stored file ({type(e).__name__})") from None

    def read(self, loc, rel):
        p = self._file(loc, rel)
        if p.is_symlink() or not p.is_file():
            return None
        try:
            with os.fdopen(open_regular(p), "rb") as f:
                return f.read(READ_CAP + 1)[:READ_CAP]
        except OSError as e:
            raise StoreError(f"{loc}: cannot read a stored file ({type(e).__name__})") from None

    @contextmanager
    def _lock(self):
        with open(self.root / ".add.lock", "a") as lf:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    def publish(self, items, meta_for, slug=None, taken=()):
        """Stage everything, then under the lock choose the slug, write dataset.json and rename into place."""
        self._need_root()
        stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.root))
        try:
            for it in items:
                dst = stage / it.rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                copy_verified(it, dst)
            with self._lock():
                used = self.slugs() | set(taken)
                if slug is None:
                    slug = next_slug(used)
                elif slug in used:
                    raise Collision(f"{slug} is already taken")
                write_meta_file(stage / dm.FILE, meta_for(slug))
                os.chmod(stage, 0o755)
                try:
                    os.rename(stage, self.root / slug)
                except OSError:
                    raise Collision(f"{slug} is already taken") from None
            return slug
        finally:
            shutil.rmtree(stage, ignore_errors=True)


class HttpStore:
    """An Artifactory-style repository path over https: PUT/GET/DELETE `<base>/<root>/...`, listing via /api/storage."""

    kind = "https"

    def __init__(self, base, root, token="", urlopen=dm._open):
        if not base.lower().startswith("https://"):
            raise StoreError("the dataset store endpoint must be an https:// URL")
        self.base, self.root = base.rstrip("/"), root.strip("/")
        self.token, self.urlopen = token, urlopen

    def _url(self, *parts, api=False):
        tail = "/".join(p for p in (self.root, *parts) if p)
        return f"{self.base}/api/storage/{tail}" if api else f"{self.base}/{tail}"

    def _call(self, method, url, what, data=None, headers=None, ok404=False):
        req = urllib.request.Request(url, data=data, method=method)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with self.urlopen(req, timeout=dm.TIMEOUT_S) as r:
                return r.read(READ_CAP + 1)[:READ_CAP]
        except urllib.error.HTTPError as e:
            if e.code == 404 and ok404:
                return None
            raise StoreError(f"{what}: the store answered HTTP {e.code}") from None
        except (OSError, http.client.HTTPException, ValueError) as e:
            raise StoreError(f"{what}: cannot reach the store ({type(e).__name__})") from None

    def _children(self, slug=None, what="listing"):
        body = self._call("GET", self._url(slug or "", api=True), what, ok404=True)
        if body is None:
            return {}
        try:
            kids = json.loads(body).get("children") or []
            return {k["uri"].lstrip("/"): bool(k.get("folder")) for k in kids}
        except (ValueError, AttributeError, KeyError, TypeError):
            raise StoreError(f"{what}: the store returned an unreadable listing") from None

    def slugs(self):
        return {n for n, folder in self._children().items() if folder and not n.startswith(".")}

    def meta_text(self, slug):
        body = self._call("GET", self._url(dm.check_slug(slug), dm.FILE), f"{slug}/{dm.FILE}", ok404=True)
        if body is None:
            return None if slug not in self.slugs() else ""
        return body.decode("utf-8", "replace")

    def read(self, loc, rel):
        return self._call("GET", self._url(*self._parts(loc, rel)), f"{loc}: read", ok404=True)

    @staticmethod
    def _parts(loc, rel):
        parts = [*str(loc).split("/"), *str(rel).split("/")]
        for p in parts:
            check_rel(p)
        return parts

    def file_info(self, loc, rel):
        body = self._call("GET", self._url(*self._parts(loc, rel), api=True), f"{loc}: file info", ok404=True)
        if body is None:
            return None
        try:
            doc = json.loads(body)
            return doc["checksums"]["sha256"], int(doc["size"])
        except (ValueError, KeyError, TypeError):
            raise StoreError(f"{loc}: the store returned unreadable file info") from None

    def _put(self, slug, item, what):
        url = self._url(slug, *item.rel.split("/"))
        headers = {"X-Checksum-Sha256": item.sha256, "Content-Type": "application/octet-stream"}
        if item.data is not None:
            self._call("PUT", url, what, data=item.data, headers=headers)
            return
        headers["Content-Length"] = str(item.size)
        with os.fdopen(open_regular(item.path), "rb") as f:
            self._call("PUT", url, what, data=f, headers=headers)

    def publish(self, items, meta_for, slug=None, taken=()):
        """Claim the slug with a marker, upload data, re-check for a racing writer, then upload dataset.json last.

        The store has no conditional create, so this narrows the race rather than closing it: of several markers the
        lowest wins, a loser removes only its own marker (never a shared file name), and every uploaded item is read
        back by checksum before the metadata goes up.
        """
        used = self.slugs() | set(taken)
        if slug is None:
            slug = next_slug(used)
        elif slug in used:
            raise Collision(f"{slug} is already taken")
        mark = f"{CLAIM}{uuid.uuid4().hex}"
        mark_item = Item(mark, 0, hashlib.sha256(b"").hexdigest(), data=b"")
        tops = {it.rel.split("/")[0] for it in items}
        self._put(slug, mark_item, f"{slug}: claim")

        def contested(stage):
            kids = self._children(slug, f"{slug}: {stage}")
            marks = {n for n in kids if n.startswith(CLAIM)}
            foreign = set(kids) - marks - tops
            if foreign or min(marks | {mark}) != mark or dm.FILE in kids:
                self._drop(slug, mark)
                raise Collision(f"{slug} is being written by another adder or is not empty")

        contested("claim check")
        for i, it in enumerate(items, 1):
            self._put(slug, it, f"{slug}: upload item {i}")
        contested("recheck")
        text = meta_for(slug)
        self._put(
            slug,
            Item(dm.FILE, len(text.encode()), hashlib.sha256(text.encode()).hexdigest(), data=text.encode()),
            f"{slug}/{dm.FILE}",
        )
        back = self._call("GET", self._url(slug, dm.FILE), f"{slug}/{dm.FILE}")
        if back != text.encode():
            raise StoreError(f"{slug}: {dm.FILE} does not read back as written; run dataset check")
        for i, it in enumerate(items, 1):
            if self.file_info(slug, it.rel) != (it.sha256, it.size):
                raise StoreError(f"{slug}: uploaded item {i} does not read back with its digest; run dataset check")
        self._drop(slug, mark)
        return slug

    def _drop(self, slug, name):
        try:
            self._call("DELETE", self._url(slug, name), f"{slug}: remove claim", ok404=True)
        except StoreError:
            pass


def store_from_spec(spec, env=None, urlopen=dm._open, write=False):
    """The store behind a Store spec: core/filesystem -> DirStore, core/artifactory -> HttpStore."""
    env = os.environ if env is None else env
    kind = spec.get("type")
    try:
        if kind == "core/filesystem":
            return DirStore(dm._ref(spec.get("path"), env, "path"))
        if kind == "core/artifactory":
            token = ""
            for r in (spec.get("auth") or {}).get("refs") or []:
                if isinstance(r, dict) and "env" in r:
                    token = (env.get(r["env"]) or "").strip()
                    if token:
                        break
            if write and not token:
                names = [r["env"] for r in (spec.get("auth") or {}).get("refs") or [] if "env" in r]
                raise StoreError(
                    f"writing needs a store credential: set one of {', '.join(names) or '(none declared)'}"
                )
            root = dm._ref(spec["root"], env, "root") if spec.get("root") else ""
            return HttpStore(dm._ref(spec.get("endpoint"), env, "endpoint"), root, token, urlopen)
    except dm.DatasetMetaError as e:
        raise StoreError(str(e)) from None
    raise StoreError(f"store type {kind!r} holds no datasets (core/filesystem | core/artifactory)")
