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

"""Dataset metadata: the `<store>/<slug>/dataset.json` file that sits beside a dataset's data.

Definitions name a dataset by an opaque slug; the store holds its real name and whether that name may be shown.
Tools label a dataset with `display_name` only when the file says `public: true`, else with the slug. The file is a
trusted input: whoever can write the store path decides what is public. A missing, unreadable or malformed file is an
error, never a fallback. Offline `validate` never calls this module.
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import regex as re

FILE = "dataset.json"
SLUG = re.compile(r"^[a-z0-9]([a-z0-9._@-]{0,126}[a-z0-9])?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TIMEOUT_S = 30


class DatasetMetaError(RuntimeError):
    """The metadata file is missing, unreadable, malformed or contradicts the asset. Messages carry the slug only."""


@dataclass(frozen=True)
class DatasetMeta:
    slug: str
    display_name: str
    public: bool
    license: str = ""
    sha256: str = ""
    notes: str = ""

    def label(self):
        """What a tool may show: the real name for a public dataset, else the slug."""
        return self.display_name if self.public else self.slug


def check_slug(slug):
    if not isinstance(slug, str) or not SLUG.match(slug):
        raise DatasetMetaError(f"dataset slug {slug!r} is not a valid slug")
    return slug


def parse(text, slug):
    """A DatasetMeta from the file text. Errors name the slug and the field, never the content."""
    check_slug(slug)
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise DatasetMetaError(f"{slug}/{FILE}: not valid JSON ({type(e).__name__})") from None
    if not isinstance(doc, dict):
        raise DatasetMetaError(f"{slug}/{FILE}: must be a JSON object")
    if "slug" in doc and doc["slug"] != slug:
        raise DatasetMetaError(f"{slug}/{FILE}: its slug field does not match the directory")
    name = doc.get("display_name")
    if not isinstance(name, str) or not name.strip():
        raise DatasetMetaError(f"{slug}/{FILE}: display_name must be a non-empty string")
    if not isinstance(doc.get("public"), bool):
        raise DatasetMetaError(f"{slug}/{FILE}: public must be true or false")
    for k in ("license", "sha256", "notes"):
        if k in doc and not isinstance(doc[k], str):
            raise DatasetMetaError(f"{slug}/{FILE}: {k} must be a string")
    sha = doc.get("sha256", "")
    if sha and not SHA256.match(sha):
        raise DatasetMetaError(f"{slug}/{FILE}: sha256 must be 64 lowercase hex characters")
    return DatasetMeta(slug, name.strip(), doc["public"], doc.get("license", ""), sha, doc.get("notes", ""))


class DirSource:
    """Metadata under a local directory (a core/filesystem store, a mounted cache or a checkout)."""

    def __init__(self, root):
        self.root = Path(root)

    def fetch(self, slug):
        path = self.root / check_slug(slug) / FILE
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise DatasetMetaError(f"{slug}/{FILE}: not found in the dataset store") from None
        except (OSError, UnicodeDecodeError) as e:
            raise DatasetMetaError(f"{slug}/{FILE}: cannot read it ({type(e).__name__})") from None


class HttpSource:
    """Metadata over HTTP from an Artifactory-style store: GET <base>/<root>/<slug>/dataset.json with a bearer token."""

    def __init__(self, base, root, token="", urlopen=urllib.request.urlopen):
        self.url = f"{base.rstrip('/')}/{root.strip('/')}" if root else base.rstrip("/")
        self.token = token
        self.urlopen = urlopen

    def fetch(self, slug):
        req = urllib.request.Request(f"{self.url}/{check_slug(slug)}/{FILE}")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with self.urlopen(req, timeout=TIMEOUT_S) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            what = "not found in the dataset store" if e.code == 404 else f"the store answered HTTP {e.code}"
            raise DatasetMetaError(f"{slug}/{FILE}: {what}") from None
        except (OSError, UnicodeDecodeError) as e:
            raise DatasetMetaError(f"{slug}/{FILE}: cannot fetch it ({type(e).__name__})") from None


def _ref(value, env, what):
    """A Store field written as {env: NAME} or {value: X}."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "value" in value:
        return str(value["value"])
    if isinstance(value, dict) and "env" in value:
        got = (env.get(value["env"]) or "").strip()
        if not got:
            raise DatasetMetaError(f"store {what} comes from {value['env']}, which is unset")
        return got
    raise DatasetMetaError(f"store {what} is not set")


def source_for_store(store_spec, env=None, urlopen=urllib.request.urlopen):
    """The metadata source of a Store spec: core/filesystem -> DirSource, core/artifactory -> HttpSource."""
    env = os.environ if env is None else env
    kind = store_spec.get("type")
    if kind == "core/filesystem":
        return DirSource(_ref(store_spec.get("path"), env, "path"))
    if kind == "core/artifactory":
        token = ""
        for r in (store_spec.get("auth") or {}).get("refs") or []:
            if isinstance(r, dict) and "env" in r:
                token = (env.get(r["env"]) or "").strip()
                if token:
                    break
        root = _ref(store_spec["root"], env, "root") if store_spec.get("root") else ""
        return HttpSource(_ref(store_spec.get("endpoint"), env, "endpoint"), root, token, urlopen)
    raise DatasetMetaError(f"store type {kind!r} holds no dataset metadata (core/filesystem | core/artifactory)")


def read(source, slug):
    return parse(source.fetch(slug), slug)


def check_assets(meta, asset_specs):
    """The metadata digest, when set, must be one the dataset's Assets declare (a manifest or a file digest)."""
    if not meta.sha256:
        return
    known = set()
    for spec in asset_specs:
        integ = spec.get("integrity") or {}
        known |= {f.get("sha256") for f in integ.get("files") or []} | {(integ.get("manifest") or {}).get("sha256")}
    known.discard(None)
    if known and meta.sha256 not in known:
        raise DatasetMetaError(f"{meta.slug}/{FILE}: sha256 matches none of the digests the Asset declares")


class Labels:
    """Slug -> label through a source, caching each read. Any read failure propagates as DatasetMetaError."""

    def __init__(self, source):
        self.source = source
        self.cache = {}

    def meta(self, slug):
        if slug not in self.cache:
            self.cache[slug] = read(self.source, slug)
        return self.cache[slug]

    def label(self, slug):
        return self.meta(slug).label()
