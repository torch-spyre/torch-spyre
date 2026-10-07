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

"""Read-only container registry access for `resolver`: an image's per-arch leaf, its config
labels, and the tag (in a tag_family from tag_families.yaml) that names it."""

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, fields
from datetime import date
from importlib import resources
from string import Formatter

import regex

OCI_ARCH = {"x86_64": "amd64"}
RELEASE = "release"
# Registry manifest reads one resolution may spend searching for a family's tag.
MAX_TAG_READS = 80
TAG_FAMILIES_ENV = "SPYRE_TAG_FAMILIES"
TEMPLATE_FIELDS = frozenset(
    {"family", "date", "time", "build", "iso_year", "iso_week", "registry_tag"}
)
_OPTIONAL = regex.compile(r"\[([^\[\]]*)\]")


def _fields_of(template: str) -> set:
    return {
        f.split(".")[0].split("[")[0] for _, f, _, _ in Formatter().parse(template) if f
    }


def _day(stamp: str) -> date:
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))


@dataclass(frozen=True)
class TagFamily:
    """One tag_families.yaml entry; that file's header documents each key and field."""

    name: str
    registry_tag: object = (
        None  # compiled regex; None for a family with no registry tag
    )
    tag: str = ""
    fallback: str = ""
    dated: bool = False
    nearest: bool = False
    prefix: bool = False

    @staticmethod
    def render(template: str, values: dict) -> str:
        """`template` from `values`: a [...] part drops out when any field in it is empty, the
        whole tag is '' when a field outside one is."""

        def part(text: str) -> str:
            if any(values.get(f) in (None, "") for f in _fields_of(text)):
                return ""
            return text.format(**values)

        return part(_OPTIONAL.sub(lambda m: part(m[1]), template))

    def from_registry(self, registry_tag: str, built=None) -> str:
        """The v2 tag a registry tag stands for in this family; `built` (the image's build
        date) gives a week-only tag its ISO year. '' when it is not this family's tag."""
        m = self.registry_tag.match(registry_tag) if self.registry_tag else None
        if not m:
            return ""
        values: dict = {"family": self.name, "registry_tag": registry_tag}
        values.update({k: v for k, v in m.groupdict().items() if v})
        if "date" in values:
            values["date"] = _day(values["date"])
            iso_year, iso_week, _ = values["date"].isocalendar()
            values.setdefault("iso_year", iso_year)
            values.setdefault("iso_week", iso_week)
        if "iso_week" in m.groupdict() and m["iso_week"]:
            week = values["iso_week"] = int(m["iso_week"])
            if not m.groupdict().get("date"):
                year, built_week, _ = (built or date.today()).isocalendar()
                # A W01 tag on an image built in late December belongs to the next ISO year.
                year += (
                    1 if week < built_week - 26 else -1 if week > built_week + 26 else 0
                )
                values["iso_year"] = year
        return self.render(self.tag, values)

    def from_date(self, tag_date) -> str:
        """The v2 tag a run names itself by when the registry names its image by none."""
        if not (self.fallback and tag_date):
            return ""
        iso_year, iso_week, _ = tag_date.isocalendar()
        values = {
            "family": self.name,
            "date": tag_date,
            "iso_year": iso_year,
            "iso_week": iso_week,
        }
        return self.render(self.fallback, values)

    def needs_build_date(self) -> bool:
        return (
            self.registry_tag is not None and "date" not in self.registry_tag.groupindex
        )


def _unique_keys_loader():
    import yaml  # here, not at module top: identity and derive-artifact-id stay stdlib-only

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
        dup = sorted({str(k) for k in keys if keys.count(k) > 1})
        if dup:
            raise ValueError(f"tag families: duplicate key(s) {dup}")
        return loader.construct_mapping(node, deep=deep)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml, Loader


def load_tag_families(path: str = "") -> dict:
    """name -> TagFamily from `path` (default: the packaged tag_families.yaml), validated;
    raises ValueError naming the entry at fault."""
    if path:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = (
            resources.files(__package__)
            .joinpath("tag_families.yaml")
            .read_text("utf-8")
        )
    yaml, Loader = _unique_keys_loader()
    where = path or "tag_families.yaml"
    try:
        raw = yaml.load(text, Loader=Loader) or {}
    except yaml.YAMLError as err:
        raise ValueError(f"{where}: {err}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected one mapping per tag family")
    known = {f.name for f in fields(TagFamily)} - {"name"}
    out = {}
    for name, entry in raw.items():
        entry = entry or {}
        if not isinstance(entry, dict) or set(entry) - known:
            extra = sorted(set(entry) - known) if isinstance(entry, dict) else entry
            raise ValueError(f"{where}: tag family {name!r}: unknown key(s) {extra}")
        try:
            pattern = (
                regex.compile(entry["registry_tag"])
                if entry.get("registry_tag")
                else None
            )
        except regex.error as err:
            raise ValueError(
                f"{where}: tag family {name!r}: bad registry_tag: {err}"
            ) from None
        for key in ("tag", "fallback"):
            bad = (
                _fields_of(_OPTIONAL.sub(r"\1", entry.get(key) or "")) - TEMPLATE_FIELDS
            )
            if bad:
                raise ValueError(
                    f"{where}: tag family {name!r}: {key} uses unknown field(s) {sorted(bad)}"
                )
        if pattern is not None and not entry.get("tag"):
            raise ValueError(
                f"{where}: tag family {name!r}: a registry_tag needs a tag"
            )
        out[str(name)] = TagFamily(
            name=str(name),
            registry_tag=pattern,
            tag=entry.get("tag") or "",
            fallback=entry.get("fallback") or "",
            dated=bool(entry.get("dated")),
            nearest=bool(entry.get("nearest")),
            prefix=bool(entry.get("prefix")),
        )
    return out


_LOADED: dict = {}


def tag_families() -> dict:
    """The active tag families: $SPYRE_TAG_FAMILIES, else the packaged file; loaded once."""
    path = os.environ.get(TAG_FAMILIES_ENV, "")
    if path not in _LOADED:
        _LOADED[path] = load_tag_families(path)
    return _LOADED[path]


def dated(tag_family: str) -> bool:
    family = tag_families().get(tag_family)
    return bool(family and family.dated)


def dated_tag(tag_family: str, tag_date=None) -> str:
    family = tag_families().get(tag_family)
    return family.from_date(tag_date) if family else ""


def family_tag(tag_family: str, registry_tag: str, built=None) -> str:
    family = tag_families().get(tag_family)
    return family.from_registry(registry_tag, built) if family else ""


def family_of(tag: str) -> str:
    """The family a full v2 tag belongs to by its prefix, longest first; '' for none."""
    names = sorted(
        (f.name for f in tag_families().values() if f.prefix), key=len, reverse=True
    )
    return next((n for n in names if tag.startswith(n + "-")), "")


class Registry:
    """Read-only Docker v2 API client for one registry, one bearer token per repository."""

    ACCEPT = ", ".join(
        (
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        )
    )

    def __init__(self, host: str = "icr.io", username: str = "", password: str = ""):
        self.host, self.username, self.password = host, username, password
        self._tokens: dict = {}
        self._tags: dict = {}
        self._manifests: dict = {}

    @classmethod
    def from_env(cls) -> "Registry":
        return cls(
            username=os.environ.get("ICR_USERNAME", ""),
            password=os.environ.get("ICR_PASSWORD", ""),
        )

    def _get(self, repo: str, path: str, accept: str = ""):
        headers = {"Accept": accept} if accept else {}
        token = self._token(repo)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"https://{self.host}/v2/{repo}/{path}", headers=headers
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.headers.get("Docker-Content-Digest", ""), json.load(r)

    def _token(self, repo: str) -> str:
        if repo not in self._tokens:
            url = f"https://{self.host}/oauth/token?service=registry&scope=repository:{repo}:pull"
            headers = {}
            if self.username:
                cred = f"{self.username}:{self.password}".encode()
                headers["Authorization"] = "Basic " + base64.b64encode(cred).decode()
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as r:
                body = json.load(r)
            self._tokens[repo] = body.get("token") or body.get("access_token") or ""
        return self._tokens[repo]

    def manifest(self, repo: str, ref: str):
        """(digest, manifest) of `ref` (a tag or digest); ('', None) when absent."""
        key = (repo, ref)
        if key not in self._manifests:
            try:
                digest, body = self._get(repo, f"manifests/{ref}", self.ACCEPT)
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    raise
                digest, body = "", None
            self._manifests[key] = (
                digest or (ref if ref.startswith("sha256:") else ""),
                body,
            )
        return self._manifests[key]

    def tags(self, repo: str) -> list:
        if repo not in self._tags:
            self._tags[repo] = self._get(repo, "tags/list")[1].get("tags") or []
        return self._tags[repo]

    def leaf(self, repo: str, digest: str, arch: str) -> tuple:
        """(leaf digest of `arch`, manifest-list digest or '') for an image digest."""
        _, body = self.manifest(repo, digest)
        if body is None:
            return "", ""
        if "manifests" not in body:
            return digest, ""
        want = OCI_ARCH.get(arch, arch)
        for m in body["manifests"]:
            if (m.get("platform") or {}).get("architecture") == want:
                return m["digest"], digest
        return "", digest

    def config(self, repo: str, leaf: str) -> dict:
        """The leaf image's config blob (`created`, `config.Labels`); {} when unreadable."""
        _, body = self.manifest(repo, leaf)
        digest = ((body or {}).get("config") or {}).get("digest")
        if not digest:
            return {}
        if (repo, digest) not in self._manifests:
            try:
                self._manifests[(repo, digest)] = (
                    "",
                    self._get(repo, f"blobs/{digest}")[1],
                )
            except urllib.error.HTTPError:
                self._manifests[(repo, digest)] = ("", {})
        return self._manifests[(repo, digest)][1] or {}

    def labels(self, repo: str, leaf: str) -> dict:
        return (self.config(repo, leaf).get("config") or {}).get("Labels") or {}

    def built(self, repo: str, leaf: str):
        """The UTC date the leaf image was built (its config's `created`), or None."""
        created = self.config(repo, leaf).get("created") or ""
        return date.fromisoformat(created[:10]) if len(created) >= 10 else None

    def names(self, repo: str, tag: str, leaf: str) -> bool:
        """Does `tag` name `leaf`, directly or as one entry of its manifest list?"""
        digest, body = self.manifest(repo, tag)
        return digest == leaf or any(
            e.get("digest") == leaf for e in (body or {}).get("manifests", [])
        )


def _candidates(tags: list, family: TagFamily, tag_date) -> list:
    """`family`'s registry tags: with `nearest`, those nearest `tag_date` first and one with
    more groups before one with fewer; otherwise newest first."""
    found = [(t, m) for t in tags if (m := family.registry_tag.match(t))]
    if family.nearest and tag_date and "date" in family.registry_tag.groupindex:
        found.sort(
            key=lambda x: (
                abs((_day(x[1]["date"]) - tag_date).days),
                -sum(1 for g in x[1].groups() if g),
                x[0],
            )
        )
        return [t for t, _ in found]
    return sorted((t for t, _ in found), reverse=True)


def split_image(image: str) -> tuple:
    """(host, repository path, tag, digest) of `[image:]<host>/<repo>[:tag][@digest]`."""
    ref = image.removeprefix("image:").split(";", 1)[0].strip()
    repo, _, digest = ref.partition("@")
    host, _, path = repo.partition("/")
    path, _, tag = path.partition(":")
    return host, path, tag, digest


def tag_of(registry, path: str, leaf: str, tag_family: str, tag_date=None) -> tuple:
    """(tag, registry_tag) naming `leaf` in `tag_family`: the registry's tag of that family,
    else the run's dated fallback; ('', '') when neither applies. No registry, no search."""
    family = tag_families().get(tag_family)
    if family is None:
        return "", ""
    if registry is not None and family.registry_tag is not None:
        for reads, t in enumerate(_candidates(registry.tags(path), family, tag_date)):
            if reads >= MAX_TAG_READS:
                break
            if registry.names(path, t, leaf):
                built = (
                    registry.built(path, leaf) if family.needs_build_date() else None
                )
                return family.from_registry(t, built), t
    return family.from_date(tag_date), ""
