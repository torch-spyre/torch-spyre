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
labels, and the supply-chain channel tag that names it."""

import base64
import json
import os
import urllib.error
import urllib.request
from datetime import date

import regex


OCI_ARCH = {"x86_64": "amd64"}

# The supply-chain channels: the dated ICR tag each pushes, and the v2 family it maps to.
CHANNELS = {
    "snap": regex.compile(r"^snap-(\d{8})(?:T(\d{6})_\d+)?$"),
    "nightly": regex.compile(r"^nightly-(\d{8})$"),
    "weekly": regex.compile(r"^weekly-W(\d{2})$"),
    "ci-cd-tech-preview": regex.compile(r"^ci-cd-tech-preview-v\d+$"),
}
FAMILIES = {
    "snap": "snap-supply-chain",
    "nightly": "nightly-supply-chain",
    "weekly": "weekly-supply-chain",
    "ci-cd-tech-preview": "ci-cd-tech-preview",
}
# Registry manifest reads one resolution may spend searching for a channel tag.
MAX_TAG_READS = 80


def _day(stamp: str) -> date:
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))


def iso_week_tag(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{FAMILIES['weekly']}-{year}-w{week:02d}"


def channel_tag(channel: str, registry_tag: str, built=None) -> str:
    """The v2 tag of a registry channel tag; `built` (a date) gives a weekly tag its year."""
    m = CHANNELS[channel].match(registry_tag)
    if not m:
        return ""
    if channel == "ci-cd-tech-preview":
        return registry_tag
    if channel == "weekly":
        year, built_week, _ = (built or date.today()).isocalendar()
        week = int(m[1])
        # A W01 tag on an image built in late December belongs to the next ISO year.
        year += 1 if week < built_week - 26 else -1 if week > built_week + 26 else 0
        return f"{FAMILIES['weekly']}-{year}-w{week:02d}"
    d = _day(m[1]).isoformat()
    time = m[2] if channel == "snap" else None
    return f"{FAMILIES[channel]}-{d}" + (f"T{time}" if time else "")


def fallback_tag(channel: str, day=None, name: str = "") -> str:
    """The v2 tag a run names when no registry tag names its image: from its own date."""
    if channel == "ci-cd-tech-preview":
        return name
    if channel not in FAMILIES or not day:
        return ""
    return iso_week_tag(day) if channel == "weekly" else f"{FAMILIES[channel]}-{day}"


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
            url = (
                f"https://{self.host}/oauth/token?service=registry"
                f"&scope=repository:{repo}:pull"
            )
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


def _candidates(tags: list, channel: str, day) -> list:
    """`channel`'s registry tags, nearest to `day` (else newest) first; a snap build's own
    timed tag before its day's aggregate."""
    found = [(t, m) for t in tags if (m := CHANNELS[channel].match(t))]
    if channel in ("snap", "nightly") and day:
        found.sort(
            key=lambda x: (abs((_day(x[1][1]) - day).days), x[1].lastindex < 2, x[0])
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


def channel_tag_of(
    registry: Registry,
    path: str,
    leaf: str,
    channel: str = "",
    day=None,
    name: str = "",
) -> tuple:
    """(tag, tag_family, registry_tag) naming `leaf`: the registry's channel tag, else the
    run's own `fallback_tag`; ('', '', '') when neither applies.

    A writer that knows its run's channel and day keeps the tag in that channel, dating it
    itself when the registry has no such tag; one that knows neither takes any channel's.
    """
    reads = 0
    rest = [c for c in CHANNELS if c != channel]
    order = [channel] + ([] if day else rest) if channel in CHANNELS else rest
    for ch in order:
        for t in _candidates(registry.tags(path), ch, day):
            if reads >= MAX_TAG_READS:
                break
            reads += 1
            if registry.names(path, t, leaf):
                built = registry.built(path, leaf) if ch == "weekly" else None
                return channel_tag(ch, t, built), FAMILIES[ch], t
    tag = fallback_tag(channel, day, name)
    return (tag, FAMILIES[channel], "") if tag else ("", "", "")
