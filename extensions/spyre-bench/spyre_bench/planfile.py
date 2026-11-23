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

"""plan.json: schema constants, the skew check, and the argv token templates shared by plan and run."""

import hashlib
import importlib.metadata
import json
import os
import secrets
import shlex
import shutil
from pathlib import Path

import regex as re

from spyre_bench import engine

PLAN_SCHEMA = "bench.plan/1"
RUN_SCHEMA = "bench.run/1"
# Bump with any change to how a stored id, series key or env id is derived: run refuses a plan made under another.
IDENTITY_VERSION = 1
RUN_TOKENS = (
    "run_id",
    "subject_id",
    "arch",
    "repeat",
    "repeats",
    "cache",
    "profile_level",
    "raw_dir",
    "repeat_dir",
    "out_dir",
    "python",
    "pin_wrapper",
)
INGEST_TOKENS = (
    "python",
    "run_id",
    "run_key",
    "subject_id",
    "arch",
    "component",
    "raw_dir",
    "out_dir",
    "case",
    "profile",
    "test_type",
)
TOKEN = re.compile(r"\{\{|\}\}|\{([A-Za-z0-9_.-]+)\}")
CASE_KEYS = ("case", "benchmark", "component", "environment")
SECRET_SEGMENTS = frozenset(
    {"token", "password", "passwd", "pass", "pw", "pwd", "passphrase", "secret", "secrets", "credential"}
    | {"credentials", "cred", "creds", "auth", "authorization", "bearer", "cookie", "cookies", "dsn", "apikey"}
    | {"secretkey", "privkey", "privatekey", "accesskey", "authkey"}
)
# Words that name a secret only when they END the name: `--auth-mode` and `--cookie-jar` do not.
TAIL_SEGMENTS = frozenset({"auth", "bearer", "cookie", "cookies", "dsn"})
# Variable names are held to a stricter standard than flags: any of these as a whole segment, `key` included.
ENV_SECRET_SEGMENTS = (SECRET_SEGMENTS - {"pwd"}) | {"key", "pat", "session", "private", "certkey"}  # PWD is the cwd
KEY_PREFIXES = frozenset({"access", "api", "secret", "private", "auth", "ssh", "signing", "encryption", "master"})
USER_SEGMENTS = frozenset({"user", "username", "login"})
URL_CREDENTIALS = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@]*:[^/\s@]*@")
AUTH_HEADER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{6,}")
HEADER_LINE = re.compile(r"(?i)\b(authorization|cookie|x-api-key|proxy-authorization)\s*[:=]\s*\S.*")
JSON_PAIR = re.compile(r"""["']([^"'\\]+)["']\s*:\s*["']((?:[^"'\\]|\\.)*)["']""")
NAME_VALUE = re.compile(r"""(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_.-]*)(=|:[ \t]+)(?:"[^"]*"|'[^']*'|[^\s;&|"']+)""")
# JWTs, vendor-prefixed tokens, and opaque 32+ character alphanumeric runs that mix letters and digits (dashes, dots and slashes
# keep UUIDs, hostnames and paths out; a pure hex run, such as a commit sha or digest, is not a token).
TOKEN_SHAPE = re.compile(
    r"^(?:eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"
    r"|(?:sk-|ghp_|gho_|ghs_|ghu_|github_pat_|hf_|xox[abprs]-|AKIA|glpat-|npm_|AIza|ya29\.|pypi-)[A-Za-z0-9_.-]{8,}"
    r"|[A-Za-z0-9+/]{24,}={1,2}"
    r"|(?![0-9a-fA-F]+$)(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{32,})$"
)
ENV_ALLOW = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TMP", "TEMP", "TERM", "TZ", "VIRTUAL_ENV", "CI"}
    | {"LD_LIBRARY_PATH", "LANG", "LANGUAGE", "PWD", "HOSTNAME", "NODE_NAME"}
    | {"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY"}
    | {"SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"}
    | {"MAKEFLAGS", "MAKELEVEL", "MFLAGS", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CPPFLAGS"}
    | {"LIBRARY_PATH", "CPATH", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
    | {"TRANSFORMERS_CACHE"}
)
ENV_ALLOW_PREFIX = (
    "LC_",
    "PYTHON",
    "SPYRE_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "NUMEXPR_",
    "XDG_",
    "CUDA_",
    "TORCH_",
    "VLLM_",
) + ("KMP_", "GOMP_", "HF_DATASETS_", "CMAKE_", "CCACHE_")
INGEST_ENV_PREFIX = ("CLICKHOUSE_", "INGEST_")


def segments(name):
    return [s for s in re.split(r"[-_.]+", name.strip("-").lower()) if s]


def secret_name(name):
    """True when a flag or word name says secret by a whole segment (`api-key` yes; `run-key`, `ssh-key-path`,
    `max-tokens`, `auth-mode` and `passes` no)."""
    parts = segments(name)
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if part == "key" and last and (len(parts) == 1 or parts[i - 1] in KEY_PREFIXES):
            return True
        if part in SECRET_SEGMENTS and (last or part not in TAIL_SEGMENTS):
            return True
    return False


def secret_env_name(name):
    """True when an environment variable name has a whole segment that says secret (SPYRE_KEYBOARD no, SPYRE_KEY yes)."""
    return any(part in ENV_SECRET_SEGMENTS for part in segments(name))


def credential_looking(value):
    """A value that embeds URL credentials or looks like a token."""
    return bool(URL_CREDENTIALS.search(value) or AUTH_HEADER.search(value) or TOKEN_SHAPE.match(value.strip()))


def env_filter(environ, pass_env=(), pass_prefixes=(), extra_prefixes=()):
    """(env, dropped): the variables a child may inherit, and {name: why} for allowlisted names withheld.

    A name is inherited when it is allowlisted or carries a passed prefix (both case-insensitive), has no secret-looking
    segment, and its value does not look like a credential. A name in `pass_env` is inherited as is.
    """
    out, dropped = {}, {}
    prefixes = tuple(p.upper() for p in ENV_ALLOW_PREFIX + tuple(pass_prefixes) + tuple(extra_prefixes))
    for name, value in environ.items():
        up = name.upper()
        if name in pass_env:
            out[name] = value
        elif up in ENV_ALLOW or up.startswith(prefixes):
            if secret_env_name(name):
                dropped[name] = "secret-looking name"
            elif credential_looking(value):
                dropped[name] = "credential-looking value"
            else:
                out[name] = value
    return out, dict(sorted(dropped.items()))


def benchmark_env(environ, pass_env=(), pass_prefixes=()):
    """The environment a benchmark command inherits (see env_filter); anything else needs an explicit --pass-env NAME."""
    return env_filter(environ, pass_env, pass_prefixes)[0]


def ingest_env(environ, pass_env=(), home=None):
    """The ingest's environment: the benchmark allowlist without PYTHON*, its own credentials, a HOME of our choosing.

    The caller's HOME (user site, dotfiles) and PYTHON* (PYTHONPATH, PYTHONHOME, PYTHONUSERBASE, ...) stay out unless
    named with --ingest-pass-env. A proxy that carries URL credentials is withheld like for a benchmark.
    """
    out = {k: v for k, v in benchmark_env(environ).items() if not k.upper().startswith("PYTHON")}
    for name, value in environ.items():
        if name in pass_env or name.upper().startswith(INGEST_ENV_PREFIX):
            out[name] = value
    if home and "HOME" not in pass_env:
        out["HOME"] = home
    out["PYTHONSAFEPATH"] = "1"  # no cwd or script directory on sys.path, whatever argv the ingest is started with
    return out


def redact_word(word):
    """One word with URL credentials, headers, JSON/dict secrets, NAME=value and `name: value` secrets (quoted or not)
    and bare tokens replaced."""
    word = URL_CREDENTIALS.sub(r"\1<redacted>@", word)
    word = HEADER_LINE.sub(lambda m: f"{m.group(1)}: <redacted>", word)
    word = AUTH_HEADER.sub(lambda m: f"{m.group(1)} <redacted>", word)
    word = JSON_PAIR.sub(
        lambda m: m.group(0).replace(m.group(2), "<redacted>", 1)
        if secret_name(m.group(1)) or credential_looking(m.group(2))
        else m.group(0),
        word,
    )

    def named(m):
        name = m.group(1)
        attached = word[m.start() - 1 : m.start()] == "-" and len(name) > 1  # -DPASSWORD=x, -eHF_TOKEN=x
        hide = secret_name(name) or (attached and secret_name(name[1:]))
        return f"{name}{m.group(2)}<redacted>" if hide else m.group(0)

    word = NAME_VALUE.sub(named, word)
    return "<redacted>" if TOKEN_SHAPE.match(word) else word


def quote_word(word):
    return shlex.quote(word) if any(c.isspace() for c in word) else word


def redact_argv(argv):
    """argv with secrets hidden, for anything written to disk or printed.

    Hidden: the value of a secret-named flag (`--token x`, `--api-key=x`, `-p x`, `-p=x`, `-px`, `-pass x`, `--user a:b`,
    `-u a:b`, `-u x`) or bare `NAME value` pair, and inside every word, including the program text of `sh -c "..."`:
    URL credentials (empty user too), Authorization/Cookie headers, Bearer/Basic values, JSON and dict
    `"password": "x"` pairs, `NAME=value` and `name: value` with a secret name, quoted or not (`--env=HF_TOKEN=x`,
    `-eHF_TOKEN=x`, `export TOKEN='x'`, `Password=x;`), and bare token-shaped words (`ghp_...`, `sk-...`, JWTs, base64
    with padding, long mixed letters-and-digits runs; a pure hex run is not one).
    """
    out, hide, hide_pair, hide_plain = [], False, False, False
    for word in argv:
        word = str(word)
        if hide or hide_pair or hide_plain:
            kind = "plain" if hide_plain else "pair" if hide_pair and not hide else "all"
            hide = hide_pair = hide_plain = False
            if (
                not word.startswith("-")
                and not (kind == "all" and word.startswith("no:"))
                and (  # -p no:plugin
                    kind == "all"
                    or (kind == "pair" and ":" in word)
                    or (kind == "plain" and not ("." in word or "/" in word))
                )
            ):
                out.append("<redacted>")
                continue
        if word.startswith("--"):
            name, eq, value = word[2:].partition("=")
            if secret_name(name):
                out.append(f"--{name}=<redacted>" if eq else word)
                hide = not eq
            elif USER_SEGMENTS & set(segments(name)):
                out.append(f"--{name}=" + ("<redacted>" if ":" in value else value) if eq else word)
                hide_pair = not eq
            else:
                out.append(f"--{name}={redact_word(value)}" if eq else word)
        elif word == "-p":
            out.append(word)
            hide = True
        elif word == "-u":
            out.append(word)
            hide_plain = True
        elif word.startswith("-p="):
            out.append("-p=<redacted>")
        elif word.startswith("-p") and len(word) > 2 and secret_name(word[1:].split("=", 1)[0]):
            name, eq, _ = word[1:].partition("=")
            out.append(f"-{name}=<redacted>" if eq else word)  # -pass x, -password=x
            hide = not eq
        elif word.startswith("-p") and len(word) > 2 and not word.startswith("-p-"):
            out.append("-p<redacted>")
        elif word.startswith("-u") and ":" in word:
            out.append("-u<redacted>")
        elif word.startswith("-"):
            out.append(redact_word(word))
        else:
            name, eq, _ = word.partition("=")
            if not eq and name.isidentifier() and name.isupper() and secret_name(name):
                out.append(word)
                hide = True
            else:
                out.append(redact_inner(redact_word(word)))
    return out


def redact_inner(word):
    """A word holding several (a shell command line, a connection string): redact it as an argv of its own."""
    if not any(c.isspace() for c in word):
        return word
    try:
        inner = shlex.split(word)
    except ValueError:
        return word
    again = [quote_word(w) for w in redact_argv(inner)]
    return " ".join(again) if again != [quote_word(w) for w in inner] else word


def redact_plan(plan):
    """A copy of a plan with the command and ingest argv of every case redacted; for display and for result copies."""
    doc = json.loads(json.dumps(plan))
    for rec in doc.get("cases", []):
        for key in ("command", "ingest"):
            if isinstance(rec.get(key), list):
                rec[key] = redact_argv(rec[key])
    return doc


def case_digest(name):
    """16 hex of the case name, for per-case ingest run keys."""
    return hashlib.sha256(str(name).encode()).hexdigest()[:16]


class SkewError(Exception):
    """The plan was made by an engine whose plan format or id formulas differ from this one."""


def engine_version():
    try:
        return importlib.metadata.version("spyre-bench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def producer():
    return {
        "name": "spyre-bench",
        "version": engine_version(),
        "plan_schema": PLAN_SCHEMA,
        "identity_version": IDENTITY_VERSION,
    }


def check_skew(plan):
    """Raise SkewError unless the plan was made under this engine's plan format and identity version."""
    eng = (plan.get("producer") or {}).get("engine") or {}
    if plan.get("plan_schema") != PLAN_SCHEMA:
        raise SkewError(f"plan_schema {plan.get('plan_schema')!r} != {PLAN_SCHEMA!r}: re-plan with this engine")
    if eng.get("identity_version") != IDENTITY_VERSION:
        raise SkewError(
            f"identity_version {eng.get('identity_version')!r} != {IDENTITY_VERSION!r}: re-plan with this engine"
        )


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def token_value(v):
    """A case value as it appears in an argv: lists join with ',', booleans are true/false, mappings are canonical JSON."""
    if isinstance(v, (list, tuple)):
        return ",".join(token_value(x) for x in v)
    return engine.dstr(v)


def render_case_tokens(template, values, where):
    """Resolve case tokens; leave run tokens and brace escapes for `run`. Returns (argv, errors)."""
    errors, out = [], []

    def sub(m):
        name = m.group(1)
        if name is None or name in RUN_TOKENS:
            return m.group(0)
        if name in values:
            return values[name].replace("{", "{{").replace("}", "}}")
        errors.append(f"{where}: command token {{{name}}} is neither a case value nor a run token {RUN_TOKENS}")
        return m.group(0)

    for part in template:
        out.append(TOKEN.sub(sub, str(part)))
    return out, errors


def render_tokens(template, values, allowed, where):
    """Resolve run-phase tokens and unescape braces. Unknown tokens raise ValueError."""

    def sub(m):
        name = m.group(1)
        if m.group(0) == "{{":
            return "{"
        if m.group(0) == "}}":
            return "}"
        if name not in allowed:
            raise ValueError(f"{where}: unknown token {{{name}}} (allowed: {', '.join(allowed)})")
        return str(values[name])

    return [TOKEN.sub(sub, str(p)) for p in template]


def slug(name):
    safe = re.sub(r"[^a-z0-9._-]", "_", str(name))[:80]
    return f"{safe}-{hashlib.sha256(str(name).encode()).hexdigest()[:8]}"


def dump(doc):
    return json.dumps(doc, indent=1, sort_keys=False, ensure_ascii=False) + "\n"


def ensure_dirs(root, path):
    """Make every directory from `root` down to `path` a real one: a link or file a benchmark planted in its place is
    removed, never followed. (`root` itself is the caller's choice.)"""
    cur = Path(root)
    for part in Path(path).relative_to(root).parts:
        cur = cur / part
        if cur.is_symlink() or (cur.exists() and not cur.is_dir()):
            cur.unlink()
        cur.mkdir(exist_ok=True)


def fresh_dir(root, path):
    """`path` as a new empty real directory under `root`, whatever was there (a link, a file, a directory)."""
    ensure_dirs(root, Path(path).parent)
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir()
    return path


def write_file(root, path, text):
    """Write an engine-owned file under `root` without following a link a benchmark may have planted.

    Parent directories are made real first; the data goes to a new temp name (O_EXCL|O_NOFOLLOW) that then replaces
    `path`, so a link sitting at `path` is replaced rather than written through.
    """
    path = Path(path)
    ensure_dirs(root, path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
