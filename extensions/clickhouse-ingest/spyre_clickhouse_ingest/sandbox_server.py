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

"""Self-serve sandbox MCP server: holds the dev admin login so callers never do, and acts on
a sandbox only for the caller whose token owns it.

A caller is identified by a bearer token: either a line in the shared per-user token file
(`<token>=<label>`, the prod MCP sidecar's), or one this server minted at POST /register in
exchange for an OpenShift login. Every sandbox query runs as that sandbox's own ClickHouse
login, so the server's checks are a second fence, not the only one.
"""

import asyncio
import base64
import contextlib
import datetime as dt
import hashlib
import hmac
import io
import json
import os
import ssl
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import clickhouse_connect
import regex as re

from .apply_schema import SchemaApplier
from .sandbox import Sandbox, SeedFilter


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


@dataclass
class Config:
    """Everything the server reads from its environment."""

    secret: bytes = field(default_factory=lambda: env("SANDBOX_SECRET").encode())
    tokens_file: Path = field(
        default_factory=lambda: Path(
            env("SANDBOX_TOKENS_FILE", "/etc/sandbox/tokens/tokens.conf")
        )
    )
    public_url: str = field(
        default_factory=lambda: env("SANDBOX_PUBLIC_URL").rstrip("/")
    )
    # Host and port users' own clients reach the dev server on (get_login), not the in-cluster one.
    public_ch_host: str = field(default_factory=lambda: env("SANDBOX_PUBLIC_CH_HOST"))
    k8s_api: str = field(
        default_factory=lambda: env("SANDBOX_K8S_API", "https://kubernetes.default.svc")
    )
    max_sandboxes: int = field(
        default_factory=lambda: int(env("SANDBOX_MAX_PER_OWNER", "3"))
    )
    max_days: int = field(default_factory=lambda: int(env("SANDBOX_MAX_DAYS", "60")))
    max_runs: int = field(default_factory=lambda: int(env("SANDBOX_MAX_RUNS", "1000")))
    default_ttl: int = field(default_factory=lambda: int(env("SANDBOX_TTL_DAYS", "14")))
    max_ttl: int = field(default_factory=lambda: int(env("SANDBOX_MAX_TTL_DAYS", "30")))
    max_rows: int = 1000


class Identity:
    """Bearer token -> owner, minted tokens, and the per-sandbox ClickHouse password."""

    MINTED = re.compile(r"^sbx_([a-z][a-z0-9_]{1,23})\.([A-Za-z0-9_-]{22})$")

    def __init__(self, secret: bytes, tokens_file: Path):
        if len(secret) < 32:
            raise SystemExit("[error] SANDBOX_SECRET must be at least 32 bytes")
        self.secret = secret
        self.tokens_file = tokens_file

    @staticmethod
    def owner(label: str) -> str:
        """A stable [a-z][a-z0-9_]{1,23} owner from a token label or an OpenShift user name."""
        name = label.strip().removeprefix("IAM#").split()[0].split("@")[0].lower()
        name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        if not name or not name[0].isalpha():
            name = "u" + name
        return name[:24].rstrip("_")

    def _sig(self, purpose: str, value: str, n: int = 16) -> str:
        mac = hmac.new(self.secret, f"{purpose}:{value}".encode(), hashlib.sha256)
        return base64.urlsafe_b64encode(mac.digest()[:n]).decode().rstrip("=")

    def mint(self, owner: str) -> str:
        return f"sbx_{owner}.{self._sig('token', owner)}"

    def password(self, db: str) -> str:
        """Derived, so the server stores no per-sandbox secret and survives restarts."""
        return self._sig("password", db, 24)

    def labels(self) -> dict:
        """token -> label from the shared file, re-read so a Secret update applies live."""
        out = {}
        with contextlib.suppress(FileNotFoundError):
            for line in self.tokens_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    token, label = line.split("=", 1)
                    out[token.strip()] = label.strip()
        return out

    def resolve(self, token: str) -> str | None:
        m = self.MINTED.match(token)
        if m:
            ok = hmac.compare_digest(m.group(2), self._sig("token", m.group(1)))
            return m.group(1) if ok else None
        label = self.labels().get(token)
        return self.owner(label) if label else None


class SchemaSource:
    """Fetches extensions/clickhouse-ingest/schema/ of a torch-spyre ref as a tarball."""

    REPO = re.compile(r"^[A-Za-z0-9-]{1,39}/[A-Za-z0-9._-]{1,100}$")
    REF = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
    SUBDIR = "extensions/clickhouse-ingest/schema/"

    @classmethod
    @contextlib.contextmanager
    def fetch(cls, repo: str, ref: str):
        if not cls.REPO.match(repo) or not cls.REF.match(ref) or ".." in ref:
            raise ValueError(f"bad schema source {repo}@{ref}")
        url = f"https://codeload.github.com/{repo}/tar.gz/{ref}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        with (
            tempfile.TemporaryDirectory() as tmp,
            tarfile.open(fileobj=io.BytesIO(data)) as tar,
        ):
            members = []
            for m in tar.getmembers():
                _, _, rel = m.name.partition("/")
                if rel.startswith(cls.SUBDIR) and (m.isfile() or m.isdir()):
                    m.name = rel[len(cls.SUBDIR) :] or "."
                    members.append(m)
            if not any(m.name.endswith(".sql") for m in members):
                raise ValueError(f"{repo}@{ref} has no {cls.SUBDIR}*.sql")
            tar.extractall(tmp, members=members, filter="data")
            yield Path(tmp)


class Sandboxes:
    """The operations behind every tool, as blocking calls scoped to one owner."""

    MANAGED = "sandbox-server"
    SUFFIX = re.compile(r"^[a-z0-9]{1,12}$")
    READS = re.compile(r"^\s*(SELECT|WITH|SHOW|DESCRIBE|DESC|EXPLAIN|EXISTS)\b", re.I)

    def __init__(self, cfg: Config, identity: Identity, admin_connect, connect):
        self.cfg, self.identity = cfg, identity
        self.admin_connect = admin_connect  # (database) -> admin client
        self.connect = connect  # (user, password, database) -> client

    def name(self, owner: str, suffix: str = "") -> str:
        if suffix and not self.SUFFIX.match(suffix):
            raise ValueError("name_suffix must be 1-12 of [a-z0-9]")
        return f"{owner}_{suffix}" if suffix else owner

    def meta(self, admin) -> dict:
        """db -> comment metadata, for every sandbox this server manages."""
        rows = admin.query(
            "SELECT name, comment FROM system.databases "
            f"WHERE startsWith(name, '{Sandbox.PREFIX}')"
        ).result_rows
        out = {}
        for db, comment in rows:
            with contextlib.suppress(ValueError):
                m = json.loads(comment or "{}")
                if m.get("managed_by") == self.MANAGED:
                    out[db] = m
        return out

    def owned(self, admin, owner: str, suffix: str) -> str:
        name = self.name(owner, suffix)
        m = self.meta(admin).get(Sandbox.database(name))
        if not m or m.get("owner") != owner:
            raise PermissionError(f"you have no sandbox named {Sandbox.database(name)}")
        return name

    def login(self, name: str):
        db = Sandbox.database(name)
        return self.connect(Sandbox.user(name), self.identity.password(db), db)

    def seed_filter(self, f: dict) -> SeedFilter:
        return SeedFilter(
            max(1, min(int(f.get("days", 14)), self.cfg.max_days)),
            max(1, min(int(f.get("runs_per_component", 200)), self.cfg.max_runs)),
            tuple(f.get("components") or ()),
            tuple(f.get("arches") or ()),
            tuple(f.get("tags") or ()),
            tuple(f.get("run_ids") or ()),
        )

    def comment(self, owner: str, ttl_days: int, schema: str, created: str = "") -> str:
        now = dt.datetime.now(dt.UTC)
        ttl = max(1, min(ttl_days, self.cfg.max_ttl))
        return json.dumps(
            {
                "managed_by": self.MANAGED,
                "owner": owner,
                "created": created or now.isoformat(timespec="seconds"),
                "expires": (now + dt.timedelta(days=ttl)).isoformat(timespec="seconds"),
                "schema": schema,
            }
        )

    def create(
        self, owner, suffix, seed, filters, repo, ref, ttl_days, replace
    ) -> dict:
        admin = self.admin_connect("default")
        name = self.name(owner, suffix)
        db = Sandbox.database(name)
        meta = self.meta(admin)
        exists = bool(admin.command(f"EXISTS DATABASE {db}"))
        if exists and meta.get(db, {}).get("owner") != owner:
            raise PermissionError(f"{db} exists and is not yours")
        if exists and not replace:
            raise ValueError(f"{db} exists -- pass replace=true to rebuild it")
        mine = [d for d, m in meta.items() if m.get("owner") == owner and d != db]
        if len(mine) >= self.cfg.max_sandboxes:
            raise ValueError(
                f"limit of {self.cfg.max_sandboxes} sandboxes reached: {', '.join(mine)}"
            )
        with SchemaSource.fetch(repo, ref) as schema_dir:
            steps = Sandbox.build(
                admin,
                self.admin_connect,
                name,
                schema_dir,
                self.comment(owner, ttl_days, f"{repo}@{ref}"),
            )
        Sandbox.grant(admin, name, self.identity.password(db))
        rows = Sandbox.seed(admin, db, self.seed_filter(filters)) if seed else []
        return {
            "database": db,
            "objects_applied": len(steps),
            "schema": f"{repo}@{ref}",
            "rows": dict(rows),
            "expires": self.meta(admin)[db]["expires"],
        }

    def seed(self, owner, suffix, filters) -> dict:
        admin = self.admin_connect("default")
        db = Sandbox.database(self.owned(admin, owner, suffix))
        return {
            "database": db,
            "rows": dict(Sandbox.seed(admin, db, self.seed_filter(filters))),
        }

    def drop(self, owner, suffix) -> dict:
        admin = self.admin_connect("default")
        name = self.owned(admin, owner, suffix)
        Sandbox.drop(admin, name)
        return {"dropped": Sandbox.database(name)}

    def extend(self, owner, suffix, ttl_days) -> dict:
        admin = self.admin_connect("default")
        db = Sandbox.database(self.owned(admin, owner, suffix))
        m = self.meta(admin)[db]
        comment = self.comment(
            owner, ttl_days, m.get("schema", ""), m.get("created", "")
        )
        admin.command(
            f"ALTER DATABASE {db} MODIFY COMMENT %(c)s", parameters={"c": comment}
        )
        return {"database": db, "expires": json.loads(comment)["expires"]}

    def list(self, owner) -> list:
        admin = self.admin_connect("default")
        meta = {d: m for d, m in self.meta(admin).items() if m.get("owner") == owner}
        sizes = dict(
            (db, (n, size))
            for db, n, size in admin.query(
                "SELECT database, sum(total_rows), formatReadableSize(sum(total_bytes)) "
                "FROM system.tables WHERE database IN %(dbs)s GROUP BY database",
                parameters={"dbs": list(meta) or [""]},
            ).result_rows
        )
        return [
            {
                "database": db,
                "rows": int(sizes.get(db, (0, ""))[0] or 0),
                "size": sizes.get(db, (0, "0 B"))[1],
                "expires": m.get("expires"),
                "schema": m.get("schema"),
            }
            for db, m in sorted(meta.items())
        ]

    def rows(self, client, sql: str, max_rows: int) -> dict:
        if not self.READS.match(sql):
            client.command(sql)
            return {"ok": True}
        r = client.query(sql)
        limit = max(1, min(max_rows, self.cfg.max_rows))
        out = [
            [
                v if isinstance(v, int | float | str | bool) or v is None else str(v)
                for v in row
            ]
            for row in r.result_rows[:limit]
        ]
        return {
            "columns": list(r.column_names),
            "rows": out,
            "truncated": len(r.result_rows) > limit,
        }

    def query(self, owner, suffix, sql, max_rows) -> dict:
        admin = self.admin_connect("default")
        return self.rows(self.login(self.owned(admin, owner, suffix)), sql, max_rows)

    def diff(self, owner, suffix, repo, ref) -> dict:
        admin = self.admin_connect("default")
        name = self.owned(admin, owner, suffix)
        db = Sandbox.database(name)
        with SchemaSource.fetch(repo, ref) as schema_dir:
            steps, extra = Sandbox.diff(self.login(name), db, schema_dir)
        label = {"create": "removed", "drift": "changed", "recreate": "view-changed"}
        return {
            "against": f"{repo}@{ref}",
            "differences": [
                {
                    "change": label.get(a, a),
                    "name": n,
                    "detail": d if a == "drift" else "",
                }
                for a, n, d in steps
            ]
            + [{"change": "added", "name": n, "detail": ddl} for n, ddl in extra],
        }

    def ddl(self, owner, suffix) -> dict:
        admin = self.admin_connect("default")
        name = self.owned(admin, owner, suffix)
        db = Sandbox.database(name)
        rows = (
            self.login(name)
            .query(
                "SELECT name, create_table_query FROM system.tables WHERE database = %(db)s "
                "ORDER BY name",
                parameters={"db": db},
            )
            .result_rows
        )
        return {n: q.replace(f"{db}.", "") for n, q in rows if n != Sandbox.RUNS}

    def verify(self, owner, repo, ref, base_repo, base_ref) -> dict:
        """A fresh apply of repo@ref, and an upgrade onto base_ref with real rows, both clean."""
        admin = self.admin_connect("default")
        report = {"schema": f"{repo}@{ref}", "base": f"{base_repo}@{base_ref}"}
        # "__" never occurs in an owner and SUFFIX has no "_", so no user sandbox shares these.
        fresh, up = f"{owner}__verify_fresh", f"{owner}__verify_up"
        comment = self.comment(owner, 1, f"verify {repo}@{ref}")
        try:
            with SchemaSource.fetch(repo, ref) as branch:
                Sandbox.build(admin, self.admin_connect, fresh, branch, comment)
                steps, _ = Sandbox.diff(
                    self.admin_connect(Sandbox.database(fresh)),
                    Sandbox.database(fresh),
                    branch,
                )
                report["fresh_pending"] = [f"{a} {n}" for a, n, _ in steps]
                with SchemaSource.fetch(base_repo, base_ref) as base:
                    Sandbox.build(admin, self.admin_connect, up, base, comment)
                Sandbox.seed(admin, Sandbox.database(up), SeedFilter(3, 5))
                target = self.admin_connect(Sandbox.database(up))
                files = SchemaApplier.selected_files(branch)
                applied = SchemaApplier.apply(
                    target,
                    Sandbox.database(up),
                    files,
                    SchemaApplier.migration_files(branch),
                )
                report["upgrade_applied"] = [f"{a} {n}" for a, n, _ in applied]
                steps, _ = Sandbox.diff(target, Sandbox.database(up), branch)
                report["upgrade_pending"] = [f"{a} {n}" for a, n, _ in steps]
        except (
            Exception
        ) as e:  # reported, not raised: the caller needs the partial report
            report["error"] = f"{type(e).__name__}: {e}"
        finally:
            for n in (fresh, up):
                Sandbox.drop(admin, n)
        report["ok"] = (
            "error" not in report
            and not report.get("fresh_pending")
            and not report.get("upgrade_pending")
        )
        return report

    def reap(self) -> list:
        admin = self.admin_connect("default")
        now = dt.datetime.now(dt.UTC)
        gone = []
        for db, m in self.meta(admin).items():
            with contextlib.suppress(KeyError, ValueError):
                if dt.datetime.fromisoformat(m["expires"]) < now:
                    Sandbox.drop(admin, db.removeprefix(Sandbox.PREFIX))
                    gone.append(db)
        return gone


def openshift_user(api: str, token: str) -> str:
    """The OpenShift user name a bearer token belongs to, as the API server reports it."""
    ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    ctx = ssl.create_default_context(cafile=ca if Path(ca).exists() else None)
    req = urllib.request.Request(
        f"{api}/apis/user.openshift.io/v1/users/~",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
        return json.load(r)["metadata"]["name"]


def build_server(cfg: Config, ops: Sandboxes):
    from mcp.server.auth.middleware.auth_context import get_access_token
    from mcp.server.auth.provider import AccessToken
    from mcp.server.auth.settings import AuthSettings
    from clickhouse_connect.driver.exceptions import DatabaseError
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    class Verifier:
        async def verify_token(self, token: str) -> AccessToken | None:
            owner = ops.identity.resolve(token)
            return (
                AccessToken(token=token, client_id=owner, scopes=[]) if owner else None
            )

    @contextlib.asynccontextmanager
    async def lifespan(_):
        async def reaper():
            while True:
                with contextlib.suppress(Exception):
                    for db in await asyncio.to_thread(ops.reap):
                        print(f"[reap] dropped expired {db}", flush=True)
                await asyncio.sleep(1800)

        task = asyncio.create_task(reaper())
        yield {}
        task.cancel()

    server = MCPServer(
        "clickhouse-sandbox",
        instructions=(
            "Personal spyre_v2 sandboxes on the dev ClickHouse server. Every tool acts only "
            "on the caller's own sandboxes (sandbox_<you> or sandbox_<you>_<suffix>). Schema "
            "changes reach spyre_v2 only by a torch-spyre PR; use schema_diff and "
            "verify_schema to build and prove one."
        ),
        token_verifier=Verifier(),
        auth=AuthSettings(
            issuer_url=cfg.public_url or "http://localhost",
            resource_server_url=f"{cfg.public_url or 'http://localhost'}/mcp",
        ),
        lifespan=lifespan,
    )

    def owner() -> str:
        tok = get_access_token()
        if tok is None:
            raise PermissionError("no caller identity")
        return tok.client_id

    async def run(fn, *a):
        # Anticipated failures reach the caller verbatim; anything else stays a logged crash.
        try:
            return await asyncio.to_thread(fn, *a)
        except (ValueError, PermissionError, DatabaseError) as e:
            raise ToolError(str(e)[:4000]) from e

    @server.tool()
    async def whoami() -> dict:
        """Your sandbox owner name, your sandboxes, and the server's limits."""
        me = owner()
        return {
            "owner": me,
            "default_sandbox": Sandbox.database(me),
            "sandboxes": await run(ops.list, me),
            "limits": {
                "sandboxes": cfg.max_sandboxes,
                "days": cfg.max_days,
                "runs_per_component": cfg.max_runs,
                "ttl_days": cfg.max_ttl,
            },
        }

    @server.tool()
    async def create_sandbox(
        name_suffix: str = "",
        days: int = 14,
        runs_per_component: int = 200,
        components: list[str] | None = None,
        arches: list[str] | None = None,
        tags: list[str] | None = None,
        run_ids: list[str] | None = None,
        schema_repo: str = "torch-spyre/torch-spyre",
        schema_ref: str = "main",
        ttl_days: int = 14,
        replace: bool = False,
        seed: bool = True,
    ) -> dict:
        """Create sandbox_<you>[_<suffix>] from the schema at github.com/<schema_repo>@<schema_ref>
        (push your branch to your fork to use it), seeded with prod runs from the last `days`
        days, `runs_per_component` most recent per component, optionally narrowed to
        components / arches / artifact tags, plus explicit run_ids. Takes ~2 minutes at the
        defaults. Dropped automatically after ttl_days unless extended."""
        filters = dict(
            days=days,
            runs_per_component=runs_per_component,
            components=components,
            arches=arches,
            tags=tags,
            run_ids=run_ids,
        )
        return await run(
            ops.create,
            owner(),
            name_suffix,
            seed,
            filters,
            schema_repo,
            schema_ref,
            ttl_days,
            replace,
        )

    @server.tool()
    async def seed_sandbox(
        name_suffix: str = "",
        days: int = 14,
        runs_per_component: int = 200,
        components: list[str] | None = None,
        arches: list[str] | None = None,
        tags: list[str] | None = None,
        run_ids: list[str] | None = None,
    ) -> dict:
        """Add another prod sample to your sandbox; rows it already has are skipped."""
        filters = dict(
            days=days,
            runs_per_component=runs_per_component,
            components=components,
            arches=arches,
            tags=tags,
            run_ids=run_ids,
        )
        return await run(ops.seed, owner(), name_suffix, filters)

    @server.tool()
    async def list_sandboxes() -> list:
        """Your sandboxes with row counts, size, schema source and expiry."""
        return await run(ops.list, owner())

    @server.tool()
    async def drop_sandbox(name_suffix: str = "") -> dict:
        """Drop one of your sandboxes and its login."""
        return await run(ops.drop, owner(), name_suffix)

    @server.tool()
    async def extend_sandbox(name_suffix: str = "", ttl_days: int = 14) -> dict:
        """Push your sandbox's expiry to ttl_days from now."""
        return await run(ops.extend, owner(), name_suffix, ttl_days)

    @server.tool()
    async def run_query(sql: str, name_suffix: str = "", max_rows: int = 200) -> dict:
        """Run any statement in your sandbox (SELECT, CREATE, ALTER, INSERT, DROP ...), as its
        own login. Read prod from here with remote(prod_v2, table='<t>'), or dev spyre_v2
        directly (read-only)."""
        return await run(ops.query, owner(), name_suffix, sql, max_rows)

    @server.tool()
    async def schema_diff(
        name_suffix: str = "",
        schema_repo: str = "torch-spyre/torch-spyre",
        schema_ref: str = "main",
    ) -> dict:
        """What your sandbox has that the schema at <schema_repo>@<schema_ref> does not: the
        content a PR must carry. Only `migrate` entries left means the branch covers it."""
        return await run(ops.diff, owner(), name_suffix, schema_repo, schema_ref)

    @server.tool()
    async def export_ddl(name_suffix: str = "") -> dict:
        """Every CREATE statement in your sandbox, database-unqualified, to copy into schema/."""
        return await run(ops.ddl, owner(), name_suffix)

    @server.tool()
    async def verify_schema(
        schema_repo: str,
        schema_ref: str,
        base_repo: str = "torch-spyre/torch-spyre",
        base_ref: str = "main",
    ) -> dict:
        """Prove a schema branch before its PR: a fresh apply must leave nothing pending, and
        applying it onto base_ref's schema with real rows must run its migrations cleanly.
        Uses two temporary sandboxes that are dropped afterwards; takes a few minutes."""
        return await run(
            ops.verify, owner(), schema_repo, schema_ref, base_repo, base_ref
        )

    @server.tool()
    async def get_login(name_suffix: str = "") -> dict:
        """Host, user and password for your sandbox, for a client outside this server
        (a local mcp-clickhouse, DBeaver, clickhouse-client)."""
        me = owner()
        admin = await run(ops.admin_connect, "default")
        name = await run(ops.owned, admin, me, name_suffix)
        db = Sandbox.database(name)
        return {
            "host": cfg.public_ch_host,
            "port": 443,
            "secure": True,
            "database": db,
            "user": Sandbox.user(name),
            "password": ops.identity.password(db),
        }

    @server.custom_route("/register", methods=["POST"])
    async def register(request: Request) -> JSONResponse:
        """Exchange an OpenShift bearer token (`oc whoami -t`) for a personal sandbox token."""
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse(
                {"error": "send Authorization: Bearer $(oc whoami -t)"}, 401
            )
        try:
            user = await asyncio.to_thread(
                openshift_user, cfg.k8s_api, auth[7:].strip()
            )
        except Exception:
            return JSONResponse({"error": "OpenShift rejected that token"}, 401)
        me = ops.identity.owner(user)
        return JSONResponse(
            {
                "owner": me,
                "token": ops.identity.mint(me),
                "mcp_url": f"{cfg.public_url}/mcp",
            }
        )

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    return server


def main() -> None:
    import uvicorn
    from mcp.server.transport_security import TransportSecuritySettings

    from .client import ClickHouse, ClickHouseEnv

    cfg = Config()
    identity = Identity(cfg.secret, cfg.tokens_file)

    def connect(user, password, database):
        return clickhouse_connect.get_client(
            host=ClickHouseEnv.host(),
            port=int(ClickHouseEnv.port()),
            user=user,
            password=password,
            database=database,
            secure=ClickHouseEnv.secure(),
        )

    ops = Sandboxes(cfg, identity, lambda db: ClickHouse.connect(database=db), connect)
    host = cfg.public_url.split("://", 1)[-1]
    app = build_server(cfg, ops).streamable_http_app(
        stateless_http=True,
        json_response=True,
        host="0.0.0.0",
        transport_security=TransportSecuritySettings(
            allowed_hosts=[host, "localhost:*", "127.0.0.1:*"]
        ),
    )
    uvicorn.run(app, host="0.0.0.0", port=int(env("SANDBOX_PORT", "8080")))


if __name__ == "__main__":
    main()
