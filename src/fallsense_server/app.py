"""FastAPI app: rooms -> incidents -> phone alerts, and the wall / phone web app.

REST (Contract 7) for carer actions, one WebSocket (Contract 6) for everything
live. No endpoint returns camera images: rooms only ever hold skeletons.

Access: this laptop (the nurse station) is trusted. Anyone else, on the hotspot or
through the Cloudflare tunnel, signs in with the shift PIN and then sends their
carer name and link token with every request, including reads and the WebSocket.
"""

import asyncio
import contextlib
import csv
import io
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.requests import HTTPConnection

from fallsense_server.clips import BONES, FPS, load_thresholds, on_floor_seconds
from fallsense_server.hub import Client, Hub
from fallsense_server.incidents import Incident, IncidentService, Push, Store
from fallsense_server.notifier import Notifier, Tokens
from fallsense_server.rooms import ReplayRoom
from fallsense_server.settings import FRONTEND_DIST, STATE_DIR, Settings, load_settings
from fallsense_server.tunnel import Tunnel, find_cloudflared

log = logging.getLogger("fallsense.server")
SCHEMA_VERSION = 1
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
# cloudflared forwards from localhost but adds these, so tunnel traffic is not local
FORWARDED = ("cf-connecting-ip", "cf-ray", "x-forwarded-for")
LOGIN_TRIES, LOGIN_WINDOW_S = 5, 600  # wrong PINs allowed per address per window


class Who(BaseModel):
    carer: str | None = None
    token: str | None = None


class Resolve(Who):
    outcome: str
    injury: str | None = None
    reason: str | None = None
    note: str | None = None


class Toggle(Who):
    on: bool


class Login(BaseModel):
    carer: str
    pin: str


class Server:
    """Everything the app holds: rooms, incidents, notifier, browser hub."""

    def __init__(self, settings: Settings, state_dir: Path = STATE_DIR) -> None:
        state_dir.mkdir(exist_ok=True)
        self.state_dir = state_dir
        self.settings = settings
        self.run = str(settings.raw["model_run"])
        self.thresholds = load_thresholds(self.run)
        self.hub = Hub()
        self.tokens = Tokens(state_dir / "secret.key")
        self.notifier = Notifier(settings, self.tokens)
        self.service = IncidentService(settings, Store(state_dir / "incidents.db"))
        replay_s = float(settings.escalation["replay_s"])
        self.rooms = {
            r["id"]: ReplayRoom(r, self.run, self.on_room_msg, replay_s)
            for r in settings.rooms
        }
        self.tasks: list[asyncio.Task[None]] = []
        self.tunnel: Tunnel | None = None
        self.failed_logins: defaultdict[str, deque[float]] = defaultdict(deque)

    # ------------------------------------------------------------ room input

    def on_room_msg(self, msg: dict[str, Any]) -> None:
        if msg["type"] == "scores":
            self.service.on_posture(msg["room"], msg["posture"])
        elif msg["type"] == "alert":
            self._log_alert(msg)
            had = self.service.active(msg["room"])
            inc = self.service.on_alert(
                msg["room"], msg["level"], msg["score"], msg["reason"]
            )
            store = self.service.store
            if inc is not None and had is None and store is not None:
                store.save_replay(inc.id, self.rooms[msg["room"]].snapshot())
            msg = {**msg, "incident": inc.id if inc else None}
        self.hub.publish(msg)
        self.flush()

    def _log_alert(self, msg: dict[str, Any]) -> None:
        """alerts.jsonl: the backend's one-JSON-line-per-alert interface, unchanged."""
        line = {
            "room": msg["room"],
            "time": datetime.fromtimestamp(msg["t"], UTC).isoformat(),
            "level": msg["level"],
            "score": msg["score"],
            "source": "tracker" if msg["reason"] == "hidden_down" else "pose",
            "model": self.run,
        }
        with open(self.state_dir / "alerts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def flush(self) -> None:
        """Publish what the incident service changed and send its pushes."""
        svc = self.service
        for inc in {i.id: i for i in svc.changed}.values():
            self.hub.publish({"type": "incident", "incident": inc.to_dict()})
        for ev in svc.new_events:
            self.hub.publish({"type": "event", **ev})
        pushes = list(svc.pushes)
        svc.changed.clear()
        svc.new_events.clear()
        svc.pushes.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # called outside the server (tests): nothing to send with
            return
        for p in pushes:
            self.tasks.append(loop.create_task(self._send(p)))

    async def _send(self, push: Push) -> None:
        results = await self.notifier.send(push)
        failed = [f"{c} ({r})" for c, r in results if r.startswith("failed")]
        if failed:
            self.service._event("push", "", f"Push failed: {', '.join(failed)}")
            self.flush()

    # ------------------------------------------------------------ background

    async def start(self) -> None:
        for room in self.rooms.values():
            self.tasks.append(asyncio.create_task(room.run()))
        self.tasks.append(asyncio.create_task(self._ticker()))
        if self.settings.raw.get("tunnel"):
            self._start_tunnel()

    async def stop(self) -> None:
        for t in self.tasks:
            t.cancel()
        if self.tunnel is not None:
            self.tunnel.stop()
        await self.notifier.close()

    def _start_tunnel(self) -> None:
        exe = find_cloudflared(self.settings.raw.get("cloudflared"))
        if exe is None:
            log.warning("tunnel: true but cloudflared is not installed; using the LAN")
            return
        loop = asyncio.get_running_loop()

        def ready(url: str) -> None:
            self.settings.tunnel_url = url
            self.service._event(
                "info", "", f"Phones can connect from any network: {url}/phone"
            )
            self.flush()

        def gone() -> None:
            self.settings.tunnel_url = None
            self.service._event(
                "info", "", "Tunnel stopped: phones need the laptop's network"
            )
            self.flush()

        self.tunnel = Tunnel(
            exe,
            int(self.settings.raw["port"]),
            on_url=lambda u: loop.call_soon_threadsafe(ready, u),
            on_exit=lambda: loop.call_soon_threadsafe(gone),
        )
        self.tunnel.start()

    async def _ticker(self) -> None:
        offline_after = float(self.settings.escalation["offline_after_s"])
        while True:
            await asyncio.sleep(1.0)
            now = self.service.clock()
            status = self.status(now)
            for s in status["rooms"]:
                if s["online"]:
                    self.service.on_online(s["room"])
                elif now - self.rooms[s["room"]].last_frame_at >= offline_after:
                    self.service.on_offline(s["room"])
            self.service.tick()
            self.hub.publish(status)
            self.tasks = [t for t in self.tasks if not t.done()]
            self.flush()

    # ------------------------------------------------------------ views

    def status(self, now: float) -> dict[str, Any]:
        off = float(self.settings.escalation["offline_after_s"])
        rooms = []
        for rid, room in self.rooms.items():
            s = room.status(now, off)
            s["carer_in_room_until"] = self.service.carer_in_room.get(rid)
            rooms.append(s)
        return {"type": "status", "now": now, "rooms": rooms}

    def hello(self) -> dict[str, Any]:
        s, th = self.settings, self.thresholds
        incs = sorted(self.service.incidents.values(), key=lambda i: i.opened_at)
        recent = [i for i in incs if i.active] + [i for i in incs if not i.active][-30:]
        return {
            "type": "hello",
            "schema_version": SCHEMA_VERSION,
            "site": s.raw["site"],
            "model": {
                "run": self.run,
                "t_high": th.get("t_high"),
                "t_low": th.get("t_low"),
                "on_floor_s": on_floor_seconds(th),
            },
            "fps": FPS,
            "bones": BONES,
            "rooms": [
                {
                    k: r.get(k)
                    for k in ("id", "camera", "name", "resident", "carers", "source")
                }
                | {"scenarios": sorted((r.get("scenarios") or {}).keys())}
                for r in s.rooms
            ],
            "carers": [
                {"name": c.name, "tier": c.tier, "role": c.role, "push": bool(c.topic)}
                for c in s.carers
            ],
            "incidents": [i.to_dict() for i in recent],
            "events": list(self.service.events)[-80:],
            "status": self.status(self.service.clock()),
            "public_url": s.public_url(),
        }

    # ------------------------------------------------------------ auth

    def can_view(self, conn: HTTPConnection) -> bool:
        """Reads: this laptop, or a signed-in carer (headers, or query on the WebSocket)."""
        if self.is_local(conn):
            return True
        carer = conn.headers.get("x-carer") or conn.query_params.get("carer") or ""
        token = conn.headers.get("x-token") or conn.query_params.get("token") or ""
        return bool(self.settings.carer(carer)) and self.tokens.check(carer, token)

    def viewer(self, conn: HTTPConnection) -> None:
        if not self.can_view(conn):
            raise HTTPException(401, "sign in on the phone page first")

    def login_allowed(self, conn: HTTPConnection) -> bool:
        tries = self.failed_logins[client_address(conn)]
        while tries and tries[0] < time.time() - LOGIN_WINDOW_S:
            tries.popleft()
        return len(tries) < LOGIN_TRIES

    def actor(self, request: Request, who: Who) -> str:
        """The carer acting: a valid link token, or this laptop (the nurse station)."""
        known = who.carer and self.settings.carer(who.carer)
        if known and who.token and self.tokens.check(str(who.carer), who.token):
            return str(who.carer)
        if self.is_local(request) and self.settings.raw.get("trust_localhost", True):
            return str(who.carer) if known else "Nurse station"
        raise HTTPException(401, "sign in on the phone page first")

    @staticmethod
    def is_local(conn: HTTPConnection) -> bool:
        """This laptop itself, and not a request relayed by the tunnel."""
        if conn.client is None or conn.client.host not in LOCAL_HOSTS:
            return False
        return not any(h in conn.headers for h in FORWARDED)


def client_address(conn: HTTPConnection) -> str:
    return conn.headers.get("cf-connecting-ip") or (
        conn.client.host if conn.client else "?"
    )


def create_app(
    settings: Settings | None = None, start: bool = True, state_dir: Path = STATE_DIR
) -> FastAPI:
    srv = Server(settings or load_settings(), state_dir)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start:
            await srv.start()
        yield
        await srv.stop()

    app = FastAPI(title="fallsense", lifespan=lifespan)
    app.state.server = srv

    def incident(inc_id: str) -> Incident:
        try:
            return srv.service.get(inc_id)
        except KeyError:
            raise HTTPException(404, f"no incident {inc_id}") from None

    def done(inc: Incident) -> dict[str, Any]:
        srv.flush()
        return inc.to_dict()

    # ------------------------------------------------------------ state

    @app.get("/api/public")
    async def public() -> dict[str, Any]:
        """What the sign-in screen needs before anyone has signed in."""
        s = srv.settings
        return {
            "site": s.raw["site"],
            "carers": [{"name": c.name, "role": c.role} for c in s.carers],
        }

    @app.get("/api/state")
    async def state(request: Request) -> dict[str, Any]:
        srv.viewer(request)
        return srv.hello()

    @app.get("/api/incidents")
    async def incidents(
        request: Request, state: str | None = None
    ) -> list[dict[str, Any]]:
        srv.viewer(request)
        items = sorted(srv.service.incidents.values(), key=lambda i: -i.opened_at)
        return [i.to_dict() for i in items if state is None or i.state == state]

    @app.get("/api/incidents/{inc_id}")
    async def get_incident(inc_id: str, request: Request) -> dict[str, Any]:
        srv.viewer(request)
        return incident(inc_id).to_dict()

    @app.get("/api/incidents/{inc_id}/replay")
    async def replay(inc_id: str, request: Request) -> dict[str, Any]:
        srv.viewer(request)
        inc = incident(inc_id)
        frames = srv.service.store.replay(inc.id) if srv.service.store else []
        return {"incident": inc.id, "fps": FPS, "frames": frames}

    # ------------------------------------------------------------ carer actions

    @app.post("/api/login")
    async def login(body: Login, request: Request) -> dict[str, str]:
        if not srv.login_allowed(request):
            raise HTTPException(429, "too many wrong PINs; wait 10 minutes")
        if not srv.settings.carer(body.carer) or body.pin != str(
            srv.settings.raw["shift_pin"]
        ):
            srv.failed_logins[client_address(request)].append(time.time())
            raise HTTPException(401, "wrong name or PIN")
        return {"carer": body.carer, "token": srv.tokens.token(body.carer)}

    @app.post("/api/incidents/{inc_id}/ack")
    async def ack(inc_id: str, body: Who, request: Request) -> dict[str, Any]:
        return done(srv.service.ack(incident(inc_id).id, srv.actor(request, body)))

    @app.post("/api/push/{inc_id}/ack")
    async def push_ack(
        inc_id: str, carer: str, token: str, request: Request
    ) -> dict[str, Any]:
        """Target of the notification's "I'm going" button (carer and token in the URL)."""
        who = srv.actor(request, Who(carer=carer, token=token))
        return done(srv.service.ack(incident(inc_id).id, who))

    @app.post("/api/incidents/{inc_id}/attend")
    async def attend(inc_id: str, body: Who, request: Request) -> dict[str, Any]:
        return done(srv.service.attend(incident(inc_id).id, srv.actor(request, body)))

    @app.post("/api/incidents/{inc_id}/escalate")
    async def escalate(inc_id: str, body: Who, request: Request) -> dict[str, Any]:
        who = srv.actor(request, body)
        return done(srv.service.escalate_now(incident(inc_id).id, who))

    @app.post("/api/incidents/{inc_id}/dismiss")
    async def dismiss(inc_id: str, body: Who, request: Request) -> dict[str, Any]:
        who = srv.actor(request, body)
        try:
            return done(srv.service.dismiss(incident(inc_id).id, who))
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/incidents/{inc_id}/resolve")
    async def resolve(inc_id: str, body: Resolve, request: Request) -> dict[str, Any]:
        who = srv.actor(request, body)
        try:
            inc = srv.service.resolve(
                incident(inc_id).id,
                who,
                body.outcome,
                body.injury,
                body.reason,
                body.note,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return done(inc)

    @app.post("/api/rooms/{room}/carer_in_room")
    async def carer_in_room(
        room: str, body: Toggle, request: Request
    ) -> dict[str, Any]:
        if room not in srv.rooms:
            raise HTTPException(404, f"no room {room}")
        srv.service.set_carer_in_room(room, body.on, srv.actor(request, body))
        srv.flush()
        return {"room": room, "until": srv.service.carer_in_room.get(room)}

    @app.post("/api/carers/{name}/test")
    async def test_push(name: str, body: Who, request: Request) -> dict[str, Any]:
        srv.actor(request, body)
        if not srv.settings.carer(name):
            raise HTTPException(404, f"no carer {name}")
        push = Push(
            [name],
            4,
            "Test alert",
            "fallsense alerts are reaching this phone",
            None,
            False,
        )
        return {"results": await srv.notifier.send(push)}

    @app.get("/api/export/incidents.csv", response_class=PlainTextResponse)
    async def export(request: Request) -> str:
        srv.viewer(request)
        cols = [
            "id", "room", "kind", "level", "opened_at", "acknowledged_by",
            "acknowledged_at", "attending_at", "resolved_by", "resolved_at",
            "outcome", "injury", "false_alarm_reason", "on_floor_since",
            "floor_ended_at", "tier", "note",
        ]  # fmt: skip
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for inc in sorted(srv.service.incidents.values(), key=lambda i: i.opened_at):
            w.writerow(inc.to_dict())
        return buf.getvalue()

    # ------------------------------------------------------------ presenter

    def presenter_only(request: Request) -> None:
        if not srv.is_local(request):
            raise HTTPException(403, "presenter controls only work on the demo laptop")

    @app.post("/api/demo/rooms/{room}/scenario/{name}")
    async def scenario(room: str, name: str, request: Request) -> dict[str, str]:
        presenter_only(request)
        r = srv.rooms.get(room)
        if r is None or name not in r.scenarios:
            raise HTTPException(404, f"no scenario {name} in {room}")
        r.play_scenario(name)
        return {"room": room, "scenario": name}

    @app.post("/api/demo/rooms/{room}/offline")
    async def offline(room: str, body: Toggle, request: Request) -> dict[str, Any]:
        presenter_only(request)
        if room not in srv.rooms:
            raise HTTPException(404, f"no room {room}")
        srv.rooms[room].set_offline(body.on)
        return {"room": room, "offline": body.on}

    @app.post("/api/demo/reset")
    async def reset(request: Request) -> dict[str, str]:
        presenter_only(request)
        srv.service.reset()
        for r in srv.rooms.values():
            r.set_offline(False)
        srv.hub.publish(srv.hello())
        return {"reset": "ok"}

    # ------------------------------------------------------------ live

    @app.websocket("/ws")
    async def ws(sock: WebSocket, frames: str = "all") -> None:
        await sock.accept()
        if not srv.can_view(sock):
            await sock.close(code=4401, reason="sign in first")
            return
        client = Client(frames)
        srv.hub.clients.add(client)

        async def reader() -> None:
            while True:
                msg = await sock.receive_json()
                if msg.get("type") == "subscribe":
                    client.subscribe(str(msg.get("frames", "all")))

        read = asyncio.create_task(reader())
        try:
            await sock.send_text(json.dumps(srv.hello()))
            while not read.done():
                try:
                    text = await asyncio.wait_for(client.queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                await sock.send_text(text)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            read.cancel()
            srv.hub.clients.discard(client)

    # ------------------------------------------------------------ web app

    if (FRONTEND_DIST / "assets").exists():
        app.mount(
            "/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets"
        )

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> Any:
        f = FRONTEND_DIST / path
        if path and f.is_file() and FRONTEND_DIST in f.resolve().parents:
            return FileResponse(f)
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse(
            "<p>Build the web app first: cd frontend; npm run build</p>"
        )

    return app
