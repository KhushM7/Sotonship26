"""The server on real replay data: a scenario fall becomes an incident a carer can close.

Needs FALL_DATA and the backend's stored scores (FALLSENSE_BACKEND); skipped otherwise.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
if not os.environ.get("FALL_DATA"):
    pytest.skip("FALL_DATA not set", allow_module_level=True)

from fastapi.testclient import TestClient

from fallsense_server.app import create_app
from fallsense_server.hub import MESSAGE_TYPES, Client
from fallsense_server.settings import load_settings


@pytest.fixture
def app_and_server(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = load_settings()
    try:
        app = create_app(settings, start=False, state_dir=tmp_path)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"replay data not available: {e}")
    return app, app.state.server


def _play(server, room_id: str, seconds: float) -> None:  # type: ignore[no-untyped-def]
    room = server.rooms[room_id]
    room.play_scenario("fall")
    room.clip_start = 0.0
    for t in np.arange(0, seconds, 1 / 15):
        room.step(float(t))


def test_scenario_fall_reaches_ack_attend_resolve(app_and_server) -> None:  # type: ignore[no-untyped-def]
    app, server = app_and_server
    _play(server, "r4", 30.0)
    open_incs = [i for i in server.service.incidents.values() if i.active]
    assert len(open_incs) == 1
    inc = open_incs[0]
    assert inc.room == "r4" and inc.level in ("FALL", "ON FLOOR")

    with TestClient(app) as client:
        assert client.get(f"/api/incidents/{inc.id}/replay").json()["frames"]
        bad = client.post(
            f"/api/incidents/{inc.id}/ack", json={"carer": "Amara Okafor", "token": "x"}
        )
        assert bad.status_code == 200  # the test client counts as this laptop
        tok = client.post(
            "/api/login", json={"carer": "Amara Okafor", "pin": "2468"}
        ).json()["token"]
        r = client.post(
            f"/api/push/{inc.id}/ack", params={"carer": "Amara Okafor", "token": tok}
        )
        assert r.json()["acknowledged_by"] == "Amara Okafor"
        client.post(f"/api/incidents/{inc.id}/attend", json={})
        r = client.post(
            f"/api/incidents/{inc.id}/resolve",
            json={"outcome": "real_fall", "injury": "none"},
        )
        assert r.json()["state"] == "resolved"
        assert "real_fall" in client.get("/api/export/incidents.csv").text


def test_messages_carry_skeletons_never_pixels(app_and_server) -> None:  # type: ignore[no-untyped-def]
    _, server = app_and_server
    spy = Client("all")
    server.hub.clients.add(spy)
    _play(server, "r5", 10.0)
    msgs = []
    while not spy.queue.empty():
        msgs.append(json.loads(spy.queue.get_nowait()))
    assert {m["type"] for m in msgs} <= MESSAGE_TYPES
    frames = [m for m in msgs if m["type"] == "frame"]
    assert frames
    for m in frames:
        assert set(m) == {"type", "room", "t", "kp", "present"}
        assert len(m["kp"]) == 13
        assert all(p is None or len(p) == 3 for p in m["kp"])
    # nothing image-sized is ever sent
    assert max(len(json.dumps(m)) for m in msgs) < 4000
