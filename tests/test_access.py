"""Through the Cloudflare tunnel, nothing is trusted: every read and action needs a sign-in.

cloudflared relays requests from localhost, so the server tells them apart by the
forwarding headers it adds. Needs the replay data (skipped otherwise).
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
if not os.environ.get("FALL_DATA"):
    pytest.skip("FALL_DATA not set", allow_module_level=True)

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fallsense_server.app import create_app
from fallsense_server.settings import load_settings
from fallsense_server.tunnel import URL

TUNNEL = {"cf-connecting-ip": "203.0.113.7", "cf-ray": "abc"}
CARER = "Amara Okafor"


@pytest.fixture
def client(tmp_path: Path):  # type: ignore[no-untyped-def]
    try:
        app = create_app(load_settings(), start=False, state_dir=tmp_path)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"replay data not available: {e}")
    with TestClient(app) as c:
        yield c


def _token(client: TestClient) -> str:
    r = client.post("/api/login", json={"carer": CARER, "pin": "2468"}, headers=TUNNEL)
    return str(r.json()["token"])


def test_tunnel_reads_need_a_sign_in(client: TestClient) -> None:
    assert client.get("/api/state").status_code == 200  # the laptop itself
    assert client.get("/api/state", headers=TUNNEL).status_code == 401
    assert client.get("/api/public", headers=TUNNEL).status_code == 200
    signed = {**TUNNEL, "x-carer": CARER, "x-token": _token(client)}
    assert client.get("/api/state", headers=signed).status_code == 200
    bad = {**TUNNEL, "x-carer": CARER, "x-token": "nope"}
    assert client.get("/api/incidents", headers=bad).status_code == 401


def test_tunnel_cannot_act_as_the_nurse_station_or_presenter(
    client: TestClient,
) -> None:
    r = client.post("/api/rooms/r4/carer_in_room", json={"on": True}, headers=TUNNEL)
    assert r.status_code == 401
    r = client.post("/api/demo/reset", headers=TUNNEL)
    assert r.status_code == 403
    token = _token(client)
    r = client.post(
        "/api/demo/reset", json={"carer": CARER, "token": token}, headers=TUNNEL
    )
    assert r.status_code == 403  # presenter controls stay on the laptop


def test_wrong_pins_are_limited(client: TestClient) -> None:
    for _ in range(5):
        r = client.post(
            "/api/login", json={"carer": CARER, "pin": "0000"}, headers=TUNNEL
        )
        assert r.status_code == 401
    r = client.post("/api/login", json={"carer": CARER, "pin": "2468"}, headers=TUNNEL)
    assert r.status_code == 429


def test_tunnel_websocket_needs_a_sign_in(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as e,
        client.websocket_connect("/ws?frames=none", headers=TUNNEL) as ws,
    ):
        ws.receive_text()
    assert e.value.code == 4401
    q = f"frames=none&carer={CARER.replace(' ', '+')}&token={_token(client)}"
    with client.websocket_connect(f"/ws?{q}", headers=TUNNEL) as ws:
        assert '"type":"hello"' in ws.receive_text().replace(" ", "")


def test_tunnel_address_is_recognised() -> None:
    line = "INF |  https://quiet-river-42.trycloudflare.com                        |"
    m = URL.search(line)
    assert m is not None and m.group(0) == "https://quiet-river-42.trycloudflare.com"
