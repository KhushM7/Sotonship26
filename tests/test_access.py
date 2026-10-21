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

from fallsense_server.app import PRESENTER, STATION, create_app
from fallsense_server.settings import load_settings
from fallsense_server.tunnel import URL

TUNNEL = {"cf-connecting-ip": "203.0.113.7", "cf-ray": "abc"}
CARER = "Khush Mehta"


@pytest.fixture
def client(tmp_path: Path):  # type: ignore[no-untyped-def]
    try:
        app = create_app(
            load_settings(local_overrides=False), start=False, state_dir=tmp_path
        )
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


def _signed(client: TestClient, name: str, pin: str) -> dict[str, str]:
    r = client.post("/api/login", json={"carer": name, "pin": pin}, headers=TUNNEL)
    assert r.status_code == 200, r.text
    return {**TUNNEL, "x-carer": name, "x-token": r.json()["token"]}


def test_each_role_has_its_own_pin(client: TestClient) -> None:
    pins = {"carer": "2468", "station": "1357", "presenter": "8642"}
    r = client.post(
        "/api/login", json={"carer": STATION, "pin": pins["carer"]}, headers=TUNNEL
    )
    assert r.status_code == 401  # the carer PIN does not open the wall
    assert (
        client.post(
            "/api/login",
            json={"carer": PRESENTER, "pin": pins["presenter"]},
            headers=TUNNEL,
        ).json()["role"]
        == "presenter"
    )


def test_demo_controls_need_the_presenter(client: TestClient) -> None:
    assert client.post("/api/demo/reset", headers=TUNNEL).status_code == 401
    carer = _signed(client, CARER, "2468")
    assert client.post("/api/demo/reset", headers=carer).status_code == 403
    station = _signed(client, STATION, "1357")
    assert client.post("/api/demo/reset", headers=station).status_code == 403
    presenter = _signed(client, PRESENTER, "8642")
    assert client.post("/api/demo/reset", headers=presenter).status_code == 200


def test_the_wall_acts_as_the_nurse_station(client: TestClient) -> None:
    r = client.post("/api/rooms/r4/carer_in_room", json={"on": True}, headers=TUNNEL)
    assert r.status_code == 401
    station = _signed(client, STATION, "1357")
    r = client.post("/api/rooms/r4/carer_in_room", json={"on": True}, headers=station)
    assert r.status_code == 200
    assert client.get("/api/state", headers=station).status_code == 200


def test_deployed_laptop_is_not_trusted(tmp_path: Path) -> None:
    settings = load_settings(local_overrides=False)
    settings.raw["trust_localhost"] = False
    app = create_app(settings, start=False, state_dir=tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/state").status_code == 401
        assert c.post("/api/demo/reset").status_code == 401


def test_public_demo_needs_no_sign_in_and_never_pushes(tmp_path: Path) -> None:
    settings = load_settings(local_overrides=False)
    settings.raw["public_demo"] = True
    settings.raw["trust_localhost"] = False
    for c in settings.carers:
        c.topic = "a-real-topic"  # ignored: the demo never posts to ntfy
    app = create_app(settings, start=False, state_dir=tmp_path)

    async def no_ntfy(*_: object) -> None:
        raise AssertionError("the demo posted to ntfy")

    app.state.server.notifier.send = no_ntfy
    with TestClient(app) as c:
        assert c.get("/api/public", headers=TUNNEL).json()["demo"] is True
        assert c.get("/api/state", headers=TUNNEL).json()["demo"] is True
        assert c.post("/api/demo/reset", headers=TUNNEL).status_code == 200
        # a browser still signed in from a real shift may use the controls too
        r = c.post("/api/login", json={"carer": STATION, "pin": "1357"}, headers=TUNNEL)
        station = {**TUNNEL, "x-carer": STATION, "x-token": r.json()["token"]}
        assert c.post("/api/demo/reset", headers=station).status_code == 200
        with c.websocket_connect("/ws?frames=none", headers=TUNNEL) as ws:
            assert '"type":"hello"' in ws.receive_text().replace(" ", "")
            phone = {**TUNNEL, "x-carer": CARER}  # a name, no token
            r = c.post(f"/api/carers/{CARER}/test", json={}, headers=phone)
            assert r.json()["results"] == [[CARER, "demo"]]
            while '"type":"push"' not in (text := ws.receive_text()):
                pass
            assert CARER in text
        assert c.get("/api/idle", headers=TUNNEL).status_code == 403
        assert c.get("/api/idle").json()["viewers"] == 0


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
