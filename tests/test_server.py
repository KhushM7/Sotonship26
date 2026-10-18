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
    settings = load_settings(local_overrides=False)
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
            f"/api/incidents/{inc.id}/ack", json={"carer": "Khush Mehta", "token": "x"}
        )
        assert bad.status_code == 200  # the test client counts as this laptop
        tok = client.post(
            "/api/login", json={"carer": "Khush Mehta", "pin": "2468"}
        ).json()["token"]
        r = client.post(
            f"/api/push/{inc.id}/ack", params={"carer": "Khush Mehta", "token": tok}
        )
        assert r.json()["acknowledged_by"] == "Khush Mehta"
        client.post(f"/api/incidents/{inc.id}/attend", json={})
        r = client.post(
            f"/api/incidents/{inc.id}/resolve",
            json={"outcome": "real_fall", "injury": "none"},
        )
        assert r.json()["state"] == "resolved"
        assert "real_fall" in client.get("/api/export/incidents.csv").text


def test_messages_carry_skeletons_never_pixels(app_and_server) -> None:  # type: ignore[no-untyped-def]
    """Frames are skeletons plus the clip time; the video itself is a separate file."""
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
        assert set(m) == {"type", "room", "t", "kp", "present", "ct", "video"}
        assert len(m["kp"]) == 13
        assert all(p is None or len(p) == 3 for p in m["kp"])
    # nothing image-sized is ever sent
    assert max(len(json.dumps(m)) for m in msgs) < 4000


def test_video_only_for_clips_the_rooms_play(app_and_server) -> None:  # type: ignore[no-untyped-def]
    app, server = app_and_server
    clip = next(c for c in server.rooms["r1"].clips.values())
    with TestClient(app) as client:
        assert client.get("/api/video/le2i/Coffee_room_01/video_1").status_code == 404
        assert client.get("/api/video/GMDCSA24/Subject_1/ADL/01").status_code == 404
        if clip.video:  # needs the clip's video on this laptop (13_demo_videos.py)
            r = client.get(clip.video, headers={"Range": "bytes=0-99"})
            assert r.status_code == 206 and r.headers["content-type"] == "video/mp4"


def _fake_recording(folder: Path) -> None:
    """A 09_live.py --record session: 4 s standing, then a FALL alert at 3 s."""
    folder.mkdir(parents=True)
    n = 60
    kp = np.full((n, 33, 4), 0.5, np.float32)
    np.savez_compressed(
        folder / "skeleton.npz",
        kp=kp,
        detected=np.ones(n, bool),
        t_ms=np.arange(n) * 66,
        fps=np.int64(15),
        width=np.int64(640),
        height=np.int64(480),
    )
    rows = ["t_s,upright,falling,fallen,controlled_low"]
    rows += [f"{t:.3f},0.9,0.05,0.03,0.02" for t in np.arange(29, n, 3) / 15]
    (folder / "scores.csv").write_text("\n".join(rows) + "\n")
    (folder / "alerts.csv").write_text("t_s,level,score\n3.0,FALL,0.8\n")
    (folder / "session.yaml").write_text("run: test\n")


def test_live_scenario_plays_the_latest_recording(  # type: ignore[no-untyped-def]
    app_and_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, server = app_and_server
    monkeypatch.setenv("FALLSENSE_RECORDINGS", str(tmp_path / "recordings"))
    room = server.rooms["r4"]
    with pytest.raises(KeyError):
        room.play_scenario("live")  # nothing recorded yet
    _fake_recording(tmp_path / "recordings" / "20261018-120000")
    room.play_scenario("live")
    assert room.clip is not None and room.clip.ref.path == "20261018-120000"
    room.clip_start = 0.0
    for t in np.arange(0, 5, 1 / 15):
        room.step(float(t))
    assert room.status(5.0, 5.0)["source"] == "recording"
    incs = [i for i in server.service.incidents.values() if i.room == "r4"]
    assert incs and incs[0].level == "FALL"
