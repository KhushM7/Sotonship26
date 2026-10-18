import pytest

from fallsense_server.incidents import (
    ACKNOWLEDGED,
    ATTENDING,
    ESCALATED,
    RESOLVED,
    IncidentService,
)
from fallsense_server.settings import Carer, Settings

ESC = {
    "ack_s": {"FALL": 30, "ON FLOOR": 30, "CHECK": 90, "OFFLINE": 60},
    "repeat_s": 30,
    "attend_s": 180,
    "carer_in_room_s": 900,
}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _service() -> tuple[IncidentService, Clock]:
    settings = Settings(
        raw={"escalation": ESC},
        carers=[
            Carer("Amara", 1),
            Carer("Tomasz", 1),
            Carer("Kofi", 2),
            Carer("Ruth", 3),
        ],
        rooms=[{"id": "r4", "name": "Room 4", "resident": "D.R.", "carers": ["Amara"]}],
    )
    clock = Clock()
    return IncidentService(settings, clock=clock), clock


def test_one_fall_is_one_incident_through_escalation_levels() -> None:
    svc, clock = _service()
    a = svc.on_alert("r4", "FALL", 0.8)
    clock.t += 20
    b = svc.on_alert("r4", "ON FLOOR", 0.9)
    assert a is b and a is not None
    assert a.level == "ON FLOOR"
    assert [x["level"] for x in a.levels] == ["FALL", "ON FLOOR"]
    # the room carer is pushed urgently both times
    assert [p.carers for p in svc.pushes] == [["Amara"], ["Amara"]]
    assert all(p.priority == 5 for p in svc.pushes)


def test_unacknowledged_fall_climbs_the_tiers() -> None:
    svc, clock = _service()
    inc = svc.on_alert("r4", "FALL", 0.8)
    assert inc is not None
    svc.pushes.clear()
    clock.t += 29
    svc.tick()
    assert svc.pushes == []
    clock.t += 1
    svc.tick()
    assert inc.state == ESCALATED and inc.tier == 2
    assert set(svc.pushes[-1].carers) == {"Amara", "Tomasz", "Kofi"}
    clock.t += 30
    svc.tick()
    assert inc.tier == 3 and "Ruth" in svc.pushes[-1].carers
    clock.t += 30
    svc.tick()  # keeps re-notifying at the top tier until someone acknowledges
    assert inc.tier == 3 and len(svc.pushes) == 3


def test_ack_attend_resolve() -> None:
    svc, clock = _service()
    inc = svc.on_alert("r4", "FALL", 0.8)
    assert inc is not None
    svc.on_posture("r4", "on_floor")
    clock.t += 5
    svc.ack(inc.id, "Amara")
    assert inc.state == ACKNOWLEDGED and inc.acknowledged_by == "Amara"
    clock.t += 60
    svc.attend(inc.id, "Amara")
    assert inc.state == ATTENDING and inc.deadline is None
    assert svc.muted("r4")
    # alerts while the carer is in the room are logged, not pushed
    n = len(svc.pushes)
    assert svc.on_alert("r4", "ON FLOOR", 0.9) is None
    assert len(svc.pushes) == n
    clock.t += 30
    svc.resolve(inc.id, "Amara", "real_fall", injury="none")
    assert inc.state == RESOLVED and inc.outcome == "real_fall"
    assert inc.on_floor_since == 1000.0 and inc.floor_ended_at == 1095.0
    assert not svc.muted("r4")


def test_acknowledged_but_never_arrives_escalates() -> None:
    svc, clock = _service()
    inc = svc.on_alert("r4", "FALL", 0.8)
    assert inc is not None
    svc.ack(inc.id, "Amara")
    clock.t += 181
    svc.tick()
    assert inc.state == ESCALATED and inc.tier == 2


def test_carer_in_room_mutes_new_incidents_then_expires() -> None:
    svc, clock = _service()
    svc.set_carer_in_room("r4", True, "Amara")
    assert svc.on_alert("r4", "FALL", 0.8) is None
    assert svc.events[-1]["kind"] == "suppressed"
    clock.t += 901
    svc.tick()
    assert svc.on_alert("r4", "FALL", 0.8) is not None


def test_only_check_can_be_dismissed() -> None:
    svc, _ = _service()
    fall = svc.on_alert("r4", "FALL", 0.8)
    assert fall is not None
    with pytest.raises(ValueError):
        svc.dismiss(fall.id, "Amara")
    svc.resolve(fall.id, "Amara", "false_alarm", reason="lying_on_purpose")
    check = svc.on_alert("r4", "CHECK", 0.4, reason="hidden_down")
    assert check is not None and svc.pushes[-1].priority == 3
    svc.dismiss(check.id, "Amara")
    assert check.outcome == "false_alarm" and check.false_alarm_reason == "looks_fine"


def test_camera_offline_is_an_incident_and_clears_itself() -> None:
    svc, _ = _service()
    inc = svc.on_offline("r4")
    assert svc.pushes[-1].priority == 4
    assert svc.on_offline("r4") is inc
    svc.on_online("r4")
    assert inc.state == RESOLVED and inc.outcome == "camera_restored"
