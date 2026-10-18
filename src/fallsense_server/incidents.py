"""Incidents (Contract 7): what a carer handles, from the backend's alert to resolved.

One fall is one incident even when it escalates CHECK -> FALL -> ON FLOOR. Nobody
has to watch a screen: every incident pushes to carers' phones, and an incident
nobody acknowledges climbs the tiers (room carers -> everyone on shift -> senior)
until someone does. The service is plain Python with an injected clock; the app
drains its outboxes (`pushes`, `changed`, `new_events`) after every call.
"""

import json
import sqlite3
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fallsense_server.settings import Settings

CHECK, FALL, ON_FLOOR, OFFLINE = "CHECK", "FALL", "ON FLOOR", "OFFLINE"
LEVEL_RANK = {CHECK: 0, FALL: 1, ON_FLOOR: 2}
PRIORITY = {ON_FLOOR: 5, FALL: 5, OFFLINE: 4, CHECK: 3}  # ntfy 1 (min) .. 5 (urgent)
INFO_PRIORITY = 2
OPEN, ESCALATED, ACKNOWLEDGED, ATTENDING, RESOLVED = (
    "open",
    "escalated",
    "acknowledged",
    "attending",
    "resolved",
)
OUTCOMES = {"real_fall", "near_miss", "false_alarm", "camera_restored"}
TIER_LABEL = {1: "room carers", 2: "everyone on shift", 3: "senior on call"}


@dataclass
class Incident:
    id: str
    room: str
    kind: str  # "fall" or "offline"
    level: str  # highest level reached
    opened_at: float
    state: str = OPEN
    tier: int = 1
    deadline: float | None = None  # next escalation, if nobody acts
    levels: list[dict[str, Any]] = field(default_factory=list)
    notified: list[str] = field(default_factory=list)
    acknowledged_by: str | None = None
    acknowledged_at: float | None = None
    attending_by: str | None = None
    attending_at: float | None = None
    resolved_by: str | None = None
    resolved_at: float | None = None
    outcome: str | None = None
    injury: str | None = None
    false_alarm_reason: str | None = None
    note: str | None = None
    on_floor_since: float | None = None
    floor_ended_at: float | None = None

    @property
    def active(self) -> bool:
        return self.state != RESOLVED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Push:
    carers: list[str]
    priority: int
    title: str
    message: str
    incident_id: str | None
    actions: bool = True  # "I'm going" / "Open" buttons


class Store:
    """SQLite file so incidents survive a server restart mid-demo."""

    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS incidents"
            " (id TEXT PRIMARY KEY, opened_at REAL, data TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS replays (id TEXT PRIMARY KEY, data TEXT)"
        )
        self.db.commit()

    def save(self, inc: Incident) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO incidents VALUES (?, ?, ?)",
            (inc.id, inc.opened_at, json.dumps(inc.to_dict())),
        )
        self.db.commit()

    def load(self) -> list[Incident]:
        rows = self.db.execute("SELECT data FROM incidents ORDER BY opened_at")
        return [Incident(**json.loads(r[0])) for r in rows]

    def save_replay(self, inc_id: str, frames: list[dict[str, Any]]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO replays VALUES (?, ?)", (inc_id, json.dumps(frames))
        )
        self.db.commit()

    def replay(self, inc_id: str) -> list[dict[str, Any]]:
        row = self.db.execute(
            "SELECT data FROM replays WHERE id = ?", (inc_id,)
        ).fetchone()
        return json.loads(row[0]) if row else []

    def clear(self) -> None:
        self.db.execute("DELETE FROM incidents")
        self.db.execute("DELETE FROM replays")
        self.db.commit()


class IncidentService:
    def __init__(
        self,
        settings: Settings,
        store: Store | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings, self.store, self.clock = settings, store, clock
        self.esc = settings.escalation
        self.rooms = {r["id"]: r for r in settings.rooms}
        self.incidents: dict[str, Incident] = {}
        if store is not None:
            self.incidents = {i.id: i for i in store.load()}
        self.carer_in_room: dict[str, float] = {}  # room -> mute expires at
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        # outboxes, drained by the app
        self.pushes: list[Push] = []
        self.changed: list[Incident] = []
        self.new_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------ queries

    def active(self, room: str, kind: str = "fall") -> Incident | None:
        return next(
            (
                i
                for i in self.incidents.values()
                if i.room == room and i.kind == kind and i.active
            ),
            None,
        )

    def get(self, inc_id: str) -> Incident:
        if inc_id not in self.incidents:
            raise KeyError(inc_id)
        return self.incidents[inc_id]

    def muted(self, room: str) -> bool:
        return self.carer_in_room.get(room, 0.0) > self.clock()

    def recipients(self, room: str, tier: int) -> list[str]:
        carers = self.settings.carers
        names = list(self.rooms.get(room, {}).get("carers", []))
        if tier >= 2 or not names:
            names += [c.name for c in carers if c.tier <= 2]
        if tier >= 3:
            names += [c.name for c in carers]
        return list(dict.fromkeys(names))

    # ------------------------------------------------------------ inputs

    def on_alert(
        self, room: str, level: str, score: float, reason: str = "pose"
    ) -> Incident | None:
        """A backend alert for `room`; returns the incident it opened or joined."""
        now = self.clock()
        name = self._room_name(room)
        if self.muted(room):
            self._event(
                "suppressed",
                room,
                f"{level} in {name} while a carer is in the room – not sent",
            )
            return None
        entry = {"t": now, "level": level, "score": round(score, 3), "reason": reason}
        inc = self.active(room)
        if inc is None:
            inc = Incident(
                id=self._next_id(), room=room, kind="fall", level=level, opened_at=now
            )
            inc.levels.append(entry)
            inc.deadline = now + float(self.esc["ack_s"][level])
            self.incidents[inc.id] = inc
            self._event("alert", room, f"{_label(level, reason)} in {name}", inc)
            self._push(inc, self.recipients(room, 1), PRIORITY[level], *self._text(inc))
        else:
            inc.levels.append(entry)
            if LEVEL_RANK[level] > LEVEL_RANK.get(inc.level, -1):
                inc.level = level
                self._event("alert", room, f"{name} now {level}", inc)
                if inc.state in (OPEN, ESCALATED):
                    who = self.recipients(room, inc.tier)
                    self._push(inc, who, PRIORITY[level], *self._text(inc))
                elif inc.state == ACKNOWLEDGED and inc.acknowledged_by:
                    self._push(
                        inc, [inc.acknowledged_by], PRIORITY[level], *self._text(inc)
                    )
        self._changed(inc)
        return inc

    def on_posture(self, room: str, posture: str) -> None:
        """Track time on the floor for the room's open fall incident."""
        inc = self.active(room)
        if inc is None:
            return
        now = self.clock()
        if posture == "on_floor" and inc.on_floor_since is None:
            inc.on_floor_since = now
            self._changed(inc)
        elif (
            posture in ("upright", "low_on_purpose")
            and inc.on_floor_since is not None
            and inc.floor_ended_at is None
        ):
            inc.floor_ended_at = now
            self._event(
                "info", room, f"Person in {self._room_name(room)} is up again", inc
            )
            self._changed(inc)

    def on_offline(self, room: str) -> Incident:
        inc = self.active(room, "offline")
        if inc is not None:
            return inc
        now = self.clock()
        inc = Incident(
            id=self._next_id(), room=room, kind="offline", level=OFFLINE, opened_at=now
        )
        inc.deadline = now + float(self.esc["ack_s"][OFFLINE])
        self.incidents[inc.id] = inc
        self._event("offline", room, f"{self._room_name(room)} camera offline", inc)
        self._push(inc, self.recipients(room, 1), PRIORITY[OFFLINE], *self._text(inc))
        self._changed(inc)
        return inc

    def on_online(self, room: str) -> None:
        inc = self.active(room, "offline")
        if inc is None:
            return
        self._close(inc, "System", outcome="camera_restored")
        self._event("online", room, f"{self._room_name(room)} camera back online", inc)

    # ------------------------------------------------------------ carer actions

    def ack(self, inc_id: str, who: str) -> Incident:
        inc = self.get(inc_id)
        if inc.state in (ATTENDING, RESOLVED):
            return inc
        now = self.clock()
        inc.state, inc.acknowledged_by, inc.acknowledged_at = ACKNOWLEDGED, who, now
        inc.deadline = now + float(self.esc["attend_s"])
        name = self._room_name(inc.room)
        self._event("ack", inc.room, f"{who} is going to {name}", inc)
        others = [c for c in inc.notified if c != who]
        if others:
            self._push(
                inc,
                others,
                INFO_PRIORITY,
                f"{name} covered",
                f"{who} is responding",
                actions=False,
            )
        self._changed(inc)
        return inc

    def attend(self, inc_id: str, who: str) -> Incident:
        inc = self.get(inc_id)
        if inc.state == RESOLVED:
            return inc
        now = self.clock()
        if inc.acknowledged_by is None:
            inc.acknowledged_by, inc.acknowledged_at = who, now
        inc.state, inc.attending_by, inc.attending_at = ATTENDING, who, now
        inc.deadline = None
        self.set_carer_in_room(inc.room, True, who, quiet=True)
        self._event(
            "attend",
            inc.room,
            f"{who} is with the resident in {self._room_name(inc.room)}",
            inc,
        )
        self._changed(inc)
        return inc

    def resolve(
        self,
        inc_id: str,
        who: str,
        outcome: str,
        injury: str | None = None,
        reason: str | None = None,
        note: str | None = None,
    ) -> Incident:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome}")
        inc = self.get(inc_id)
        self._close(inc, who, outcome, injury, reason, note)
        self.carer_in_room.pop(inc.room, None)
        text = outcome.replace("_", " ")
        if reason:
            text += f" ({reason.replace('_', ' ')})"
        self._event(
            "resolve",
            inc.room,
            f"{self._room_name(inc.room)} resolved by {who}: {text}",
            inc,
        )
        return inc

    def dismiss(self, inc_id: str, who: str) -> Incident:
        """CHECK only: "looks fine", recorded as a false alarm."""
        inc = self.get(inc_id)
        if inc.level != CHECK:
            raise ValueError("only a CHECK can be dismissed; resolve a FALL instead")
        return self.resolve(inc_id, who, "false_alarm", reason="looks_fine")

    def escalate_now(self, inc_id: str, who: str) -> Incident:
        """ "Can't go, pass it on": alert the next tier, minus the carer passing it on."""
        inc = self.get(inc_id)
        if not inc.active:
            return inc
        if inc.acknowledged_by == who:  # they are no longer on the way
            inc.acknowledged_by = inc.acknowledged_at = None
        self._escalate(inc, f"{who} can't go", title="PASSED ON", skip=who)
        return inc

    def set_carer_in_room(
        self, room: str, on: bool, who: str, quiet: bool = False
    ) -> None:
        name = self._room_name(room)
        if on:
            self.carer_in_room[room] = self.clock() + float(self.esc["carer_in_room_s"])
            if not quiet:
                self._event("mute", room, f"{who}: carer in {name}, alerts muted")
        elif self.carer_in_room.pop(room, None) is not None:
            self._event("mute", room, f"{who}: carer left {name}, alerts on")

    # ------------------------------------------------------------ time

    def tick(self) -> None:
        """Escalate anything nobody has acted on; expire "carer in room" mutes."""
        now = self.clock()
        for inc in list(self.incidents.values()):
            if not inc.active or inc.deadline is None or now < inc.deadline:
                continue
            if inc.state == ACKNOWLEDGED:
                self._escalate(inc, f"{inc.acknowledged_by} has not reached the room")
            elif inc.state in (OPEN, ESCALATED):
                self._escalate(inc, "No response")
        for room, until in list(self.carer_in_room.items()):
            if until <= now:
                del self.carer_in_room[room]
                self._event(
                    "mute", room, f"{self._room_name(room)}: carer-in-room mute ended"
                )

    def reset(self) -> None:
        self.incidents.clear()
        self.carer_in_room.clear()
        self.events.clear()
        if self.store is not None:
            self.store.clear()

    # ------------------------------------------------------------ internals

    def _escalate(
        self,
        inc: Incident,
        why: str,
        title: str = "NO RESPONSE",
        skip: str | None = None,
    ) -> None:
        now = self.clock()
        inc.tier = min(inc.tier + 1, 3)
        who = [c for c in self.recipients(inc.room, inc.tier) if c != skip]
        while not who and inc.tier < 3:  # nobody else at this tier: go one higher
            inc.tier += 1
            who = [c for c in self.recipients(inc.room, inc.tier) if c != skip]
        inc.state = ESCALATED
        inc.deadline = now + float(self.esc["repeat_s"])
        name = self._room_name(inc.room)
        title = f"{title} – {_label(inc.level)} {name}"
        msg = f"{why}. Escalated to {TIER_LABEL[inc.tier]}. Tap to respond."
        self._event(
            "escalate",
            inc.room,
            f"{why} – {name} escalated to {TIER_LABEL[inc.tier]}",
            inc,
        )
        self._push(inc, who, PRIORITY[inc.level], title, msg)
        self._changed(inc)

    def _close(
        self,
        inc: Incident,
        who: str,
        outcome: str,
        injury: str | None = None,
        reason: str | None = None,
        note: str | None = None,
    ) -> None:
        now = self.clock()
        inc.state, inc.resolved_by, inc.resolved_at = RESOLVED, who, now
        inc.outcome, inc.injury, inc.false_alarm_reason, inc.note = (
            outcome,
            injury,
            reason,
            note,
        )
        inc.deadline = None
        if inc.on_floor_since is not None and inc.floor_ended_at is None:
            inc.floor_ended_at = now
        self._changed(inc)

    def _text(self, inc: Incident) -> tuple[str, str]:
        name = self._room_name(inc.room)
        resident = self.rooms.get(inc.room, {}).get("resident", "")
        reason = inc.levels[-1]["reason"] if inc.levels else "pose"
        body = {
            FALL: "Fall detected",
            ON_FLOOR: "On the floor and has not got up",
            CHECK: "Went out of view low down – please check"
            if reason == "hidden_down"
            else "Possible fall – please check",
            OFFLINE: f"Camera offline – nobody is monitoring {name}",
        }[inc.level]
        who = f" · {resident}" if resident and inc.kind == "fall" else ""
        return f"{_label(inc.level)} – {name}", f"{body}{who} · tap to respond"

    def _push(
        self,
        inc: Incident,
        carers: list[str],
        priority: int,
        title: str,
        message: str,
        actions: bool = True,
    ) -> None:
        if not carers:
            return
        if actions:
            inc.notified = list(dict.fromkeys([*inc.notified, *carers]))
        self.pushes.append(Push(carers, priority, title, message, inc.id, actions))
        if actions:
            self._event("push", inc.room, f"Alert sent to {', '.join(carers)}", inc)

    def _event(
        self, kind: str, room: str, text: str, inc: Incident | None = None
    ) -> None:
        ev = {
            "t": self.clock(),
            "kind": kind,
            "room": room,
            "text": text,
            "incident": inc.id if inc else None,
        }
        self.events.append(ev)
        self.new_events.append(ev)

    def _changed(self, inc: Incident) -> None:
        if self.store is not None:
            self.store.save(inc)
        self.changed.append(inc)

    def _next_id(self) -> str:
        return f"INC-{len(self.incidents) + 1:04d}"

    def _room_name(self, room: str) -> str:
        return str(self.rooms.get(room, {}).get("name", room))


def _label(level: str, reason: str = "pose") -> str:
    if level == CHECK and reason == "hidden_down":
        return "CHECK (out of view)"
    return level
