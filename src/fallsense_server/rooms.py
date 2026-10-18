"""Room streams: the boundary between the backend and everything after an alert.

Any camera source must produce the same four messages (Contract 6):

    frame   15 Hz  13 keypoints (normalised x, y, visibility; None where lost), present
    scores   5 Hz  [p_upright, p_falling, p_fallen, p_controlled_low], fall_score, posture
    alert          level, score, reason – decided by the backend, never here
    (status is added by the app)

`ReplayRoom` plays recorded clips with the backend's stored scores and alerts. A
`LiveRoom` will wrap the backend's live loop (09_live.py) once it exists; nothing
downstream changes.
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from fallsense_server.clips import FPS, Clip, ClipRef, load_clip

POSTURE = ["upright", "falling", "on_floor", "low_on_purpose"]  # argmax label
NOT_VISIBLE_S = 1.0  # no skeleton this long -> "not_visible"
SCORE_WINDOWS = 3  # fall score shown = max p_falling over the last 3 windows

Emit = Callable[[dict[str, Any]], None]


class ReplayRoom:
    def __init__(
        self, cfg: dict[str, Any], run: str, emit: Emit, replay_s: float = 10.0
    ) -> None:
        self.cfg, self.id, self.model_run, self.emit = cfg, cfg["id"], run, emit
        self.idle = [ClipRef.from_cfg(d) for d in cfg.get("idle", [])]
        self.scenarios = {
            k: ClipRef.from_cfg(d) for k, d in (cfg.get("scenarios") or {}).items()
        }
        self.clips: dict[ClipRef, Clip] = {
            ref: load_clip(ref, run) for ref in [*self.idle, *self.scenarios.values()]
        }
        self.offline = False
        self.ring: deque[dict[str, Any]] = deque(maxlen=round(replay_s * FPS))
        self.clip: Clip | None = None
        self.scenario: str | None = None
        self.idle_pos = -1
        self.clip_start = 0.0
        self.frame_i = self.win_i = -1
        self.alert_i = 0
        self.last_frame_at = 0.0
        self.last_seen_at = 0.0
        self.posture = "upright"
        self.frame_times: deque[float] = deque(maxlen=30)

    # ------------------------------------------------------------ control

    def play_scenario(self, name: str) -> None:
        self._start(self.scenarios[name], name)

    def set_offline(self, on: bool) -> None:
        self.offline = on

    def _next_idle(self) -> None:
        if not self.idle:
            self.clip = None
            return
        self.idle_pos = (self.idle_pos + 1) % len(self.idle)
        self._start(self.idle[self.idle_pos], None)

    def _start(self, ref: ClipRef, scenario: str | None) -> None:
        self.clip, self.scenario = self.clips[ref], scenario
        self.clip_start = time.time()
        self.frame_i = self.win_i = -1
        self.alert_i = 0

    # ------------------------------------------------------------ playback

    async def run(self) -> None:
        period = 1.0 / FPS
        while True:
            self.step(time.time())
            await asyncio.sleep(period)

    def step(self, now: float) -> None:
        if self.offline:
            return
        if self.clip is None or now - self.clip_start >= self.clip.duration_s:
            self._next_idle()
            if self.clip is None:
                return
        clip, ct = self.clip, now - self.clip_start

        fi = min(int(ct * FPS), len(clip.kp) - 1)
        if fi != self.frame_i:
            self.frame_i = fi
            self._emit_frame(clip, fi, now)

        wi = self.win_i
        while wi + 1 < len(clip.win_t) and clip.win_t[wi + 1] <= ct:
            wi += 1
        if wi != self.win_i:
            self.win_i = wi
            self._emit_scores(clip, wi, now, ct)

        while self.alert_i < len(clip.alerts) and clip.alerts[self.alert_i][0] <= ct:
            _, level, score = clip.alerts[self.alert_i]
            self.alert_i += 1
            self.emit(
                {
                    "type": "alert",
                    "room": self.id,
                    "t": now,
                    "level": level,
                    "score": round(score, 3),
                    "reason": "pose",
                }
            )

    def _emit_frame(self, clip: Clip, fi: int, now: float) -> None:
        present = bool(clip.present[fi])
        kp = [
            None
            if not np.isfinite(x)
            else [round(float(x), 4), round(float(y), 4), round(float(v), 2)]
            for x, y, v in clip.kp[fi]
        ]
        if present:
            self.last_seen_at = now
        self.last_frame_at = now
        self.frame_times.append(now)
        msg = {"type": "frame", "room": self.id, "t": now, "kp": kp, "present": present}
        self.ring.append({"t": now, "kp": kp, "present": present})
        self.emit(msg)

    def _emit_scores(self, clip: Clip, wi: int, now: float, ct: float) -> None:
        p = clip.probs[wi]
        recent = clip.probs[max(0, wi - SCORE_WINDOWS + 1) : wi + 1, 1]
        if now - self.last_seen_at > NOT_VISIBLE_S:
            self.posture = "not_visible"
        else:
            self.posture = POSTURE[int(np.argmax(p))]
        self.emit(
            {
                "type": "scores",
                "room": self.id,
                "t": now,
                "p": [round(float(x), 4) for x in p],
                "fall_score": round(float(recent.max()), 4),
                "posture": self.posture,
                "held": clip.held_from_s is not None and ct >= clip.held_from_s,
            }
        )

    # ------------------------------------------------------------ status

    def status(self, now: float, offline_after_s: float) -> dict[str, Any]:
        ft = self.frame_times
        fps = (
            (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 1 and ft[-1] > ft[0] else 0.0
        )
        clip = self.clip
        return {
            "room": self.id,
            "online": now - self.last_frame_at < offline_after_s,
            "source": "replay",
            "clip": f"{clip.ref.dataset}/{clip.ref.path}" if clip else None,
            "scenario": self.scenario,
            "aspect": round(clip.aspect, 4) if clip else 16 / 9,
            "fps": round(fps, 1),
        }

    def snapshot(self) -> list[dict[str, Any]]:
        """The last few seconds of skeletons, for an incident's replay."""
        return list(self.ring)
