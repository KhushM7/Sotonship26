"""Live fall detection on a webcam (or a video file), skeleton-only display.

    python scripts/09_live.py --run e9_ens_gru                      # webcam 0
    python scripts/09_live.py --run e9_ens_gru --pose lite          # if under 10 fps
    python scripts/09_live.py --run e9_ens_gru --video clip.mp4 --no-display
    python scripts/09_live.py --run e9_ens_gru --camera 0 --t-low 0.6 --record

Every frame at 15 fps goes through the same code as training and evaluation:
PoseExtractor -> windows.select_keypoints -> occlusion.LastSeenState ->
features.make_features over the last 30 frames -> calibrate.Scorer (one run or an
ensemble) -> calibrate.AlertState with the run's thresholds.yaml. A window is
scored every 3 frames (stride_eval), as in val/test, so the alert numbers match.

The display draws the skeleton on black and never shows the camera image.
Keys: q or Esc quits, r resets the alert state.

--record [DIR] writes video.mp4 (the camera frames the model saw, at 15 fps),
skeleton.npz (Contract 1 layout, so the session can be labelled and evaluated
later), scores.csv (every scored window), alerts.csv and session.yaml. Without DIR
it goes to {FALL_DATA}/recordings/<date-time>, where the monitoring wall plays the
newest session over its video (rooms.yaml `{recording: latest}`). Times in the
recording count frames from its start, so they match video.mp4 even after `r`.
The video is only written to disk; the window on screen stays skeleton-only.
"""

import argparse
import csv
import sys
import time
from collections import deque
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fallsense import calibrate, windows
from fallsense.config import DATA, load_config
from fallsense.features import make_features
from fallsense.io_video import iter_frames
from fallsense.occlusion import LastSeenState
from fallsense.pose import PoseExtractor

CFG = load_config()
FPS = CFG["fps"]
WINDOW = CFG["window"]
STRIDE = CFG["stride_eval"]
CAP_S = CFG["occlusion"]["ctx_cap_s"]
POSE_MODELS = {
    "full": REPO / "models" / "pose" / "pose_landmarker_full.task",
    "lite": REPO / "models" / "pose" / "pose_landmarker_lite.task",
}
# skeleton edges within the 13 config keypoints (nose, shoulders, elbows, wrists,
# hips, knees, ankles)
EDGES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8),
         (7, 9), (9, 11), (8, 10), (10, 12), (0, 1), (0, 2)]  # fmt: skip
COLOURS = {  # BGR
    "OK": (80, 200, 80),
    calibrate.CHECK: (0, 190, 255),
    calibrate.FALL: (40, 40, 230),
    calibrate.ON_FLOOR: (200, 60, 200),
}
BANNER_S = 6.0  # an alert stays on screen this long
WIN_TITLE = "FallSense (skeleton only)"
WARMUP_S = 3.0  # model loading: no fps warning before this
LOWER_BODY = [7, 8, 9, 10, 11, 12]  # hips, knees, ankles in the 13 keypoints


class LivePipeline:
    """Frames in, (window probabilities, alert) out; no display."""

    def __init__(
        self, run_dir: Path, thresholds: dict, on_floor_s: float, wh: tuple[int, int]
    ) -> None:
        self.scorer = calibrate.Scorer(run_dir)
        self.temperature = float(thresholds.get("temperature", 1.0))
        self.th = thresholds
        self.on_floor_s = on_floor_s
        self.wh = np.array(wh)
        self.reset()

    def reset(self) -> None:
        self.ctx_state = LastSeenState(FPS, CAP_S)
        self.kp: deque[np.ndarray] = deque(maxlen=WINDOW)
        self.present: deque[bool] = deque(maxlen=WINDOW)
        self.ctx: deque[np.ndarray] = deque(maxlen=WINDOW)
        self.alerts = calibrate.AlertState(
            float(self.th["t_high"]), float(self.th["t_low"]), self.on_floor_s
        )
        self.i = -1

    def step(
        self, kp33: np.ndarray, detected: bool
    ) -> tuple[np.ndarray, np.ndarray | None, tuple[str, float] | None]:
        """One 15 fps frame -> (13-keypoint pixels, probs or None, alert or None)."""
        self.i += 1
        w, h = int(self.wh[0]), int(self.wh[1])
        kp13 = windows.select_keypoints(kp33[None], np.array([detected]), w, h)[0]
        self.kp.append(kp13)
        self.present.append(detected)
        self.ctx.append(self.ctx_state.update(kp13, detected, self.wh))
        if len(self.kp) < WINDOW or (self.i - (WINDOW - 1)) % STRIDE:
            return kp13, None, None
        present = np.array(self.present)
        ctx = np.stack(self.ctx)
        # the same rule that drops windows from val/test: too long out of sight
        if (~(present | (ctx[:, 0] < CAP_S))).mean() > 0.40:
            return kp13, None, None
        feats = make_features(np.stack(self.kp), self.wh, present, ctx)
        probs = calibrate.softmax_t(self.scorer.logits(feats[None]), self.temperature)[
            0
        ]
        alert = self.alerts.step(self.i / FPS, float(probs[1]), float(probs[2]))
        return kp13, probs, alert


def webcam_frames(
    camera: int, width: int, height: int
) -> Iterator[tuple[int, np.ndarray]]:
    """(t_ms, frame) at 15 fps from a webcam; frames in between are dropped."""
    # default backend (Media Foundation on Windows): DirectShow gave 1 fps on Laptop A
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {camera}")
    t0, n, misses = time.monotonic(), 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # DirectShow drops the first reads after a resolution change
                misses += 1
                if misses > 100:
                    raise SystemExit(
                        "camera stopped returning frames (is another program, or "
                        "another copy of 09_live.py, using it?)"
                    )
                time.sleep(0.02)
                continue
            misses = 0
            t = time.monotonic() - t0
            if t * FPS + 1e-6 >= n:
                # one frame per 1/15 s tick; if processing falls behind, skip ticks
                n = int(t * FPS) + 1
                yield int(t * 1000), frame
    finally:
        cap.release()


def list_cameras(max_index: int = 4) -> None:
    """Print each camera index that returns frames: size, frame rate, brightness and
    whether a person is detected (pick the one that sees the room)."""
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue
        for _ in range(5):
            cap.read()
        t0, n, shape, seen, last = time.monotonic(), 0, None, 0, None
        with PoseExtractor(POSE_MODELS["lite"]) as pose:
            while time.monotonic() - t0 < 2:
                ok, frame = cap.read()
                if ok:
                    n, shape, last = n + 1, frame.shape, frame
                    seen += pose(frame, int(1000 * (time.monotonic() - t0)) + n)[2]
        cap.release()
        if shape and last is not None:
            print(
                f"camera {idx}: {shape[1]}x{shape[0]}, {n / 2:.0f} fps, "
                f"brightness {last.mean():.0f}/255, person in "
                f"{seen / max(n, 1):.0%} of frames"
            )


def draw(
    kp13: np.ndarray,
    wh: tuple[int, int],
    probs: np.ndarray | None,
    banner: tuple[str, float] | None,
    fps: float,
) -> np.ndarray:
    w, h = wh
    img = np.zeros((h, w, 3), np.uint8)
    level = banner[0] if banner else "OK"
    colour = COLOURS[level]
    pts = [
        None if not np.isfinite(p[:2]).all() else (int(p[0]), int(p[1])) for p in kp13
    ]
    for a, b in EDGES:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], colour, 3, cv2.LINE_AA)
    for p in pts:
        if p:
            cv2.circle(img, p, 4, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 44), colour, -1)
    text = level if not banner else f"{level}  ({banner[1]:.2f})"
    cv2.putText(img, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    lower = np.nan_to_num(kp13[LOWER_BODY, 2])
    if not np.isfinite(kp13[:, :2]).any():
        cv2.putText(img, "no person detected - check the camera (--list-cameras)",
                    (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 190, 255), 2)  # fmt: skip
    elif lower.mean() < 0.5:
        # the model was trained on full-body views; a seated close-up reads as ambiguous
        cv2.putText(img, "full body not in view - step back", (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 255), 2)  # fmt: skip
    cv2.putText(img, f"{fps:4.1f} fps", (w - 120, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)  # fmt: skip
    cv2.putText(img, "q quit  r reset", (w - 150, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (140, 140, 140), 1)  # fmt: skip
    if probs is not None:
        for k, (name, p) in enumerate(zip(CFG["classes"], probs)):
            y = h - 20 - 22 * (3 - k)
            cv2.putText(img, f"{name:15s}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)  # fmt: skip
            cv2.rectangle(
                img, (150, y - 12), (150 + int(200 * p), y), (200, 200, 200), -1
            )
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", help="folder in models/ (run or ensemble)")
    ap.add_argument("--list-cameras", action="store_true")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video", type=Path, help="replay a video file instead")
    ap.add_argument("--pose", choices=sorted(POSE_MODELS), default="full")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--t-high", type=float)
    ap.add_argument("--t-low", type=float)
    ap.add_argument(
        "--on-floor",
        type=float,
        help="seconds on the floor before ON FLOOR (default: on_floor_seconds_demo)",
    )
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    ap.add_argument(
        "--record",
        type=Path,
        nargs="?",
        const=Path("auto"),
        help="folder for video/skeleton/scores/alerts "
        "(no value: FALL_DATA/recordings/<date-time>, which the wall picks up)",
    )
    args = ap.parse_args()
    if args.list_cameras:
        list_cameras()
        return 0
    if not args.run:
        ap.error("--run is required")

    run_dir = REPO / "models" / args.run
    th = calibrate.load_thresholds(run_dir)
    if not th:
        raise SystemExit(f"{run_dir} has no thresholds.yaml; run 07_calibrate.py")
    if args.t_high is not None:
        th["t_high"] = args.t_high
    if args.t_low is not None:
        th["t_low"] = args.t_low
    th["t_low"] = min(float(th["t_low"]), float(th["t_high"]))
    on_floor = args.on_floor or float(th.get("on_floor_seconds_demo", 20))
    print(
        f"{args.run}: t_high {th['t_high']}, t_low {th['t_low']}, "
        f"T {th.get('temperature', 1.0)}, ON FLOOR after {on_floor:g} s, "
        f"pose {args.pose}"
    )

    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        wh = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        frames = iter_frames(args.video, FPS)
    else:
        wh = (args.width, args.height)
        frames = webcam_frames(args.camera, *wh)

    if args.record == Path("auto"):
        args.record = DATA / "recordings" / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer: cv2.VideoWriter | None = None
    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        print(f"recording to {args.record}")

    pipe = LivePipeline(run_dir, th, on_floor, wh)
    pose_cfg = CFG["pose"]
    rec_kp, rec_det, rec_t, rec_scores, rec_alerts = [], [], [], [], []
    banner: tuple[str, float] | None = None
    banner_until = 0.0
    tick, n_tick, fps = time.monotonic(), 0, 0.0
    last_t_ms = -1
    first = True
    started = time.monotonic()
    with PoseExtractor(
        POSE_MODELS[args.pose],
        min_det=pose_cfg["min_det"],
        min_presence=pose_cfg["min_presence"],
        min_track=pose_cfg["min_track"],
    ) as pose:
        try:
            for t_ms, frame in frames:
                if first and not args.video:
                    # the camera may ignore the requested size
                    h0, w0 = frame.shape[:2]
                    if (w0, h0) != wh:
                        wh = (w0, h0)
                        pipe = LivePipeline(run_dir, th, on_floor, wh)
                    first = False
                t_ms = max(t_ms, last_t_ms + 1)  # VIDEO mode needs strictly increasing
                last_t_ms = t_ms
                kp33, _, det = pose(frame, t_ms)
                kp13, probs, alert = pipe.step(kp33, det)
                t_s = pipe.i / FPS
                t_rec = len(rec_t) / FPS  # this frame's time in video.mp4
                if args.record:
                    if writer is None:
                        fh, fw = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.record / "video.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS,
                            (fw, fh),
                        )
                    writer.write(frame)
                    rec_kp.append(kp33)
                    rec_det.append(det)
                    rec_t.append(t_ms)
                    if probs is not None:
                        rec_scores.append((round(t_rec, 3), *np.round(probs, 4)))
                if alert is not None:
                    banner, banner_until = alert, t_s + BANNER_S
                    rec_alerts.append((round(t_rec, 3), alert[0], round(alert[1], 4)))
                    print(
                        f"{t_s:8.2f} s  {alert[0]:8s} score {alert[1]:.2f}", flush=True
                    )
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                if banner and t_s > banner_until:
                    banner = None

                n_tick += 1
                now = time.monotonic()
                if now - tick >= 1.0:
                    fps, tick, n_tick = n_tick / (now - tick), now, 0
                    if not args.video and fps < 10 and now - started > WARMUP_S:
                        print(
                            f"WARNING: {fps:.1f} fps (< 10); try --pose lite",
                            flush=True,
                        )
                if not args.no_display:
                    cv2.imshow(WIN_TITLE, draw(kp13, wh, probs, banner, fps))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):  # upper case: Caps Lock on
                        break
                    if cv2.getWindowProperty(WIN_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break  # window closed with its X button
                    if key in (ord("r"), ord("R")):
                        pipe.reset()
                        banner = None
                        print(f"{t_s:8.2f} s  reset", flush=True)
        except KeyboardInterrupt:
            print("stopped (Ctrl+C)")
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    if args.record:
        out = args.record
        np.savez_compressed(
            out / "skeleton.npz",
            kp=np.stack(rec_kp).astype(np.float32) if rec_kp else np.zeros((0, 33, 4)),
            detected=np.array(rec_det, bool),
            t_ms=np.array(rec_t, np.int64),
            fps=np.int64(FPS),
            width=np.int64(wh[0]),
            height=np.int64(wh[1]),
            pose_model=np.str_(POSE_MODELS[args.pose].stem),
        )
        with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["t_s", *CFG["classes"]], *rec_scores])
        with open(out / "alerts.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["t_s", "level", "score"], *rec_alerts])
        with open(out / "session.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"run": args.run, "thresholds": {k: th[k] for k in ("t_high", "t_low")},
                 "on_floor_s": on_floor, "pose": args.pose, "frames": len(rec_t)},
                f, sort_keys=False,
            )  # fmt: skip
        print(f"recorded {len(rec_t)} frames, {len(rec_alerts)} alerts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
