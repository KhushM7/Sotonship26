"""The video under each skeleton: the clip a room replays, or a 09_live.py recording.

Sources are the datasets' own files (inventory.csv `file`, else
{FALL_DATA}/cache/videos/<dataset>/video/<path>.mp4) and a recording's video.mp4.
Each is transcoded once to a small H.264 MP4 under {FALL_DATA}/cache/web, which every
browser plays and seeks. Skeleton frame i is at i / fps seconds into the source, as
in 03_extract_skeletons.py and 09_live.py, so the wall lines the two up by clip time.
"""

import logging
import os
import shutil
import subprocess
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from fallsense.config import DATA

log = logging.getLogger("fallsense.videos")
CACHE = DATA / "cache" / "videos"
WEB = DATA / "cache" / "web"
RECORDING = "recording"  # ClipRef.dataset of a 09_live.py recording
EXTS = (".mp4", ".avi", ".mov", ".mkv")
WEB_HEIGHT = 480
# GMDCSA24 is on GitHub (MIT); the other datasets come from 01_download.py's caches
GMDCSA24_RAW = (
    "https://raw.githubusercontent.com/ekramalam/"
    "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/master/"
)
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def recordings_dir() -> Path:
    """Where 09_live.py --record puts sessions (FALLSENSE_RECORDINGS, else FALL_DATA)."""
    return Path(os.environ.get("FALLSENSE_RECORDINGS", DATA / "recordings"))


def latest_recording() -> str | None:
    """The newest recording folder that has finished writing (session.yaml exists)."""
    root = recordings_dir()
    done = [p.parent for p in root.glob("*/session.yaml")] if root.exists() else []
    return max(done, key=lambda p: p.stat().st_mtime).name if done else None


@lru_cache(maxsize=1)
def _inventory() -> dict[tuple[str, str], str]:
    path = DATA / "inventory.csv"
    if not path.exists():
        return {}
    inv = pd.read_csv(path, usecols=["dataset", "path", "file"])
    return {(r.dataset, r.path): str(r.file) for r in inv.itertuples()}


def source_video(dataset: str, path: str) -> Path | None:
    """The original video of a clip or recording, if it is on this laptop."""
    if dataset == RECORDING:
        p = recordings_dir() / path / "video.mp4"
        return p if p.exists() else None
    known = _inventory().get((dataset, path))
    if known and Path(known).exists():
        return Path(known)
    for ext in EXTS:
        p = CACHE / dataset / "video" / f"{path}{ext}"
        if p.exists():
            return p
    return None


def fetch_source(dataset: str, path: str) -> Path | None:
    """Download a missing clip video where there is a public per-file source."""
    if dataset != "GMDCSA24":
        return None
    subject, kind, name = path.split("/")  # Subject_2/ADL/07
    url = GMDCSA24_RAW + quote(f"{subject.replace('_', ' ')}/{kind}/{name}.mp4")
    out = CACHE / dataset / "video" / f"{path}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(out)
    return out


def web_path(dataset: str, path: str) -> Path:
    return WEB / dataset / f"{path}.mp4"


def web_video(dataset: str, path: str) -> Path | None:
    """The browser copy of a clip's video, transcoding it on first use."""
    src = source_video(dataset, path)
    out = web_path(dataset, path)
    if src is None:
        return out if out.exists() else None
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    with _locks_guard:
        lock = _locks.setdefault(f"{dataset}/{path}", threading.Lock())
    with lock:
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            return out
        if not transcode(src, out):
            return None
    return out


def transcode(src: Path, out: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.warning("ffmpeg not found: no video for %s (skeleton only)", src)
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-an",
        "-vf", f"scale=-2:'trunc(min({WEB_HEIGHT},ih)/2)*2'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp),
    ]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        log.warning("ffmpeg failed on %s: %s", src, res.stderr.strip()[-300:])
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


def video_url(dataset: str, path: str) -> str | None:
    """The wall's address for a clip's video, or None if there is none (skeleton only)."""
    if source_video(dataset, path) is None and not web_path(dataset, path).exists():
        return None
    return f"/api/video/{dataset}/{path}"
