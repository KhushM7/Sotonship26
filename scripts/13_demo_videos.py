"""Get the video under every room's skeleton ready: fetch, then transcode for browsers.

    python scripts/13_demo_videos.py

For each clip in configs/rooms.yaml (and the newest 09_live.py recording) it finds
the original video, downloads GMDCSA24 clips that are not on this laptop from the
dataset's GitHub, and writes the H.264 copy the wall plays to {FALL_DATA}/cache/web.
The server does the same on demand; running this first keeps the demo snappy.
Clips without a video still play, as skeleton only.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fallsense_server.clips import ClipRef
from fallsense_server.settings import load_settings
from fallsense_server.videos import (
    fetch_source,
    source_video,
    web_video,
)


def main() -> int:
    refs: list[ClipRef] = []
    for room in load_settings().rooms:
        entries = [*room.get("idle", []), *(room.get("scenarios") or {}).values()]
        for d in entries:
            ref = ClipRef.from_cfg(d).resolved()
            if ref is not None and ref not in refs:
                refs.append(ref)
    missing = 0
    for ref in refs:
        src = source_video(ref.dataset, ref.path)
        how = "here"
        if src is None:
            try:
                src = fetch_source(ref.dataset, ref.path)
                how = "downloaded"
            except OSError as e:
                print(f"  download failed: {e}")
        if src is None:
            how = "cached"  # the source is gone, but its browser copy is kept
        out = web_video(ref.dataset, ref.path)
        missing += out is None
        print(f"{ref.dataset}/{ref.path}: {how if out else 'NO VIDEO (skeleton only)'}")
    print(f"{len(refs) - missing} of {len(refs)} clips have video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
