"""Pack what the server needs into one tarball for the AWS instance (scripts/aws.ps1).

    python scripts/aws/bundle.py OUT.tar.gz

app/    the repo's tracked files as they are on disk, frontend/dist, the model run's
        stored scores and thresholds, and configs/server.local.yaml
data/   a FALL_DATA holding only what configs/rooms.yaml plays: index.csv, each
        clip's skeleton and browser video, and any 09_live.py recordings
state/  server_data/secret.key, so carers' links and sign-ins work on both servers

The laptop's FALL_DATA is ~120 GB; the bundle is a few tens of MB.
"""

import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fallsense.config import DATA, skeleton_path
from fallsense_server.clips import ClipRef, clip_id
from fallsense_server.settings import FRONTEND_DIST, STATE_DIR, load_settings
from fallsense_server.videos import recordings_dir, web_path


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout.decode()
    return [REPO / p for p in out.split("\0") if p and (REPO / p).is_file()]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    settings = load_settings()
    run, split = settings.raw["model_run"], settings.raw["scores_split"]
    files: dict[str, Path] = {}  # name in the tarball -> file on disk
    missing: list[str] = []

    def add(name: str, path: Path, required: bool = True) -> None:
        if path.is_file():
            files[name] = path
        elif required:
            missing.append(str(path))

    # this folder too, committed or not (its requirements.txt is installed from here)
    for p in [*tracked_files(), *Path(__file__).parent.glob("*.*")]:
        files["app/" + p.relative_to(REPO).as_posix()] = p
    if not (FRONTEND_DIST / "index.html").exists():
        missing.append(f"{FRONTEND_DIST} (cd frontend; npm run build)")
    for p in FRONTEND_DIST.rglob("*"):
        if p.is_file():
            files["app/" + p.relative_to(REPO).as_posix()] = p
    add("app/configs/server.local.yaml", REPO / "configs" / "server.local.yaml")
    scores = Path("reports") / "scores" / run
    splits = {split}  # each clip's scores are in its own split's file
    for th in (Path("models") / run / "thresholds.yaml", scores / "thresholds.yaml"):
        add(f"app/{th.as_posix()}", REPO / th, required=False)
    if not any(n.endswith("/thresholds.yaml") for n in files):
        missing.append(f"thresholds.yaml for {run}")
    add("state/secret.key", STATE_DIR / "secret.key", required=False)

    add("data/skeletons/index.csv", DATA / "skeletons" / "index.csv")
    for room in settings.rooms:
        for d in [*room.get("idle", []), *(room.get("scenarios") or {}).values()]:
            ref = ClipRef.from_cfg(d).resolved()
            if ref is None:
                missing.append(f"{room['id']}: no recording yet")
            elif ref.is_recording:
                folder = recordings_dir() / ref.path
                for p in folder.iterdir():
                    add(f"data/recordings/{ref.path}/{p.name}", p)
                web = web_path(ref.dataset, ref.path)
                add(f"data/{web.relative_to(DATA).as_posix()}", web, required=False)
            else:
                splits.add(clip_id(ref)[1])
                sk = skeleton_path(ref.dataset, ref.path)
                add(f"data/{sk.relative_to(DATA).as_posix()}", sk)
                web = web_path(ref.dataset, ref.path)
                # no browser video: the room still plays, as skeleton only
                add(f"data/{web.relative_to(DATA).as_posix()}", web, required=False)

    for sp in sorted(splits):
        add(f"app/{scores.as_posix()}/{sp}.parquet", REPO / scores / f"{sp}.parquet")
    if missing:
        print("Missing, so not bundled:\n  " + "\n  ".join(missing))
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name)
    size = sum(p.stat().st_size for p in files.values()) / 1e6
    print(f"{out}: {len(files)} files, {size:.1f} MB before compression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
