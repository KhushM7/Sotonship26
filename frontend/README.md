# fallsense – monitoring wall and carer alerts

The web app and server described in `docs/Frontend Plan (Fall Detection).md`. Only what
happens *after* the backend decides: incidents, phone alerts, escalation, the wall and
the demo. It reads the backend's outputs and never re-implements them.

## Run it

```powershell
# The server shares this checkout and .venv with the backend: it reads
# models/<model_run> and reports/scores/ from this repo.

# once: server packages (in requirements.txt) and the web app
uv pip install --link-mode=copy --python .venv\Scripts\python.exe fastapi "uvicorn[standard]" httpx
cd frontend; npm install; npm run build; cd ..

# once, and after changing rooms.yaml: each clip's original video (needs ffmpeg)
$env:FALL_DATA = "C:\fall-data"                          # skeletons, index.csv, videos
.venv\Scripts\python.exe scripts\13_demo_videos.py

# every time
.venv\Scripts\python.exe scripts\11_serve.py
```

Every room shows its clip's original video with the model's skeleton drawn over it.
Videos come from `{FALL_DATA}/cache/videos` (GMDCSA24 clips are downloaded from the
dataset's GitHub) and are transcoded once to `{FALL_DATA}/cache/web`. A clip without
a video plays as skeleton only. The model itself only ever sees the skeleton.

## Webcam demo (09_live.py recording)

```powershell
.venv\Scripts\python.exe scripts\09_live.py --run e9_ens_gru --camera 0 --t-low 0.6 --record
```

The live window stays skeleton-only; `--record` also saves the camera video, the
skeleton and the scores and alerts the live script decided, to
`{FALL_DATA}/recordings/<date-time>`. On the presenter page, CAM 04 → "webcam
recording" plays the newest session without a restart: that video, its skeleton,
and the live script's own alerts, which go through incidents and phone pushes as usual.

| Page | Address | Who |
| --- | --- | --- |
| Monitoring wall | `http://localhost:8000` | the laptop; press "Start monitoring" to allow sound |
| Carer phone | `http://<laptop LAN IP>:8000/phone` (printed at start) | carers on the same Wi-Fi/hotspot; PIN in `configs/server.yaml` |
| Presenter | `http://localhost:8000/presenter` | demo controls: play a fall, cut a camera, reset |
| Shift log | `http://localhost:8000/log` | handover summary and CSV export |

For front-end work with hot reload: run the server, then `cd frontend; npm run dev` (proxies `/api` and `/ws`).

## Phone alerts (ntfy)

1. Each carer installs the free **ntfy** app and subscribes to a long random topic, e.g. `fallsense-amara-7f3k9q2x`.
2. Put the topics in `configs/server.local.yaml` (gitignored):

   ```yaml
   carers:
     Amara Okafor: {topic: fallsense-amara-7f3k9q2x}
   ```

3. Restart the server; "Send me a test alert" on the phone page checks it.

Without topics everything still runs and pushes are logged as a dry run.

## Phones on any network (Cloudflare tunnel)

```powershell
winget install --id Cloudflare.cloudflared --exact   # once
.venv\Scripts\python.exe scripts\11_serve.py --tunnel
```

A few seconds after start the log, and the wall's Activity list, show
`https://<words>.trycloudflare.com/phone`. Phones open that from mobile data or any
Wi-Fi; the laptop only needs internet. The address is new on every start and goes into
every notification automatically. Without `--tunnel`, phones must be on the laptop's
network (`public_url: auto`).

## Deployed: one address for everything

Day to day, start it with `.\scripts\start.ps1` (add `-Build` after pulling changes).
It sets the `FALL_DATA` default, prepares the videos, builds the web app if needed and
starts the server and tunnel. Keep the laptop plugged in and stop it sleeping.

Set in `configs/server.local.yaml`: `tunnel: true`, `trust_localhost: false`, your own
`pins`, and for a fixed address a named tunnel (`tunnel_token` from Cloudflare Zero
Trust → Networks → Tunnels, public hostname → `http://localhost:8000`, plus
`public_url: https://fallsense.<your-domain>`). Then every page is used through that
address, and each signs in with its role's PIN:

| Page | Sign in as | Can |
| --- | --- | --- |
| `/phone` | a carer | see incidents, respond |
| `/` (wall), `/log` | Nurse station | see everything, respond, carer-in-room, export |
| `/presenter` | Presenter | demo controls (play a fall, cut a camera, reset) |

Every read and action carries the signed-in name and token, wrong PINs are limited to
5 per 10 minutes per address, and tunnel traffic is never treated as the laptop.
Stop the server with Ctrl+C so the tunnel closes with it.

## Configs

- `configs/rooms.yaml` – cameras, their replay clips and presenter scenarios (`scripts/12_demo_clips.py --run e9_ens_gru` suggests clips and says which have video).
- `configs/carers.yaml` – carers and tiers (1 room carers, 2 everyone on shift, 3 senior).
- `configs/server.yaml` – model run, escalation timings, PIN, public URL.

The model run is a config value: when the backend freezes a better model, change `model_run`.
