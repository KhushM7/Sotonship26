# fallsense – monitoring wall and carer alerts

The web app and server described in `docs/Frontend Plan (Fall Detection).md`. Only what
happens *after* the backend decides: incidents, phone alerts, escalation, the wall and
the demo. It reads the backend's outputs and never re-implements them.

## Run it

```powershell
# once: a venv for the server that borrows torch/numpy/pandas from the backend venv
uv venv .venv --python <backend checkout>\.venv\Scripts\python.exe
# add a .pth in .venv\Lib\site-packages listing this repo's src and the backend venv's site-packages
uv pip install --link-mode=copy --python .venv\Scripts\python.exe fastapi "uvicorn[standard]" httpx

# once: build the web app
cd frontend; npm install; npm run build; cd ..

# every time
$env:FALL_DATA = "C:\fall-data"                          # skeletons + index.csv
$env:FALLSENSE_BACKEND = "<backend checkout>"            # models/ and reports/scores/
.venv\Scripts\python.exe scripts\11_serve.py
```

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

Without topics everything still runs and pushes are logged as a dry run. The notification's
"I'm going" button posts back to the laptop, so the phone must reach it (same hotspot,
or set `public_url` to a tunnel and `trust_localhost: false`).

## Configs

- `configs/rooms.yaml` – cameras, their replay clips and presenter scenarios (`scripts/12_demo_clips.py` suggests clips).
- `configs/carers.yaml` – carers and tiers (1 room carers, 2 everyone on shift, 3 senior).
- `configs/server.yaml` – model run, escalation timings, PIN, public URL.

The model run is a config value: when the backend freezes a better model, change `model_run`.
