# Running fallsense

Commands to run the site on AWS, run it on this laptop, and run the live webcam demo.
Run everything from the repo root in PowerShell (Windows PowerShell 5.1 or PowerShell 7).

**Only one server may hold the tunnel at a time.** AWS and the laptop both serve
https://fall-safe.bid through the same Cloudflare tunnel. If both run, Cloudflare splits
visitors between them. Stop one before you start the other (see [Switching](#switching-between-aws-and-the-laptop)).

---

## Quick reference: the public site (fall-safe.bid)

How it is set up: https://fall-safe.bid is a Cloudflare domain whose tunnel leads to one
AWS EC2 instance (`t3.small`, eu-west-2, AWS profile `fallsense`) running the site in
public demo mode. The instance stops itself 15 minutes after the last visitor leaves
(or after 3 hours up). While it is stopped, the Cloudflare Worker `fallsense-wake`
shows visitors a **Wake the demo** button that starts it again. Idle cost: about
$1.50/month (the disk); about $0.03/hour while running. The domain renews yearly in
Cloudflare.

**Sign in first** (each session expires after a while):

```powershell
aws login --profile fallsense                          # AWS
cd scripts\aws\wake; npx wrangler login; cd ..\..\..    # Cloudflare (only for `wake`)
```

| I want to… | Command |
| --- | --- |
| see if it is running | `.\scripts\aws.ps1 status` |
| put new code on the site | `.\scripts\aws.ps1 deploy -Build` (`-Build` rebuilds the web app; leave it out if only Python changed) |
| start it / stop it now | `.\scripts\aws.ps1 up` / `.\scripts\aws.ps1 down` (or let it stop itself) |
| read the server logs | `.\scripts\aws.ps1 logs` (`-Follow` to stream) |
| get a shell on it | `.\scripts\aws.ps1 ssh` |
| keep it awake while I work on it | in `ssh`: `touch /opt/fallsense/state/keep-awake`; `rm` it when done |
| set up the wake page (once) | `cd scripts\aws\wake; npm install; npx wrangler login; cd ..\..\..` then `.\scripts\aws.ps1 wake` |
| test the wake page | `.\scripts\aws.ps1 down`, open https://fall-safe.bid, press **Wake the demo** |
| serve from this laptop instead | `.\scripts\aws.ps1 down`, then `.\scripts\start.ps1` (see [Switching](#switching-between-aws-and-the-laptop)) |
| turn demo mode off (PINs, real phone pushes) | `public_demo: false` in `configs\server.local.yaml`, then `deploy` |

**Shut it all down for good** (stops every AWS and Cloudflare charge except the domain):

1. `.\scripts\aws.ps1 destroy`: deletes the instance and its disk, the SSH endpoint,
   the S3 bucket, the firewall group and the key pair (it asks you to type `yes`).
2. AWS console, **IAM > Users**: delete `fallsense-wake` (made by `wake`).
3. Cloudflare dashboard, **Workers & Pages**: delete `fallsense-wake`. Without it, the
   site shows a Cloudflare error while nothing serves it.
4. Optionally, Cloudflare, **Zero Trust > Networks > Tunnels**: delete the tunnel, and
   **Domain Registration**: turn off auto-renew for fall-safe.bid.

To bring it back after `destroy`, run `.\scripts\aws.ps1 create` and then
`.\scripts\aws.ps1 wake` again (the instance ID changes, so the Worker needs the new one).

**When something goes wrong:**

| Symptom | Fix |
| --- | --- |
| a command says the session expired / no credentials | `aws login --profile fallsense` |
| `deploy` or `up` hangs for minutes on SSH | the AWS Instance Connect link stalled. Ctrl+C and run it again (or end the stuck `ssh.exe` in Task Manager; the script retries) |
| site shows a Cloudflare 502 error | the laptop's Cloudflared service is also on the tunnel: in an Administrator PowerShell, `Stop-Service Cloudflared; Set-Service Cloudflared -StartupType Manual` |
| site shows Cloudflare error 1033 | nothing is serving: `.\scripts\aws.ps1 up` (or set up the wake page) |
| wake page says "Could not wake it" | wait a minute and press again (a fresh AWS key takes a moment); if it persists, run `.\scripts\aws.ps1 wake` again |
| page loads but shows old code | `.\scripts\aws.ps1 deploy -Build`, then Ctrl+F5 |

---

## 1. AWS server (`scripts/aws.ps1`)

One `t3.small` EC2 instance in the default VPC (region from `aws configure`, else `eu-west-2`).
No port is open to the internet: SSH goes through an EC2 Instance Connect Endpoint and the
web traffic through the Cloudflare tunnel.

### Before the first run

- The AWS CLI is signed in: `aws login --profile fallsense` (the script uses that profile
  when it exists; `AWS_PROFILE` overrides it).
- Windows OpenSSH (`ssh`, `scp`) is on PATH.
- `configs/server.local.yaml` holds the tunnel token, and `FALL_DATA` points at the data
  (default `C:\fall-data`).
- The laptop is not serving: no `start.ps1` running, and the Cloudflared Windows service is stopped:
  ```powershell
  # in an Administrator PowerShell, once
  Stop-Service Cloudflared
  Set-Service Cloudflared -StartupType Manual
  ```

### Commands

```powershell
.\scripts\aws.ps1 create            # once: key pair, firewall, SSH endpoint, instance, first deploy
.\scripts\aws.ps1 deploy            # push this checkout + demo data to the instance
.\scripts\aws.ps1 deploy -Build     # rebuild the web app (frontend/dist) first, then deploy
.\scripts\aws.ps1 up                # start the stopped instance (the server starts with it)
.\scripts\aws.ps1 down              # stop the instance (only the disk is billed, ~$1.50/month)

.\scripts\aws.ps1 status            # instance state + last 5 server log lines
.\scripts\aws.ps1 logs              # last 200 log lines
.\scripts\aws.ps1 logs -Follow      # stream the logs (Ctrl+C to stop)
.\scripts\aws.ps1 preview           # the instance's wall at http://localhost:8001 over SSH
.\scripts\aws.ps1 ssh               # shell on the instance (service: fallsense, app in /opt/fallsense)

.\scripts\aws.ps1 destroy           # delete instance, disk (incident log), endpoint, bucket, firewall, key
```

`up` and `deploy` wait until the server answers and print `Live: https://...` once the
tunnel is ready. They refuse to run while something on the laptop listens on port 8000 or
the Cloudflared service is running.

Running costs about $0.03/hour. You don't need to run `down`: the instance stops itself
once nobody has had the site open for 15 minutes, or after 3 hours up in any case
(`scripts/aws/idle-stop.sh`, a systemd timer installed by `deploy`). While stopped, only
the disk is billed (~$1.50/month).

### Wake on visit (`.\scripts\aws.ps1 wake`, once)

A Cloudflare Worker (`scripts/aws/wake/worker.js`) sits in front of https://fall-safe.bid.
While the site is up it passes every request through. While the instance is stopped it
shows a "The demo is asleep" page with a **Wake the demo** button. The button starts the
instance and the page reloads into the demo when it is ready, about a minute later.
Visitors can only wake it by pressing the button, so crawlers never start it.

```powershell
cd scripts\aws\wake; npm install; npx wrangler login; cd ..\..\..   # once: sign Wrangler in to Cloudflare
.\scripts\aws.ps1 wake      # IAM user fallsense-wake (may only start this instance), then deploy the Worker
```

Run `wake` again after `destroy` + `create`, because the instance ID changes. Each run
replaces the Worker's AWS key. To undo it, delete the `fallsense-wake` Worker in the
Cloudflare dashboard and the `fallsense-wake` IAM user.

To keep the instance up while you work on it: `.\scripts\aws.ps1 ssh`, then
`touch /opt/fallsense/state/keep-awake` (`rm` it when done).

### Public demo mode (`public_demo: true`)

With `public_demo: true` in `configs/server.local.yaml`, the site is a self-serve demo:

- nobody signs in (no PINs);
- the wall gets a **Try it** bar to replay the fall scenarios, unplug a camera and reset;
  `/presenter` still works;
- the carer's phone (the real `/phone` page) sits next to the wall. It is a column on
  wide screens and behind **Show the carer's phone** otherwise;
- alerts are shown on that phone as an ntfy-style banner. **ntfy is never called**, so no
  real phone is notified, whatever topics are set;
- every start (each wake-up) begins on a quiet shift.

All visitors share one demo, so one visitor's Reset clears everyone's incidents. Set
`public_demo: false` (the default in `server.yaml`) for a real shift with PINs and phone
pushes.

### Trying the instance while the laptop serves the site

```powershell
.\scripts\aws.ps1 deploy -NoTunnel  # deploy without the tunnel (skips the laptop check)
.\scripts\aws.ps1 up -NoTunnel      # start it without the tunnel
.\scripts\aws.ps1 preview           # look at it on http://localhost:8001
```

Run `deploy` again without `-NoTunnel` when the instance should take over the public address.

---

## 2. Local server on this laptop (`scripts/start.ps1`)

### Start

```powershell
.\scripts\aws.ps1 down              # first, if the AWS instance is running
.\scripts\start.ps1                 # wall, presenter, log and phones at https://fall-safe.bid
.\scripts\start.ps1 -Build          # rebuild the web app first (after pulling changes)
```

`start.ps1` fetches and transcodes any missing demo videos, builds `frontend/dist` if it is
missing, then runs `scripts\11_serve.py` with the tunnel from `configs/server.local.yaml`.
It refuses to start if port 8000 is already in use.

You can also start the server without the tunnel (local network only):

```powershell
.venv\Scripts\activate
python scripts\11_serve.py                  # http://localhost:8000; phones must be on the same network
python scripts\11_serve.py --port 8080
```

### Stop

Press **Ctrl+C** in the window running `start.ps1`, so the tunnel closes along with the server.

If the window has been closed and port 8000 is still in use:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
Stop-Process -Id <OwningProcess>
```

---

## Switching between AWS and the laptop

| From → to     | Commands                                                                 |
|---------------|--------------------------------------------------------------------------|
| Laptop → AWS  | Ctrl+C in `start.ps1`, then `.\scripts\aws.ps1 up` (or `deploy` for new code) |
| AWS → Laptop  | `.\scripts\aws.ps1 down` (wait for "Stopped."), then `.\scripts\start.ps1` |

`.\scripts\aws.ps1 status` warns you if the instance and the laptop's Cloudflared
service are both serving.

---

## 3. Live webcam demo (`scripts/live.ps1`)

Runs the frozen model (`e9_ens_gru`) on a webcam at about 15 fps on CPU. The window shows
only the skeleton on black, never the camera image, with a banner across the top:

| Banner     | Meaning                                         |
|------------|-------------------------------------------------|
| `OK`       | green: no alert                                 |
| `CHECK`    | amber: possible fall, worth a look              |
| `FALL`     | red: fall detected (with its score)             |
| `ON FLOOR` | purple: still down 20 s after the fall          |

Bars at the bottom show the model's upright / falling / fallen probabilities.
An alert stays on screen for 6 s.

**Keys (click the window first):** **q** or **Esc** quits, **r** resets the alert state.
Closing the window or pressing Ctrl+C in the terminal also stops it.

```powershell
.\scripts\live.ps1                  # webcam 0 (the Poly camera that sees the room)
.\scripts\live.ps1 -List            # which cameras work and whether they see a person
.\scripts\live.ps1 -Camera 1        # another camera
.\scripts\live.ps1 -Lite            # lighter pose model, if the fps drops under 10
.\scripts\live.ps1 -Record          # also save the session to %FALL_DATA%\recordings\<date-time>
.\scripts\live.ps1 -Video clip.mp4  # replay a video file instead of the webcam
.\scripts\live.ps1 -TLow 0.3        # more CHECK alerts (the default is 0.6)
```

Only one program can use the camera at a time, so close Teams, Zoom or the Camera app
first. The camera must see the whole body; if it can't, the window says
"full body not in view - step back".

### Calling `09_live.py` directly

`live.ps1` is a wrapper around `scripts/09_live.py`, which takes the full set of options:

```powershell
.venv\Scripts\activate
python scripts\09_live.py --list-cameras                     # find the camera that sees the room
python scripts\09_live.py --run e9_ens_gru --camera 0        # live detection with the run's thresholds
python scripts\09_live.py --run e9_ens_gru --pose lite       # if the frame rate drops under 10 fps
python scripts\09_live.py --run e9_ens_gru --camera 0 --t-low 0.6   # override the alert threshold
python scripts\09_live.py --run e9_ens_gru --video clip.mp4 --no-display   # replay a video file
```

Other options: `--width/--height` (default 640x480), `--t-high`, `--on-floor <seconds>`,
`--seconds <n>` (stop after n seconds), `--record [DIR]` (save to a folder you choose).

### Recording a session for the monitoring wall

```powershell
.\scripts\live.ps1 -Record                                                         # -> %FALL_DATA%\recordings\<date-time>
python scripts\09_live.py --run e9_ens_gru --camera 0 --record logs\live\dry_run_1  # -> a folder you choose
```

A recording contains `video.mp4`, `skeleton.npz`, `scores.csv`, `alerts.csv` and `session.yaml`.

To play a recording on the wall, point a room in `configs/rooms.yaml` at its folder name.
CAM 06 plays `20261018-071944` like this:

```yaml
    source: recording
    idle:
      - {recording: 20261018-071944, end_s: 30}
    scenarios:
      fall: {recording: 20261018-071944, end_s: 34, hold_s: 20}
```

`{recording: latest}` picks the newest session instead. Then restart the server:

- Laptop: Ctrl+C, then `.\scripts\start.ps1`
- AWS: `.\scripts\aws.ps1 deploy`. The bundle includes only the recordings that `rooms.yaml` plays.
