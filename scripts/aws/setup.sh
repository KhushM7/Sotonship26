#!/usr/bin/env bash
# Runs on the AWS instance (scripts/aws.ps1 deploy): installs or updates fallsense from
# the bundle made by scripts/aws/bundle.py, then (re)starts it as a systemd service
# that also starts on every boot. Safe to run again; incidents (state/) are kept.
#
#   bash setup.sh /tmp/fallsense.tar.gz [--no-tunnel]
set -euo pipefail

BUNDLE="$1"
SERVE_ARGS="${2:-}"  # passed to 11_serve.py; --no-tunnel while the laptop has the site
ROOT=/opt/fallsense
APT="sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o DPkg::Lock::Timeout=600"

# first boot: let cloud-init finish its own apt work first
cloud-init status --wait >/dev/null 2>&1 || true

if ! command -v cloudflared >/dev/null || ! command -v ffmpeg >/dev/null; then
    echo "== system packages"
    $APT update -q
    $APT install -q ffmpeg curl
    curl -fsSL -o /tmp/cloudflared.deb \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i /tmp/cloudflared.deb
fi
if ! command -v uv >/dev/null; then
    echo "== uv"
    curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi

echo "== unpack"
sudo mkdir -p "$ROOT/state"
sudo chown -R "$USER:" "$ROOT"
rm -rf "$ROOT/new"
mkdir -p "$ROOT/new"
tar -xzf "$BUNDLE" -C "$ROOT/new"
# keep the instance's own secret if it has one, else take the laptop's
if [ -f "$ROOT/new/state/secret.key" ] && [ ! -f "$ROOT/state/secret.key" ]; then
    cp "$ROOT/new/state/secret.key" "$ROOT/state/secret.key"
fi
ln -sfn "$ROOT/state" "$ROOT/new/app/server_data"
sudo systemctl stop fallsense 2>/dev/null || true
rm -rf "$ROOT/app" "$ROOT/data"
mv "$ROOT/new/app" "$ROOT/app"
mv "$ROOT/new/data" "$ROOT/data"
rm -rf "$ROOT/new"

echo "== python"
export UV_CACHE_DIR="$ROOT/.uv-cache"
[ -d "$ROOT/venv" ] || uv venv -q --python 3.11 "$ROOT/venv"
VIRTUAL_ENV="$ROOT/venv" uv pip install -q -r "$ROOT/app/scripts/aws/requirements.txt"
VIRTUAL_ENV="$ROOT/venv" uv pip install -q torch==2.14.1 \
    --index-url https://download.pytorch.org/whl/cpu

echo "== service"
sudo tee /etc/systemd/system/fallsense.service >/dev/null <<EOF
[Unit]
Description=fallsense server (wall, phones, alerts) with its Cloudflare tunnel
Wants=network-online.target
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$ROOT/app
Environment=FALL_DATA=$ROOT/data
Environment=PYTHONUNBUFFERED=1
ExecStart=$ROOT/venv/bin/python scripts/11_serve.py $SERVE_ARGS
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable -q fallsense
sudo systemctl restart fallsense
echo "== started (logs: journalctl -u fallsense -f)"
