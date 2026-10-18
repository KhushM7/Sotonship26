"""Start the monitoring wall, the carer phone pages and the alert service.

    python scripts/11_serve.py                 # http://localhost:8000
    python scripts/11_serve.py --tunnel        # phones on any network (Cloudflare)
    python scripts/11_serve.py --port 8080

Needs FALL_DATA (skeletons, index.csv, videos). Models and scores come from this
repo (FALLSENSE_BACKEND overrides). Build the web app first:
cd frontend; npm install; npm run build. Without --tunnel, phones must be on the
laptop's network and open the "Phones" address printed below; with it, the
tunnel address is printed (and shown on the wall) a few seconds after start.
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fallsense_server.app import create_app
from fallsense_server.settings import load_settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets phones connect")
    ap.add_argument(
        "--tunnel",
        action=argparse.BooleanOptionalAction,
        help="public HTTPS address via cloudflared (default: `tunnel` in server.yaml)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = load_settings()
    if args.port:
        settings.raw["port"] = args.port
    if args.tunnel is not None:
        settings.raw["tunnel"] = args.tunnel
    port = int(settings.raw["port"])
    pushing = [c.name for c in settings.carers if c.topic]
    print(f"Wall:   http://localhost:{port}")
    if settings.raw.get("tunnel"):
        print("Phones: starting a Cloudflare tunnel; its address is logged when ready")
    else:
        print(f"Phones: {settings.public_url()}/phone")
    print(
        f"Push:   {', '.join(pushing) if pushing else 'no ntfy topics set (dry run)'}"
    )
    uvicorn.run(create_app(settings), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
