"""Start the monitoring wall, the carer phone pages and the alert service.

    python scripts/11_serve.py                 # http://localhost:8000
    python scripts/11_serve.py --port 8080

Needs FALL_DATA (skeletons, index.csv) and FALLSENSE_BACKEND (the checkout with
models/ and reports/scores/; defaults to this repo). Build the web app first:
cd frontend; npm install; npm run build. Phones on the same network open the
"Phones" address printed below.
"""

import argparse
import logging

import uvicorn

from fallsense_server.app import create_app
from fallsense_server.settings import load_settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets phones connect")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = load_settings()
    if args.port:
        settings.raw["port"] = args.port
    port = int(settings.raw["port"])
    pushing = [c.name for c in settings.carers if c.topic]
    print(f"Wall:   http://localhost:{port}")
    print(f"Phones: {settings.public_url()}/phone")
    print(
        f"Push:   {', '.join(pushing) if pushing else 'no ntfy topics set (dry run)'}"
    )
    uvicorn.run(create_app(settings), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
