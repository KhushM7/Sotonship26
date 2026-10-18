"""Server, carer and room config: configs/*.yaml with configs/server.local.yaml on top."""

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
# incidents.db, alerts.jsonl and the token secret; gitignored
STATE_DIR = REPO / "server_data"
FRONTEND_DIST = REPO / "frontend" / "dist"


def backend_root() -> Path:
    """The checkout holding models/ and reports/scores/ (FALLSENSE_BACKEND, else this repo)."""
    return Path(os.environ.get("FALLSENSE_BACKEND", REPO))


def _read(name: str) -> dict[str, Any]:
    path = CONFIGS / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            _merge(out[k], v)
            if isinstance(out.get(k), dict) and isinstance(v, dict)
            else v
        )
    return out


@dataclass
class Carer:
    name: str
    tier: int
    role: str = "Carer"
    topic: str = ""  # ntfy topic; empty = no push for this carer


@dataclass
class Settings:
    raw: dict[str, Any]
    carers: list[Carer]
    rooms: list[dict[str, Any]]
    tunnel_url: str | None = None  # set while a Cloudflare tunnel is running

    @property
    def escalation(self) -> dict[str, Any]:
        return dict(self.raw["escalation"])

    def carer(self, name: str) -> Carer | None:
        return next((c for c in self.carers if c.name == name), None)

    def public_url(self) -> str:
        """Where phones reach this laptop: the tunnel if running, else per public_url."""
        if self.tunnel_url:
            return self.tunnel_url
        url = str(self.raw.get("public_url", "auto"))
        if url != "auto":
            return url.rstrip("/")
        return f"http://{lan_ip()}:{self.raw['port']}"


def lan_ip() -> str:
    """This laptop's address on the current network (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def load_settings() -> Settings:
    local = _read("server.local.yaml")
    raw = _merge(
        _read("server.yaml"), {k: v for k, v in local.items() if k != "carers"}
    )
    local_carers: dict[str, dict[str, Any]] = local.get("carers", {}) or {}
    carers = [
        Carer(**{**c, **local_carers.get(c["name"], {})})
        for c in _read("carers.yaml").get("carers", [])
    ]
    rooms_cfg = _read("rooms.yaml")
    return Settings(
        raw=raw,
        carers=carers,
        rooms=rooms_cfg.get("rooms", []),
    )
