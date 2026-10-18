"""Phone alerts through ntfy (https://ntfy.sh): no app of our own, no app store.

Each carer installs the free ntfy app and subscribes to their private topic
(configs/server.local.yaml). Urgent incidents carry two buttons: "I'm going" posts
straight back to this server, "Open" opens the incident page. A carer without a
topic is skipped and the push is logged as a dry run, so the system runs without
phones too.
"""

import hashlib
import hmac
import logging
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from fallsense_server.incidents import Push
from fallsense_server.settings import Settings

log = logging.getLogger("fallsense.notify")
TAGS = {5: ["rotating_light"], 4: ["warning"], 3: ["eyes"], 2: ["white_check_mark"]}


class Tokens:
    """Per-carer link tokens: HMAC of the name with a secret kept in server_data/."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(secrets.token_hex(32), encoding="utf-8")
        self.key = path.read_text(encoding="utf-8").strip().encode()

    def token(self, carer: str) -> str:
        return hmac.new(self.key, carer.encode(), hashlib.sha256).hexdigest()[:24]

    def check(self, carer: str, token: str) -> bool:
        return hmac.compare_digest(self.token(carer), token or "")


class Notifier:
    def __init__(self, settings: Settings, tokens: Tokens) -> None:
        self.settings, self.tokens = settings, tokens
        self.server = str(settings.raw["ntfy"]["server"]).rstrip("/")
        self.client = httpx.AsyncClient(timeout=10)

    def payload(self, push: Push, carer: str, topic: str) -> dict[str, Any]:
        base = self.settings.public_url()
        # encoded: a raw space in "Amara Okafor" cuts the link short on Android
        auth = urlencode({"carer": carer, "token": self.tokens.token(carer)})
        body: dict[str, Any] = {
            "topic": topic,
            "title": push.title,
            "message": push.message,
            "priority": push.priority,
            "tags": TAGS.get(push.priority, []),
        }
        if push.incident_id:
            page = f"{base}/phone/i/{push.incident_id}?{auth}"
            body["click"] = page
            if push.actions:
                # "I'm going" opens the page, which acknowledges on load. A background
                # HTTP action fails on the phone hosting the hotspot: ntfy sends it over
                # mobile data, which cannot reach the laptop; the browser can.
                body["actions"] = [
                    {
                        "action": "view",
                        "label": "I'm going",
                        "url": f"{page}&ack=1",
                        "clear": True,
                    },
                    {"action": "view", "label": "Open", "url": page},
                ]
        return body

    async def send(self, push: Push) -> list[tuple[str, str]]:
        """Send to every carer in the push; returns (carer, "sent" | "dry run" | error)."""
        out = []
        for name in push.carers:
            carer = self.settings.carer(name)
            if carer is None or not carer.topic:
                out.append((name, "dry run"))
                continue
            try:
                r = await self.client.post(
                    self.server, json=self.payload(push, name, carer.topic)
                )
                r.raise_for_status()
                out.append((name, "sent"))
            except httpx.HTTPError as e:
                log.warning("push to %s failed: %s", name, e)
                out.append((name, f"failed: {e.__class__.__name__}"))
        return out

    async def close(self) -> None:
        await self.client.aclose()
