"""Fan-out of Contract 6 messages to every connected browser."""

import asyncio
import json
from typing import Any

# Contract 6 message types; nothing else is ever sent, and none carries pixels
MESSAGE_TYPES = {"hello", "frame", "scores", "alert", "incident", "event", "status"}


class Client:
    def __init__(self, frames: str = "all") -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        self.subscribe(frames)

    def subscribe(self, frames: str) -> None:
        """ "all", "none" or a comma list of room ids whose skeleton frames to receive."""
        self.frames: set[str] | None = (
            None
            if frames == "all"
            else {r for r in frames.split(",") if r and r != "none"}
        )

    def wants(self, msg: dict[str, Any]) -> bool:
        if msg["type"] != "frame":
            return True
        return self.frames is None or msg["room"] in self.frames


class Hub:
    def __init__(self) -> None:
        self.clients: set[Client] = set()

    def publish(self, msg: dict[str, Any]) -> None:
        if msg["type"] not in MESSAGE_TYPES:
            raise ValueError(f"not a Contract 6 message: {msg['type']}")
        text = json.dumps(msg, separators=(",", ":"))
        for c in self.clients:
            if not c.wants(msg):
                continue
            try:
                c.queue.put_nowait(text)
            except asyncio.QueueFull:
                # a stalled browser loses skeleton frames, never incidents
                if msg["type"] != "frame":
                    _drop_one(c.queue)
                    c.queue.put_nowait(text)


def _drop_one(q: asyncio.Queue[str]) -> None:
    try:
        q.get_nowait()
    except asyncio.QueueEmpty:
        pass
