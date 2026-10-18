"""A Cloudflare quick tunnel: a public HTTPS address for this laptop, on any network.

`cloudflared tunnel --url http://localhost:<port>` prints a random
https://<words>.trycloudflare.com address; phones use it from mobile data or any
Wi-Fi, so nothing depends on a shared network or the laptop's IP. The address
changes on every start, which is fine: the server reads it here and puts it in
every notification. No Cloudflare account is needed.
"""

import logging
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger("fallsense.tunnel")
URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
WINDOWS_DEFAULT = Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe")


def find_cloudflared(configured: str | None = None) -> str | None:
    for candidate in (configured, shutil.which("cloudflared"), str(WINDOWS_DEFAULT)):
        if candidate and Path(candidate).exists():
            return candidate
    return None


class Tunnel:
    """Runs cloudflared in the background; calls `on_url` once the address is known."""

    def __init__(
        self,
        exe: str,
        port: int,
        on_url: Callable[[str], None],
        on_exit: Callable[[], None],
    ) -> None:
        self.exe, self.port, self.on_url, self.on_exit = exe, port, on_url, on_exit
        self.proc: subprocess.Popen[str] | None = None
        self.url: str | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [
                self.exe,
                "tunnel",
                "--no-autoupdate",
                "--url",
                f"http://localhost:{self.port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # cloudflared logs, including the address, to stderr
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._watch, name="cloudflared", daemon=True).start()

    def _watch(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            if self.url is None and (m := URL.search(line)):
                self.url = m.group(0)
                log.info("tunnel ready: %s", self.url)
                self.on_url(self.url)
            elif " ERR " in line:
                log.warning("cloudflared: %s", line.strip())
        log.warning("cloudflared exited (code %s)", self.proc.wait())
        self.url = None
        self.on_exit()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
