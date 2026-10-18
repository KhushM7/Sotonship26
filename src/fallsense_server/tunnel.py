"""A Cloudflare tunnel: a public HTTPS address for this laptop, on any network.

Quick tunnel (no account): `cloudflared tunnel --url http://localhost:<port>` prints a
random https://<words>.trycloudflare.com address, new on every start; the server
reads it here and puts it in every notification.

Named tunnel (free account + your domain): `cloudflared tunnel run` with the token
from the Cloudflare dashboard (passed in TUNNEL_TOKEN, not on the command line). The
address is your fixed hostname, routed to localhost:<port> in the dashboard; it is
ready once cloudflared registers its first connection.
"""

import logging
import os
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
        token: str | None = None,
        fixed_url: str | None = None,
    ) -> None:
        self.exe, self.port, self.on_url, self.on_exit = exe, port, on_url, on_exit
        self.token, self.fixed_url = token, fixed_url
        self.proc: subprocess.Popen[str] | None = None
        self.url: str | None = None

    def start(self) -> None:
        env = dict(os.environ)
        if self.token:
            args = ["tunnel", "--no-autoupdate", "run"]
            env["TUNNEL_TOKEN"] = self.token
        else:
            args = [
                "tunnel",
                "--no-autoupdate",
                "--url",
                f"http://localhost:{self.port}",
            ]
        self.proc = subprocess.Popen(
            [self.exe, *args],
            env=env,
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
            if self.url is None:
                if self.token and "Registered tunnel connection" in line:
                    self.url = self.fixed_url
                elif not self.token and (m := URL.search(line)):
                    self.url = m.group(0)
                if self.url:
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
