"""Registry of the user's projects JARVIS can see, launch and open.

Kept in one place so the chat tools, the status probe and the launcher all
agree on paths, ports and how each project is started.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field

HOME = os.path.expanduser("~")


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    emoji: str
    directory: str
    port: int
    # PowerShell-friendly launch: program + args. Vite/Next dev servers get
    # launched detached (see launcher.py) because they die otherwise.
    program: str
    args: list[str] = field(default_factory=list)
    kind: str = "web"  # web | api
    blurb: str = ""

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"


PROJECTS: list[Project] = [
    Project(
        "life-sim", "LIFE//SIM", "🎮",
        rf"{HOME}\life-sim", 5179,
        "npm.cmd", ["run", "dev", "--", "--port", "5179", "--strictPort"],
        blurb="AI life-simulation RPG with a local-LLM Dungeon Master.",
    ),
    Project(
        "four-realms", "The Four Realms", "⚔️",
        rf"{HOME}\four-realms", 5174,
        "npm.cmd", ["run", "dev", "--", "--port", "5174", "--strictPort"],
        blurb="Vanilla-JS narrative game across four worlds.",
    ),
    Project(
        "synapse-web", "SYNAPSE", "🧠",
        rf"{HOME}\synapse\apps\web", 5173,
        "npm.cmd", ["run", "dev"],
        blurb="Multi-agent software-engineering platform (frontend).",
    ),
    Project(
        "synapse-api", "SYNAPSE API", "🔌",
        rf"{HOME}\synapse\apps\api", 8000,
        rf"{HOME}\synapse\apps\api\.venv\Scripts\python.exe",
        ["-m", "uvicorn", "synapse.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        kind="api",
        blurb="SYNAPSE FastAPI backend.",
    ),
]

PROJECT_MAP: dict[str, Project] = {p.id: p for p in PROJECTS}


def is_up(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    """True if something is listening on the port (IPv4 or IPv6)."""
    for family, addr in ((socket.AF_INET, host), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((addr, port)) == 0:
                    return True
        except OSError:
            continue
    return False


def snapshot() -> list[dict]:
    """Current status of every project, for the HUD and the chat tools."""
    out = []
    for p in PROJECTS:
        out.append(
            {
                "id": p.id,
                "name": p.name,
                "emoji": p.emoji,
                "url": p.url,
                "port": p.port,
                "kind": p.kind,
                "blurb": p.blurb,
                "running": is_up(p.port),
            }
        )
    return out

