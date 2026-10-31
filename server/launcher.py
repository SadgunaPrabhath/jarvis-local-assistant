"""Launch a project's dev server, detached so it survives this process.

On this machine, dev servers started as ordinary child processes get reaped
shortly after launch, and only stay alive when spawned via PowerShell
Start-Process with their stdout/stderr redirected to files. This module
encapsulates that trick.
"""
from __future__ import annotations

import os
import subprocess
import time

from projects import Project, PROJECT_MAP, is_up


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def launch(project: Project) -> None:
    log_dir = os.path.join(project.directory, "_jarvis")
    os.makedirs(log_dir, exist_ok=True)
    out_log = os.path.join(log_dir, f"{project.id}.out.log")
    err_log = os.path.join(log_dir, f"{project.id}.err.log")

    arg_list = ",".join(_ps_quote(a) for a in project.args)
    ps = (
        f"Start-Process -FilePath {_ps_quote(project.program)} "
        f"-ArgumentList @({arg_list}) "
        f"-WorkingDirectory {_ps_quote(project.directory)} "
        f"-WindowStyle Hidden "
        f"-RedirectStandardOutput {_ps_quote(out_log)} "
        f"-RedirectStandardError {_ps_quote(err_log)}"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_running(project_id: str, wait_seconds: float = 12.0) -> dict:
    """Launch the project if it isn't already up, then wait until it responds."""
    project = PROJECT_MAP.get(project_id)
    if not project:
        return {"ok": False, "error": f"Unknown project '{project_id}'."}

    if is_up(project.port):
        return {"ok": True, "already_running": True, "url": project.url, "name": project.name}

    try:
        launch(project)
    except subprocess.CalledProcessError as exc:
        return {"ok": False, "error": (exc.stderr or str(exc)).strip()}

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_up(project.port):
            return {"ok": True, "launched": True, "url": project.url, "name": project.name}
        time.sleep(0.6)

    # Booted but not yet answering — still report the URL; dev servers can be slow.
    return {"ok": True, "launched": True, "slow": True, "url": project.url, "name": project.name}
