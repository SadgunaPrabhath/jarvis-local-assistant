"""JARVIS HUD backend — keyless FastAPI service.

Exposes project status/control, system telemetry and the Ollama-powered chat
brain to the holographic web HUD.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import brain
import launcher
from projects import PROJECT_MAP, snapshot

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="JARVIS", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only HUD
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class LaunchRequest(BaseModel):
    id: str


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": "JARVIS", "model": brain.detect_model()}


@app.get("/api/projects")
def projects() -> dict:
    return {"projects": snapshot()}


@app.get("/api/system")
def system() -> dict:
    return brain._system_stats()


@app.post("/api/launch")
def launch(req: LaunchRequest) -> dict:
    return launcher.ensure_running(req.id)


@app.post("/api/open")
def open_project(req: LaunchRequest) -> dict:
    result = launcher.ensure_running(req.id)
    project = PROJECT_MAP.get(req.id)
    if project:
        result.setdefault("url", project.url)
    return result


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    history = [{"role": m.role, "content": m.content} for m in req.history]
    return brain.chat(req.message, history)


# ---------------------------------------------------------------------------
# Serve the holographic HUD (single-file, no build step) from this same server.
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
