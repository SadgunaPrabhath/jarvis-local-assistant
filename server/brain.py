"""JARVIS's reasoning core: a local Ollama model with tool-calling.

Keyless. Uses Ollama's native tool API so the model can *act* — list the user's
projects, launch them, open them, and read system stats — then narrate the
result in the JARVIS persona.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from urllib.parse import quote

import psutil
import requests

import launcher
from projects import snapshot

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL_PREFERENCE = ["llama3.1", "llama3.2", "qwen2.5", "mistral", "llama3"]

# Small models over-eagerly call tools even for plain questions (popping open a
# stray browser tab). We only expose the action tools when the message actually
# expresses intent to *do* something on the machine; otherwise it's pure chat.
_ACTION_INTENT = re.compile(
    r"\b(open|launch|start|run|go\s?to|goto|visit|play|show|bring up|pull up|"
    r"search|google|look up|check|status|stats|running|projects?|website|site|"
    r"app|close|cpu|memory|ram|battery|system|life[\s-]?sim|four[\s-]?realms|"
    r"bakery|synapse|satwika|youtube|gmail|notepad|calculator)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are JARVIS, the user's personal AI — a fully capable general-purpose "
    "assistant in the style of a calm, witty, impeccably polite British valet "
    "(think Tony Stark's JARVIS). You can do ANYTHING a helpful AI assistant "
    "does: answer questions on any topic, explain things, write and debug code, "
    "draft text and emails, brainstorm ideas, do maths, translate, give advice, "
    "and hold a normal conversation. Always attempt the request directly and "
    "helpfully — never refuse just because it isn't about the user's projects.\n\n"
    "You ALSO have tools that let you act on this machine: control the user's "
    "software projects, open any website, launch desktop apps, and read system "
    "stats. Use a tool ONLY when the user clearly asks you to open, launch, "
    "start, visit, go to, play, or check something. For ordinary questions, "
    "explanations, chat, writing, maths or advice, DO NOT call any tool and DO "
    "NOT open a browser — simply answer from your own knowledge. Never open a "
    "web search just because you were asked a question.\n\n"
    "Style: your replies are read aloud, so keep them conversational and as "
    "concise as the question allows — a sentence or two for simple things, more "
    "when the user genuinely needs detail (an explanation, code, a list). Refer "
    "to the user as 'sir' occasionally, not every line. Be warm, dry-humoured, "
    "and never robotic."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List all of the user's projects and whether each is currently running.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "launch_project",
            "description": "Start a project's dev server if it isn't already running.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Project id, e.g. life-sim, four-realms, bakery, satwika, synapse-web, synapse-api.",
                    }
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_project",
            "description": "Ensure a project is running and open it in the user's browser.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Project id to open."}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_stats",
            "description": "Get current CPU, memory and battery status of the machine.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_website",
            "description": (
                "Open any website in the user's browser. Use for requests like "
                "'open YouTube', 'go to gmail', 'search Google for X', 'open "
                "github.com'. Accepts a full URL or a common site name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {
                        "type": "string",
                        "description": "A URL (https://...) or a site name like 'youtube', 'gmail', 'wikipedia'.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search text; if given, performs a web search for it.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": (
                "Launch a Windows desktop application by name, e.g. notepad, "
                "calculator, paint, explorer, cmd, settings, spotify, vscode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The application name to open."}
                },
                "required": ["name"],
            },
        },
    },
]


def detect_model() -> str | None:
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
        res.raise_for_status()
        models = [m["name"] for m in res.json().get("models", [])]
    except requests.RequestException:
        return None
    if not models:
        return None
    for pref in MODEL_PREFERENCE:
        for m in models:
            if m.lower().startswith(pref):
                return m
    return models[0]


def _system_stats() -> dict:
    battery = None
    try:
        b = psutil.sensors_battery()
        if b is not None:
            battery = {"percent": round(b.percent), "plugged": b.power_plugged}
    except Exception:
        battery = None
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "cpu_count": psutil.cpu_count(logical=True),
        "mem_percent": mem.percent,
        "mem_used_gb": round(mem.used / 1e9, 1),
        "mem_total_gb": round(mem.total / 1e9, 1),
        "battery": battery,
    }


_SITE_ALIASES = {
    "youtube": "https://www.youtube.com",
    "yt": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "mail": "https://mail.google.com",
    "github": "https://github.com",
    "wikipedia": "https://www.wikipedia.org",
    "maps": "https://www.google.com/maps",
    "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
    "whatsapp": "https://web.whatsapp.com",
    "spotify": "https://open.spotify.com",
    "netflix": "https://www.netflix.com",
    "twitter": "https://twitter.com",
    "x": "https://twitter.com",
    "reddit": "https://www.reddit.com",
    "amazon": "https://www.amazon.in",
    "stackoverflow": "https://stackoverflow.com",
}

# Windows apps we can launch. `start`-style names work via os.startfile / cmd.
_APP_ALIASES = {
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "paint": "mspaint",
    "explorer": "explorer",
    "files": "explorer",
    "cmd": "cmd",
    "terminal": "wt",
    "powershell": "powershell",
    "settings": "ms-settings:",
    "task manager": "taskmgr",
    "taskmgr": "taskmgr",
    "control panel": "control",
    "snipping tool": "snippingtool",
    "camera": "microsoft.windows.camera:",
    "spotify": "spotify:",
    "vscode": "code",
    "code": "code",
}


def _resolve_url(site: str, query: str = "") -> str:
    site = (site or "").strip()
    query = (query or "").strip()

    # Models often put the site name in `query` instead of `site` (or vice
    # versa). If `site` is empty but `query` names a known site or a bare
    # domain, promote it so "open youtube" lands on youtube.com, not a search.
    if not site and query and (query.lower() in _SITE_ALIASES or ("." in query and " " not in query)):
        site, query = query, ""

    key = site.lower()
    if key in _SITE_ALIASES:
        base = _SITE_ALIASES[key]
        # A named site plus a search term → search the web for both.
        if query:
            return "https://www.google.com/search?q=" + quote(f"{query} {key}")
        return base
    if site.startswith("http://") or site.startswith("https://"):
        return site
    if "." in site and " " not in site:  # bare domain like github.com
        return "https://" + site

    # Anything left is free text → web search.
    term = (site + " " + query).strip()
    return "https://www.google.com/search?q=" + quote(term)


def _open_app(name: str) -> dict:
    raw = (name or "").strip()
    key = raw.lower()
    target = _APP_ALIASES.get(key, raw)
    try:
        if target.endswith(":") or target.startswith("ms-"):
            # URI-protocol app (settings:, spotify:, camera:) — hand to the shell.
            os.startfile(target)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["cmd", "/c", "start", "", target], shell=False)
        return {"ok": True, "opened": raw or target}
    except Exception as exc:  # noqa: BLE001 - report any launch failure to the model
        return {"ok": False, "error": f"Couldn't open '{raw}': {exc}"}


def _run_tool(name: str, args: dict, actions: list[dict]) -> dict:
    if name == "list_projects":
        return {"projects": snapshot()}
    if name == "launch_project":
        return launcher.ensure_running(str(args.get("id", "")))
    if name == "open_project":
        result = launcher.ensure_running(str(args.get("id", "")))
        if result.get("ok") and result.get("url"):
            actions.append({"type": "open_url", "url": result["url"], "name": result.get("name")})
        return result
    if name == "system_stats":
        return _system_stats()
    if name == "open_website":
        url = _resolve_url(str(args.get("site", "")), str(args.get("query", "")))
        actions.append({"type": "open_url", "url": url, "name": url})
        return {"ok": True, "url": url}
    if name == "open_app":
        return _open_app(str(args.get("name", "")))
    return {"error": f"Unknown tool '{name}'."}


def chat(user_message: str, history: list[dict], model: str | None = None) -> dict:
    """Run one JARVIS turn. Returns {reply, actions, model}."""
    model = model or detect_model()
    if not model:
        return {
            "reply": "I'm afraid my reasoning core is offline, sir — Ollama isn't responding. "
            "Start it and pull a model, then try me again.",
            "actions": [],
            "model": None,
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-12:])  # keep context bounded
    messages.append({"role": "user", "content": user_message})

    actions: list[dict] = []

    # Tool loop. On a small CPU model, every round is ~20s and each extra tool
    # call compounds the wait, so we allow tools on the FIRST generation only.
    # Once tools have run, we re-call WITHOUT the tools param, forcing the model
    # to narrate the result in words instead of looping. Worst case: 2 rounds.
    #
    # Intent gate: only offer tools when the user clearly wants an action, so a
    # plain question is answered directly without spawning a stray browser tab.
    tools_offered = bool(_ACTION_INTENT.search(user_message))
    for _ in range(3):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.6},
        }
        if tools_offered:
            payload["tools"] = TOOLS
        try:
            res = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
            res.raise_for_status()
            msg = res.json().get("message", {})
        except requests.RequestException as exc:
            return {"reply": f"My apologies, sir — a fault in the link: {exc}", "actions": actions, "model": model}

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return {"reply": (msg.get("content") or "").strip() or "…", "actions": actions, "model": model}

        # Record the assistant's tool request, then answer each call.
        messages.append(msg)
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            result = _run_tool(name, raw_args or {}, actions)
            messages.append({"role": "tool", "content": json.dumps(result)})

        # Tools have run once; force a spoken answer on the next pass.
        tools_offered = False

    # Fell through — summarise whatever actions we performed.
    if actions:
        names = ", ".join(a.get("name") or a.get("url", "") for a in actions)
        return {"reply": f"Done, sir — {names} is up.", "actions": actions, "model": model}
    return {"reply": "Done, sir.", "actions": actions, "model": model}
