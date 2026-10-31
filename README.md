# Jarvis Local Assistant

A local-first voice assistant and browser HUD. Speech is transcribed with Faster-Whisper, answers come from a local Ollama model, and optional spoken replies use ElevenLabs text-to-speech.

The FastAPI HUD exposes local health, project status, and chat endpoints. Its project registry uses paths relative to your home folder; edit `server/projects.py` for your own project list.

## HUD

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
cd server
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. Ollama must be installed and running for local chat.

## Voice mode

Copy `.env.example` to `.env`, set your ElevenLabs credentials if you want speech output, then run `python app.py`. Voice mode uses the microphone continuously until you say “stop Jarvis”; it sends generated speech text to ElevenLabs when configured. Keep `.env` private.

## Dependencies

See `requirements.txt`. Model downloads for Whisper and Ollama happen locally; ElevenLabs is an optional external service for speech synthesis.
