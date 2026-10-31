import io
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel

WAKE_WORD = "hey jarvis"
STOP_WORD = "stop jarvis"

SYSTEM_PROMPT = (
    "You are Jarvis, a concise and polished British AI assistant. "
    "You provide practical answers, short when possible, and ask clarifying questions only when needed."
)


@dataclass
class Config:
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    ollama_model: str = "llama3.1:latest"
    whisper_model_size: str = "base"
    language: str = "en"
    sample_rate: int = 16000
    record_seconds: int = 7


class VoiceAssistant:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.whisper = WhisperModel(config.whisper_model_size, device="cpu", compute_type="int8")
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def calibrate(self) -> None:
        print("Microphone ready (sounddevice mode, no PyAudio).")

    def record_phrase_to_wav(self) -> str | None:
        print(f"\nListening for {self.config.record_seconds}s... (say 'hey jarvis ...')")
        try:
            frames = int(self.config.record_seconds * self.config.sample_rate)
            audio = sd.rec(frames, samplerate=self.config.sample_rate, channels=1, dtype="int16")
            sd.wait()
        except Exception as exc:
            print(f"Audio capture error: {exc}")
            return None

        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="jarvis_")
        os.close(fd)
        sf.write(temp_path, audio, self.config.sample_rate, subtype="PCM_16")
        return temp_path

    def transcribe(self, wav_path: str) -> str:
        segments, _ = self.whisper.transcribe(
            wav_path,
            language=self.config.language,
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def extract_command_if_awake(self, text: str) -> str | None:
        lowered = text.lower()
        idx = lowered.find(WAKE_WORD)
        if idx == -1:
            return None
        return text[idx + len(WAKE_WORD):].strip(" ,.:;!-")

    def think(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})

        convo = []
        for msg in self.history:
            if msg["role"] == "system":
                continue
            convo.append(f"{msg['role'].upper()}: {msg['content']}")
        prompt = f"{SYSTEM_PROMPT}\n" + "\n".join(convo) + "\nASSISTANT:"

        try:
            result = subprocess.run(
                ["ollama", "run", self.config.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            raise RuntimeError(f"Ollama CLI failed: {stderr or stdout}") from exc

        assistant_text = (result.stdout or "").strip()
        if not assistant_text:
            raise RuntimeError("Ollama returned empty response.")

        self.history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def speak(self, text: str) -> None:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.elevenlabs_voice_id}"
        headers = {
            "xi-api-key": self.config.elevenlabs_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.85, "style": 0.3},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()

        audio_stream = io.BytesIO(response.content)
        data, samplerate = sf.read(audio_stream, dtype="float32")
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        sd.play(data, samplerate=samplerate)
        sd.wait()

    def run(self) -> None:
        self.calibrate()
        print("\nJarvis wake-word mode started. Say 'hey jarvis' before your request.")
        print("Say 'stop jarvis' to exit.")

        while True:
            wav_path = self.record_phrase_to_wav()
            if not wav_path:
                continue

            try:
                text = self.transcribe(wav_path)
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

            if not text:
                print("Heard silence/noise, waiting...")
                continue

            print(f"You: {text}")
            lowered = text.lower()

            if STOP_WORD in lowered:
                print("Jarvis shutting down.")
                break

            command = self.extract_command_if_awake(text)
            if command is None:
                print("Wake word not detected. Waiting...")
                continue
            if not command:
                print("Yes? Say your request after 'hey jarvis'.")
                continue

            try:
                answer = self.think(command)
            except Exception as exc:
                print(f"LLM error: {exc}")
                continue

            print(f"Jarvis: {answer}")
            try:
                self.speak(answer)
            except Exception as exc:
                print(f"TTS error (check ElevenLabs key/voice): {exc}")
                time.sleep(0.8)


def load_config() -> Config:
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY missing in .env")
    if not voice_id:
        raise ValueError("ELEVENLABS_VOICE_ID missing in .env")

    return Config(
        elevenlabs_api_key=api_key,
        elevenlabs_voice_id=voice_id,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
        language=os.getenv("LANGUAGE", "en"),
    )


if __name__ == "__main__":
    cfg = load_config()
    assistant = VoiceAssistant(cfg)
    assistant.run()