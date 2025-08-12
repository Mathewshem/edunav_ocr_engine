# scripts/tts_engine.py
"""
Simple cross-platform text-to-speech helper.
- On Windows: uses pyttsx3 (offline TTS)
- On Raspberry Pi/Linux: uses espeak if available, else pyttsx3.
"""

import platform
import subprocess

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


def speak_text(text: str):
    system = platform.system().lower()

    # Raspberry Pi / Linux first choice: espeak
    if "linux" in system:
        try:
            subprocess.run(["espeak", text], check=True)
            return
        except Exception as e:
            print(f"[TTS] espeak failed: {e}")

    # Windows or fallback: pyttsx3
    if pyttsx3:
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS] pyttsx3 failed: {e}")
    else:
        print("[TTS] No available TTS engine installed.")
