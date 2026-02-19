"""
Voiceover using OpenAI TTS (no ElevenLabs required). Uses OPENAI_API_KEY from .env.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "output")


def generate_voiceover_openai(script: str, output_path: str = None, voice: str = "alloy") -> str:
    """
    Generate speech from script using OpenAI TTS. Saves to data/output/voiceover.mp3.
    Returns path to the audio file. Script is chunked if too long (TTS has a limit).
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
        raise ValueError("OPENAI_API_KEY not set in .env")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "voiceover.mp3")
    output_path = os.path.abspath(output_path)
    script = (script or "").strip()
    if not script:
        raise ValueError("Script is empty")
    # OpenAI TTS limit ~4096 chars per request; split by paragraphs
    chunks = [s.strip() for s in script.replace("\n\n", "\n").split("\n") if s.strip()]
    if not chunks:
        chunks = [script[:4000]]
    else:
        # Merge small chunks so we don't make too many requests
        merged = []
        buf = ""
        for c in chunks:
            if len(buf) + len(c) < 3500:
                buf = (buf + " " + c).strip()
            else:
                if buf:
                    merged.append(buf)
                buf = c
        if buf:
            merged.append(buf)
        chunks = merged[:10]
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    audio_parts = []
    for i, text in enumerate(chunks):
        resp = client.audio.speech.create(model="tts-1-hd", voice=voice, input=text[:4096])
        audio_parts.append(resp.content)
    with open(output_path, "wb") as f:
        for part in audio_parts:
            f.write(part)
    return output_path
