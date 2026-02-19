import os
from elevenlabs import save
from elevenlabs.client import ElevenLabs
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "audio")

def generate_voiceover(text, output_path=None, voice_name="Adam"):
    """
    Generates a professional AI voiceover using ElevenLabs.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("⚠️ ELEVENLABS_API_KEY not found. Please set it in .env")
        return None

    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "narration_elevenlabs.mp3")

    client = ElevenLabs(api_key=api_key)
    
    try:
        # Using 'Adam' for documentary-style narration. 
        # You can change this to any voice ID or name available in your account.
        audio = client.generate(
            text=text,
            voice=voice_name,
            model="eleven_multilingual_v2"
        )
        
        save(audio, output_path)
        return output_path
    except Exception as e:
        print(f"ElevenLabs generation failed: {e}")
        return None
