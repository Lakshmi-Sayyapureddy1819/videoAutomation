import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
COVERR_API_KEY = os.getenv("COVERR_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

def validate_keys():
    missing = []
    if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
    # Pexels/Pixabay are optional but recommended for stock
    
    if missing:
        return False, f"Missing keys: {', '.join(missing)}"
    return True, "OK"