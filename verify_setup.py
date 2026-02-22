import torch
import os
from dotenv import load_dotenv

def verify_all():
    print("--- 🛠️ VidRush Setup Verification ---")
    
    # 1. Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Detected: {gpu_name}")
        print(f"✅ Available VRAM: {total_vram:.2f} GB")
    else:
        print("❌ ERROR: GPU not detected! Check your RunPod template.")

    # 2. Check API Keys
    load_dotenv()
    keys = ["ELEVENLABS_API_KEY", "OPENAI_API_KEY", "YOUTUBE_API_KEY"]
    for key in keys:
        if os.getenv(key):
            print(f"✅ {key} found.")
        else:
            print(f"❌ ERROR: {key} missing from .env file.")

if __name__ == "__main__":
    verify_all()