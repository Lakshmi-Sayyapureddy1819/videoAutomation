import os
import sys
from cloud_worker import run_task_on_cloud
from processor import analyze_frames

# Setup paths to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # The specific file you requested
    video_path = r"C:\Users\CHARVY\videoAutomation\data\raw_videos\Screenrecording_20260210_161420.mp4"
    prompt = "Describe the specific action or scene you want to find in this recording"
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return

    print(f"📂 Processing: {video_path}")
    print(f"📝 Prompt: {prompt}")

    # Option 1: Cloud (RunPod)
    # Ensure .env has RUNPOD_API_KEY and RUNPOD_POD_ID
    print("\n--- Attempting Cloud Analysis (RunPod) ---")
    try:
        result = run_task_on_cloud(prompt=prompt, file_path=video_path)
        if result and "timestamp" in result:
            print(f"✅ Cloud Result: Found at {result['timestamp']}s")
        else:
            print(f"⚠️ Cloud failed: {result}")
    except Exception as e:
        print(f"❌ Cloud Error: {e}")

    # Option 2: Local (CPU/GPU)
    print("\n--- Attempting Local Analysis ---")
    try:
        ts = analyze_frames(video_path, prompt)
        print(f"✅ Local Result: Found at {ts}s")
    except Exception as e:
        print(f"❌ Local Error: {e}")

if __name__ == "__main__":
    main()