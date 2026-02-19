import runpod
import os
from scout import download_video
from processor import analyze_frames

def handler(job):
    """
    RunPod handler: Downloads video from URL and runs CLIP analysis.
    Input: {"url": "https://youtube.com/...", "prompt": "text description"}
    Output: {"timestamp": float}
    """
    job_input = job["input"]
    url = job_input.get("url")
    prompt = job_input.get("prompt")

    if not url or not prompt:
        return {"error": "Missing url or prompt"}

    print(f"🚀 Processing URL: {url} with prompt: {prompt}")

    # 1. Download (uses scout.py logic, ensuring 720p limit)
    # We download to the local container storage
    try:
        video_path = download_video(url)
        if not video_path or not os.path.exists(video_path):
            return {"error": "Download failed"}

        # 2. Analyze
        timestamp = analyze_frames(video_path, prompt)

        # 3. Cleanup (save space on the worker)
        if os.path.exists(video_path):
            os.remove(video_path)

        return {"timestamp": timestamp}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})