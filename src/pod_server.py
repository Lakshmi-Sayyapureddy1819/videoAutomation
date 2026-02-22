"""
Run this script INSIDE your RunPod GPU Pod.
It listens for requests from your local Streamlit app.

Commands:
- On RunPod (Linux): python3 -m gunicorn -w 2 -k gevent -b 0.0.0.0:8000 --timeout 1500 pod_server:app
- On Local Windows:  python src/pod_server.py
"""
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import open_clip
import threading
import uuid
import time
import os
import sys
import json
import yt_dlp

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor import analyze_frames, trim_final_video

app = Flask(__name__)

# --- PRE-LOAD MODEL (Saves ~45s per request & prevents timeouts) ---
print("⏳ Loading CLIP Model on GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion400m_e32", force_quick_gelu=True)
model = model.to(device).eval()
print("✅ Model Ready.")
print("🎧 VidRush Pipeline Ready: /health, /analyze, /cleanup, /render (Whisper+GPU)")

# --- JOB QUEUE (File-Based for Multi-Worker Support) ---
JOB_DIR = "/tmp/vidrush_jobs"
os.makedirs(JOB_DIR, exist_ok=True)

def save_job(job_id, data):
    with open(os.path.join(JOB_DIR, f"{job_id}.json"), "w") as f:
        json.dump(data, f)

def get_job(job_id):
    path = os.path.join(JOB_DIR, f"{job_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def download_video_direct(url, output_dir="/tmp/raw_videos"):
    """Self-contained download function to avoid dependency on scout.py"""
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'nopart': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'ios']}},
    }
    
    # Inject PO Token if available in Pod Environment
    po_token = os.getenv("YOUTUBE_PO_TOKEN")
    if po_token:
        ydl_opts['extractor_args']['youtube']['po_token'] = [po_token]
        if os.getenv("YOUTUBE_VISITOR_DATA"):
            ydl_opts['extractor_args']['youtube']['visitor_data'] = [os.getenv("YOUTUBE_VISITOR_DATA")]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        print(f"Download error: {e}")
        return None

def background_analyze(job_id, url, prompt, video_path, model, preprocess, device):
    """Runs analysis in background to avoid Cloudflare 524 timeouts."""
    try:
        job = get_job(job_id)
        job['status'] = 'processing'
        save_job(job_id, job)
        
        # 1. Download (if URL provided and no path yet)
        if url and not video_path:
            video_path = download_video_direct(url)
            
        if not video_path or not os.path.exists(video_path):
             job['status'] = 'failed'
             job['error'] = "Download/Upload failed"
             save_job(job_id, job)
             return

        # 2. Analyze
        timestamp = analyze_frames(video_path, prompt, model=model, preprocess=preprocess, device=device)
        
        # 3. Cleanup
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass
            
        job['status'] = 'completed'
        job['timestamp'] = timestamp
        save_job(job_id, job)
        print(f"✅ Job {job_id} finished: {timestamp}s")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        job = get_job(job_id)
        job['status'] = 'failed'
        job['error'] = str(e)
        save_job(job_id, job)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "gpu": torch.cuda.is_available()}), 200

@app.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    job = get_job(job_id)
    if not job: return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/analyze', methods=['POST'])
def analyze():
    prompt = None
    video_path = None
    url = None
    
    # Check if it's a JSON request (YouTube URL)
    if request.is_json:
        data = request.json
        url = data.get('url')
        prompt = data.get('prompt')
    # Check if it's a File Upload (Multipart)
    elif 'file' in request.files:
        file = request.files['file']
        prompt = request.form.get('prompt')
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join("/tmp", filename)
            print(f"📂 Receiving upload: {filename}")
            file.save(video_path)

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    print(f"🚀 Processing: {url if url else 'File Upload'} with prompt: {prompt}")
    
    # Start Async Job
    job_id = str(uuid.uuid4())
    job_data = {'status': 'queued', 'submitted_at': time.time()}
    save_job(job_id, job_data)
    
    thread = threading.Thread(target=background_analyze, args=(job_id, url, prompt, video_path, model, preprocess, device))
    thread.start()
    
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Deletes temporary video files to free up space."""
    removed_count = 0
    # Check both /tmp and the raw_videos subdir
    dirs_to_clean = ["/tmp", "/tmp/raw_videos", JOB_DIR]
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith((".mp4", ".mov", ".avi", ".mkv", ".json")):
                    try: os.remove(os.path.join(d, f)); removed_count += 1
                    except: pass
    return jsonify({"status": "cleaned", "files_removed": removed_count})

@app.route('/render', methods=['POST'])
def render():
    """
    VidRush Cloud Render:
    Receives an Edit Decision List (JSON) and Audio.
    Downloads clips, stitches them on GPU, and returns the final MP4.
    """
    import json
    from flask import send_file
    
    try:
        # 1. Get Audio
        audio_file = request.files.get('audio')
        audio_path = None
        if audio_file:
            audio_path = os.path.join("/tmp", secure_filename(audio_file.filename))
            audio_file.save(audio_path)
            
        # 2. Get EDL (Edit Decision List)
        data_str = request.form.get('data')
        if not data_str: return jsonify({"error": "Missing data JSON"}), 400
        
        data = json.loads(data_str)
        segments_info = data.get('segments', []) # List of {url, start}
        subtitle_text = data.get('subtitles')
        
        # 3. Download Clips (Fast on Cloud)
        local_segments = []
        for seg in segments_info:
            url = seg.get('url')
            start = float(seg.get('start', 0))
            if url:
                path = download_video_direct(url)
                if path: local_segments.append((path, start))
        
        if not local_segments: return jsonify({"error": "No segments processed"}), 400

        # 4. Render with VidRush Processor
        output_path = os.path.join("/tmp", f"render_{os.urandom(4).hex()}.mp4")
        use_gpu = torch.cuda.is_available()
        
        from processor import trim_final_video
        final_path = trim_final_video(local_segments, audio_path, output_path, 
                                      subtitle_text=subtitle_text, use_gpu=use_gpu,
                                      crop_headers=True, mute_original=True, fade_duration=0.5)
        
        return send_file(final_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible via RunPod Proxy
    print("🎧 Listening on port 8000... (Health Check Enabled)")
    app.run(host='0.0.0.0', port=8000)