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
import threading
import uuid
import time
import os
import sys
import json
import yt_dlp
import shutil

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor import trim_final_video
from models.video_model import XClipHandler
from vector_db import VectorDB

app = Flask(__name__)

# --- PRE-LOAD MODELS & DB ---
xclip = XClipHandler(device="cuda" if torch.cuda.is_available() else "cpu")
# The dimension (dim) must match the output of your X-CLIP model.
# The base model outputs 512-dimensional vectors.
vector_db = VectorDB(dim=512) 
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

def background_rank(job_id, candidates, prompt):
    """
    Ranks videos using X-CLIP, with a VectorDB cache to avoid re-processing.
    """
    try:
        job = get_job(job_id)
        job['status'] = 'processing'
        save_job(job_id, job)
        
        text_emb = xclip.encode_text(prompt)
        if text_emb is None:
            raise ValueError("Failed to encode prompt text.")

        ranked_results = []
        
        # Build a quick lookup for existing URLs in the DB
        url_to_idx = {meta['url']: i for i, meta in enumerate(vector_db.metadata)}
        
        new_items_added = False
        for cand in candidates:
            url = cand.get('url')
            if not url:
                continue

            vid_emb = None
            # --- Cache Check ---
            if url in url_to_idx:
                idx = url_to_idx[url]
                vid_emb = vector_db.index.reconstruct(idx).reshape(1, -1)
                print(f"⚡ Cache HIT for: {url}")
            
            # --- Process if not in cache ---
            else:
                print(f"🐌 Cache MISS for: {url}. Processing...")
                path = download_video_direct(url, output_dir="/tmp/rank_cache")
                if path:
                    try:
                        vid_emb = xclip.encode_video(path)
                        if vid_emb is not None:
                            # Add to DB for future requests
                            vector_db.add_item(vid_emb, {'url': url})
                            new_items_added = True
                    finally:
                        os.remove(path) # Cleanup immediately

            # --- Compute Score ---
            if vid_emb is not None:
                score = xclip.compute_similarity(text_emb, vid_emb)
                ranked_results.append({**cand, "score": score})

        # Save the DB if we added new items
        if new_items_added:
            print("💾 Saving updated vector database...")
            vector_db.save()
        
        # Sort by score
        ranked_results.sort(key=lambda x: x['score'], reverse=True)

        job['status'] = 'completed'
        job['results'] = ranked_results
        save_job(job_id, job)
        print(f"✅ Job {job_id} finished ranking {len(candidates)} clips.")
        
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

@app.route('/rank', methods=['POST'])
def rank_videos():
    """
    Receives a list of candidate video URLs and a prompt.
    Downloads them, ranks them using X-CLIP, and returns the best ones.
    """
    data = request.json
    prompt = data.get('prompt')
    candidates = data.get('candidates', []) # List of {url, source, id...}
    
    if not prompt or not candidates:
        return jsonify({"error": "Missing prompt or candidates"}), 400

    print(f"🚀 Ranking {len(candidates)} clips for prompt: {prompt}")
    
    # Start Async Job
    job_id = str(uuid.uuid4())
    job_data = {'status': 'queued', 'submitted_at': time.time()}
    save_job(job_id, job_data)
    
    thread = threading.Thread(target=background_rank, args=(job_id, candidates, prompt))
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