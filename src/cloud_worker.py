import runpod
import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

# The API Key is the "Passport" that lets you in without a password
runpod.api_key = os.getenv("RUNPOD_API_KEY")

def ping_runpod():
    """Checks if the RunPod Pod is reachable."""
    pod_id = os.getenv("RUNPOD_POD_ID", "tx1n5nqj29goku")
    if not pod_id or pod_id.startswith("your_"):
        return False, "POD_ID not set in .env"
    
    api_url = f"https://{pod_id}-8000.proxy.runpod.net/health"
    try:
        response = requests.get(api_url, timeout=3)
        if response.status_code == 200:
            return True, "Online"
        if response.status_code == 502:
            return False, "Status 502 (Pod starting or Port 8000 not exposed)"
        return False, f"Status {response.status_code} (Server running but /health missing)"
    except Exception as e:
        return False, f"Unreachable ({e})"

def render_video_on_cloud(segments, audio_path, subtitle_text):
    """
    Sends the Edit Decision List to the Pod for rendering.
    segments: list of {'url': str, 'start': float}
    """
    pod_id = os.getenv("RUNPOD_POD_ID", "tx1n5nqj29goku")
    if not pod_id: return None, "POD_ID not set"
    
    api_url = f"https://{pod_id}-8000.proxy.runpod.net/render"
    print(f"☁️ Sending Render Job to {api_url}...")
    
    try:
        files = {}
        if audio_path and os.path.exists(audio_path):
            files['audio'] = open(audio_path, 'rb')
            
        payload = {
            'data': json.dumps({'segments': segments, 'subtitles': subtitle_text})
        }
        
        response = requests.post(api_url, files=files, data=payload, timeout=1200) # 20 min timeout for render
        if response.status_code == 200:
            return response.content, None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)

def run_task_on_cloud(video_url=None, prompt=None, file_path=None):
    """
    Sends the video URL to the RunPod GPU.
    The GPU downloads it, analyzes it, and sends back the result.
    Supports both Direct Pod Connection and Serverless Endpoint.
    """
    # --- OPTION 1: Direct Pod Connection (Persistent GPU) ---
    pod_id = os.getenv("RUNPOD_POD_ID", "tx1n5nqj29goku")
    if pod_id and not pod_id.startswith("your_"):
        # Construct the proxy URL for the pod (Port 8000)
        # Format: https://{pod_id}-8000.proxy.runpod.net
        api_url = f"https://{pod_id}-8000.proxy.runpod.net/analyze"
        print(f"☁️ Sending task to RunPod Proxy: {api_url}")
        
        try:
            response = requests.post(api_url, json={"url": video_url, "prompt": prompt}, timeout=600)
            if file_path:
                # Upload mode: Send the file binary
                print(f"☁️ Uploading {os.path.basename(file_path)} to Pod...")
                with open(file_path, "rb") as f:
                    response = requests.post(api_url, files={"file": f}, data={"prompt": prompt}, timeout=600)
            else:
                # URL mode: Send JSON
                response = requests.post(api_url, json={"url": video_url, "prompt": prompt}, timeout=600)
            
            # Check for server errors and print the specific message from the pod
            if response.status_code != 200:
                try:
                    error_msg = response.json().get("error", response.text)
                except Exception:
                    error_msg = response.text
                print(f"❌ RunPod Server Error ({response.status_code}): {error_msg}")
                return {"error": f"RunPod Error: {error_msg}"}

            data = response.json()
            
            # Handle Async Job Response (Fixes 524 Timeout)
            if "job_id" in data:
                job_id = data["job_id"]
                print(f"⏳ Job {job_id} queued. Polling for results...")
                
                status_url = f"https://{pod_id}-8000.proxy.runpod.net/status/{job_id}"
                start_time = time.time()
                
                while time.time() - start_time < 600: # 10 min timeout
                    try:
                        poll_res = requests.get(status_url, timeout=10)
                        if poll_res.status_code == 200:
                            job_data = poll_res.json()
                            status = job_data.get("status")
                            
                            if status == "completed":
                                return {"timestamp": job_data["timestamp"]}
                            elif status == "failed":
                                return {"error": job_data.get("error", "Unknown error")}
                        
                        time.sleep(2)
                    except Exception as e:
                        print(f"⚠️ Polling error: {e}")
                        time.sleep(2)
                return {"error": "Polling timed out"}
            
            return data
        except Exception as e:
            print(f"❌ Pod Connection Error: {e}")
            return {"error": str(e)}

    # --- OPTION 2: Serverless Endpoint ---
    # This ID is created when you set up the 'Endpoint' in RunPod
    if file_path:
        print("⚠️ Serverless endpoints do not support direct file uploads yet. Use Direct Pod Connection.")
        return {"error": "Uploads require persistent pod"}
        
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "YOUR_ENDPOINT_ID")
    endpoint = runpod.Endpoint(endpoint_id) 
    
    try:
        # This starts the 'Job' on the client's paid cloud account
        # Note: We send 'url' to match what rp_handler.py expects
        job = endpoint.run({"url": video_url, "prompt": prompt})
        
        print(f"☁️ Job {job.id} sent to RunPod. Waiting for GPU...")
        result = job.get(timeout=600) # Wait up to 10 mins for 10-min video
        return result
    except Exception as e:
        print(f"❌ Cloud Error: {e}")
        return None