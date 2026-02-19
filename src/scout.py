import os
import yt_dlp
import platform
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Project root: one level up from src/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if platform.system() == "Windows":
    RAW_VIDEOS_DIR = os.path.join(_PROJECT_ROOT, "data", "raw_videos")
else:
    # Linux/RunPod: Use /tmp for faster IO and to avoid permission issues
    RAW_VIDEOS_DIR = "/tmp/raw_videos"

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_links(query, limit=2):
    """Search YouTube via API (preferred) or yt-dlp fallback."""
    # Simplify the query: Use only the first 5-6 words to avoid API confusion/timeouts
    short_query = " ".join(query.split()[:6])

    # 1. Try YouTube Data API (Fast, reliable, requires YOUTUBE_API_KEY in .env)
    if API_KEY:
        try:
            youtube = build('youtube', 'v3', developerKey=API_KEY, cache_discovery=False)
            
            # SEARCH 1: Attempt Creative Commons (Legal)
            request = youtube.search().list(
                q=f"{short_query} cinematic 4k",
                part="id,snippet",
                maxResults=limit,
                type="video",
                videoLicense="creativeCommon"
            )
            res = request.execute()
            links = [f"https://www.youtube.com/watch?v={i['id']['videoId']}" for i in res.get('items', [])]
            
            # FALLBACK: If CC finds nothing, search all videos
            if not links:
                print(f"⚠️ CC search failed for '{short_query}', trying general search...")
                request = youtube.search().list(
                    q=f"{short_query} stock footage 4k",
                    part="id,snippet",
                    maxResults=limit,
                    type="video"
                )
                res = request.execute()
                links = [f"https://www.youtube.com/watch?v={i['id']['videoId']}" for i in res.get('items', [])]
            
            if links:
                return links
        except Exception as e:
            print(f"⚠️ YouTube API failed (check key/quota): {e}. Falling back to yt-dlp.")

    # 2. Fallback to yt-dlp (Slower, no key needed, but may get rate-limited)
    print(f"🐢 Using yt-dlp fallback for search (slower). Query: '{short_query}'")
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    search_url = f"ytsearch{max(1, limit)}:{short_query}"
    ydl_opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "no_download": True,
        "socket_timeout": 30,
        "retries": 5,
        "ignoreerrors": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}, "deno": {}},
        "remote_components": ["ejs:github"],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
        if not info or "entries" not in info:
            return []
        entries = [e for e in info["entries"] if e]
        links = []
        for e in entries[:limit]:
            vid_id = e.get("id") or e.get("url", "").strip().split("watch?v=")[-1].split("&")[0]
            if vid_id:
                links.append(f"https://www.youtube.com/watch?v={vid_id}")
        return links
    except Exception as e:
        print(f"❌ Search failed for '{query}': {e}")
        return []


def download_video(url, progress_callback=None):
    """Download a YouTube video to data/raw_videos. Returns local file path. Skips download if already present."""
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    
    # Extract id so we can skip re-download
    vid_id = "unknown"
    if "watch?v=" in url:
        vid_id = url.split("watch?v=")[-1].split("&")[0].strip()
    
    # Check if file already exists
    if vid_id != "unknown":
        for name in os.listdir(RAW_VIDEOS_DIR):
            if name.startswith(vid_id) and name.endswith(".mp4"):
                if progress_callback:
                    progress_callback(1.0)
                return os.path.join(RAW_VIDEOS_DIR, name)

    out_tmpl = os.path.join(RAW_VIDEOS_DIR, "%(id)s.%(ext)s")
    
    def progress_hook(d):
        if progress_callback:
            if d['status'] == 'downloading':
                try:
                    total = d.get('total_bytes') or d.get('total_bytes_estimate')
                    downloaded = d.get('downloaded_bytes', 0)
                    if total:
                        progress_callback(downloaded / total)
                except Exception:
                    pass
            elif d['status'] == 'finished':
                progress_callback(1.0)
    
    # Common options
    base_opts = {
        'outtmpl': out_tmpl,
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        'socket_timeout': 30,
        'retries': 5,
        'merge_output_format': 'mp4',
        'js_runtimes': {"node": {}, "deno": {}},
        'progress_hooks': [progress_hook],
        "remote_components": ["ejs:github"],
        # Fix for "Sign in to confirm you’re not a bot" - Use Android client
        'extractor_args': {'youtube': {'player_client': ['android', 'ios']}},
    }
    
    # Optional: Use cookies.txt if present in project root
    cookies_path = os.path.join(_PROJECT_ROOT, "cookies.txt")
    if os.path.exists(cookies_path):
        base_opts['cookiefile'] = cookies_path
    
    # Strategies: 1. Optimized 720p (Fast/Small), 2. Robust Fallback (Any quality)
    strategies = [
        'best[height<=720][ext=mp4]/best[ext=mp4]/best',
        'bestvideo+bestaudio/best'
    ]

    for i, fmt in enumerate(strategies):
        ydl_opts = base_opts.copy()
        ydl_opts['format'] = fmt
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                ext = info.get("ext", "mp4")
                final_id = info.get("id", vid_id)
                return os.path.join(RAW_VIDEOS_DIR, f"{final_id}.{ext}")
        except Exception as e:
            if i < len(strategies) - 1:
                print(f"⚠️ Attempt {i+1} failed for {url}: {e}. Retrying with fallback...")
                continue
            print(f"❌ Download failed for {url}: {e}")
            return None


def check_youtube() -> tuple[bool, str]:
    """Return (success, message) for YouTube/yt-dlp status."""
    try:
        links = get_youtube_links("test", limit=1)
        return True, "OK" if links else "No results (check network)"
    except Exception as e:
        return False, str(e)[:80]
