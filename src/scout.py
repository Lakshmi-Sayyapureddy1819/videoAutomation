import os
import yt_dlp
import platform
from googleapiclient.discovery import build
import requests
from dotenv import load_dotenv
from internetarchive import search_items

# Project root: one level up from src/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if platform.system() == "Windows":
    RAW_VIDEOS_DIR = os.path.join(_PROJECT_ROOT, "data", "raw_videos")
else:
    # Change from /tmp to workspace for persistence
    RAW_VIDEOS_DIR = "/workspace/data/raw_videos"

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
COVERR_API_KEY = os.getenv("COVERR_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

USED_VIDEO_IDS = []

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
            links = []
            for i in res.get('items', []):
                vid_id = i['id']['videoId']
                if vid_id not in USED_VIDEO_IDS:
                    USED_VIDEO_IDS.append(vid_id)
                    links.append(f"https://www.youtube.com/watch?v={vid_id}")
            
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
                for i in res.get('items', []):
                    vid_id = i['id']['videoId']
                    if vid_id not in USED_VIDEO_IDS:
                        USED_VIDEO_IDS.append(vid_id)
                        links.append(f"https://www.youtube.com/watch?v={vid_id}")
            
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
                if vid_id not in USED_VIDEO_IDS:
                    USED_VIDEO_IDS.append(vid_id)
                    links.append(f"https://www.youtube.com/watch?v={vid_id}")
        return links
    except Exception as e:
        print(f"❌ Search failed for '{query}': {e}")
        return []

def search_pexels(query, limit=1):
    """Searches Pexels for videos."""
    if not PEXELS_API_KEY:
        return []
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        url = f"https://api.pexels.com/videos/search?query={query}&per_page={limit}&orientation=landscape"
        res = requests.get(url, headers=headers, timeout=10).json()
        # Return the highest quality link
        links = []
        for v in res.get('videos', []):
            if v.get('video_files'):
                best_file = max(v['video_files'], key=lambda x: x.get('width', 0))
                links.append(best_file['link'])
        return links
    except Exception as e:
        print(f"⚠️ Pexels API failed: {e}")
        return []

def search_pixabay(query, limit=1):
    """Searches Pixabay for videos."""
    if not PIXABAY_API_KEY:
        return []
    try:
        url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={query}&per_page={limit}&video_type=film&orientation=horizontal"
        res = requests.get(url, timeout=10).json()
        # Prefer large resolution
        links = []
        for v in res.get('hits', []):
            if v.get('videos'):
                if 'large' in v['videos']: links.append(v['videos']['large']['url'])
                elif 'medium' in v['videos']: links.append(v['videos']['medium']['url'])
        return links
    except Exception as e:
        print(f"⚠️ Pixabay API failed: {e}")
        return []

def search_coverr(query, limit=1):
    """Searches Coverr for high-quality cinematic B-roll."""
    if not COVERR_API_KEY:
        return []
    try:
        # Note: Coverr's free API might be limited. This is a sample structure.
        url = f"https://api.coverr.co/videos?query={query}&api_key={COVERR_API_KEY}"
        res = requests.get(url, timeout=10).json()
        return [v['urls']['mp4'] for v in res.get('hits', [])[:limit]]
    except Exception as e:
        print(f"⚠️ Coverr API failed: {e}")
        return []

def search_unsplash_photo(query, limit=1):
    """Fallback to a high-res photo for Ken Burns effect if no video is found."""
    if not UNSPLASH_ACCESS_KEY:
        return []
    try:
        url = f"https://api.unsplash.com/search/photos?query={query}&per_page={limit}&client_id={UNSPLASH_ACCESS_KEY}"
        res = requests.get(url, timeout=10).json()
        # Return the full resolution URL
        return [p['urls']['full'] for p in res.get('results', [])]
    except Exception as e:
        print(f"⚠️ Unsplash API failed: {e}")
        return []

def search_nasa(query, limit=1):
    """Searches NASA Image and Video Library (Great for space/science)."""
    try:
        url = "https://images-api.nasa.gov/search"
        params = {
            "q": query,
            "media_type": "video",
            "page_size": limit
        }
        res = requests.get(url, params=params, timeout=10).json()
        links = []
        for item in res.get('collection', {}).get('items', []):
            href = item.get('href')
            if href:
                try:
                    # NASA returns a collection JSON that lists the actual video files
                    media_res = requests.get(href, timeout=5).json()
                    for m in media_res:
                        # Prefer original quality or standard mp4
                        if m.endswith("~orig.mp4"):
                            links.append(m)
                            break
                        elif m.endswith(".mp4"):
                            links.append(m)
                            break
                except Exception:
                    pass
        return links
    except Exception as e:
        print(f"⚠️ NASA API failed: {e}")
        return []

def search_internet_archive(query, limit=1):
    """Searches Internet Archive for public domain footage (Great for history)."""
    try:
        base_url = "https://archive.org/advancedsearch.php"
        # Construct query for movies/videos
        q = f"{query} AND mediatype:(movies)"
        params = {
            "q": q,
            "fl[]": "identifier",
            "rows": limit,
            "output": "json"
        }
        res = requests.get(base_url, params=params, timeout=10).json()
        docs = res.get('response', {}).get('docs', [])
        results = []
        for doc in docs:
            identifier = doc['identifier']
            results.append(f"https://archive.org/download/{identifier}/{identifier}.mp4")
        return results
    except Exception as e:
        print(f"⚠️ Internet Archive failed: {e}")
        return []

def get_best_visual_source(query, limit=1):
    """
    Three-Tier Fallback Strategy to find the best video source.
    Returns a tuple: (list_of_links, source_name)
    The order is optimized for client-grade, distraction-free content.
    """
    # Tier 1: Historical & Scientific (Highest accuracy for documentaries)
    if any(k in query.lower() for k in ["history", "archive", "war", "1920s", "1940s", "1950s", "1960s"]):
        links = search_internet_archive(query, limit=limit)
        if links: return links, "internet_archive"
    if any(k in query.lower() for k in ["nasa", "space", "rocket", "science"]):
        links = search_nasa(query, limit=limit)
        if links: return links, "nasa"

    # Tier 2: High-Quality Cinematic Stock
    links = search_coverr(query, limit=limit)
    if links: return links, "coverr"
    
    links = search_pexels(query, limit=limit)
    if links: return links, "pexels"

    # Tier 3: General Stock & YouTube Fallback
    links = search_pixabay(query, limit=limit)
    if links: return links, "pixabay"

    links = get_youtube_links(query, limit=limit)
    if links: return links, "youtube"

    # Final Fallback: High-Res Photo for Ken Burns effect
    print("⚠️ No video found. Falling back to high-resolution photo search...")
    photo_links = search_unsplash_photo(query, limit=limit)
    if photo_links: return photo_links, "unsplash_photo"

    return [], None

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
        'nopart': True,
        'socket_timeout': 30,
        'retries': 5,
        'merge_output_format': 'mp4',
        'js_runtimes': {"node": {}, "deno": {}},
        'progress_hooks': [progress_hook],
        "remote_components": ["ejs:github"],
        # Fix for "Sign in to confirm you’re not a bot" - Use Android client
        'extractor_args': {'youtube': {'player_client': ['android', 'ios']}},
    }
    
    # Inject PO Token if available (Fixes 403/Bot detection)
    po_token = os.getenv("YOUTUBE_PO_TOKEN")
    if po_token:
        if 'youtube' not in base_opts['extractor_args']:
            base_opts['extractor_args']['youtube'] = {}
        base_opts['extractor_args']['youtube']['po_token'] = [po_token]
        if os.getenv("YOUTUBE_VISITOR_DATA"):
            base_opts['extractor_args']['youtube']['visitor_data'] = [os.getenv("YOUTUBE_VISITOR_DATA")]

    # Optional: Use cookies.txt if present in project root
    cookies_path = os.path.join(_PROJECT_ROOT, "cookies.txt")
    if os.path.exists(cookies_path):
        base_opts['cookiefile'] = cookies_path
    
    # Strategies: 1. Optimized 720p (Fast/Small), 2. Robust Fallback (Any quality)
    strategies = [
        'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
        'best[height<=480][ext=mp4]/best'
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
