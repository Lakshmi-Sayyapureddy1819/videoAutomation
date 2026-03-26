import hashlib
import json
import math
import os
import platform
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
import yt_dlp
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Project root: one level up from src/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if platform.system() == "Windows":
    RAW_VIDEOS_DIR = os.path.join(_PROJECT_ROOT, "data", "raw_videos")
    VAULT_DIR = os.path.join(_PROJECT_ROOT, "data", "vault")
else:
    RAW_VIDEOS_DIR = os.path.join(_PROJECT_ROOT, "data", "raw_videos")
    VAULT_DIR = os.path.join(_PROJECT_ROOT, "data", "vault")

CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "cache")
SEARCH_CACHE_DIR = os.path.join(CACHE_DIR, "search")

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
COVERR_API_KEY = os.getenv("COVERR_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

USED_VIDEO_IDS = []


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _cache_key(source, query, limit):
    raw = f"{source}:{limit}:{query.strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_path(source, query, limit):
    _ensure_dir(SEARCH_CACHE_DIR)
    return os.path.join(SEARCH_CACHE_DIR, f"{_cache_key(source, query, limit)}.json")


def _load_cached_results(source, query, limit):
    path = _cache_path(source, query, limit)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _save_cached_results(source, query, limit, results):
    path = _cache_path(source, query, limit)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _parse_iso8601_duration(value):
    if not value:
        return None
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return (hours * 3600) + (minutes * 60) + seconds


def _candidate_metadata_text(candidate):
    fields = [
        candidate.get("title", ""),
        candidate.get("description", ""),
        " ".join(candidate.get("tags", []) or []),
        candidate.get("query", ""),
        candidate.get("source", ""),
    ]
    return " | ".join(part.strip() for part in fields if part and str(part).strip())


def _direct_extension_from_url(url, default=".mp4"):
    parsed = urlparse(url)
    _, ext = os.path.splitext(parsed.path)
    if ext:
        return ext.lower()
    return default


def _download_direct_media(url, progress_callback=None):
    _ensure_dir(RAW_VIDEOS_DIR)
    ext = _direct_extension_from_url(url)
    file_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    local_path = os.path.join(RAW_VIDEOS_DIR, f"asset_{file_hash}{ext}")

    if os.path.exists(local_path):
        if progress_callback:
            progress_callback(1.0)
        return local_path

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    written = 0

    with open(local_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            written += len(chunk)
            if progress_callback and total:
                progress_callback(min(1.0, written / total))

    if progress_callback:
        progress_callback(1.0)
    return local_path


def check_local_vault(visual_prompt):
    """Check for manually uploaded perfect clips first."""
    if not os.path.exists(VAULT_DIR):
        os.makedirs(VAULT_DIR, exist_ok=True)

    query_words = visual_prompt.lower().split()
    for file in os.listdir(VAULT_DIR):
        if file.lower().endswith((".mp4", ".mov", ".jpg", ".jpeg", ".png", ".webp")):
            if any(word in file.lower() for word in query_words if len(word) > 3):
                print(f"Found manual override in Vault: {file}")
                return os.path.join(VAULT_DIR, file)
    return None


def search_youtube_candidates(query, limit=10):
    cached = _load_cached_results("youtube", query, limit)
    if cached is not None:
        return cached

    short_query = " ".join(query.split()[:8]).strip()
    candidates = []

    if API_KEY:
        try:
            youtube = build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)
            search_response = youtube.search().list(
                q=short_query,
                part="id,snippet",
                maxResults=min(limit, 25),
                type="video",
                relevanceLanguage="en",
                videoEmbeddable="true",
            ).execute()

            video_ids = [item["id"]["videoId"] for item in search_response.get("items", []) if item.get("id", {}).get("videoId")]
            details = {}
            if video_ids:
                detail_response = youtube.videos().list(
                    id=",".join(video_ids),
                    part="snippet,contentDetails",
                    maxResults=min(limit, 25),
                ).execute()
                details = {item["id"]: item for item in detail_response.get("items", [])}

            for item in search_response.get("items", []):
                video_id = item.get("id", {}).get("videoId")
                if not video_id:
                    continue
                detail = details.get(video_id, {})
                snippet = detail.get("snippet", item.get("snippet", {}))
                thumbnail_url = (
                    snippet.get("thumbnails", {}).get("high", {}).get("url")
                    or snippet.get("thumbnails", {}).get("default", {}).get("url")
                    or item.get("snippet", {}).get("thumbnails", {}).get("default", {}).get("url")
                )
                candidate = {
                    "id": f"youtube:{video_id}",
                    "source": "youtube",
                    "query": query,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "page_url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "tags": snippet.get("tags", []) or [],
                    "thumbnail_url": thumbnail_url,
                    "duration": _parse_iso8601_duration(detail.get("contentDetails", {}).get("duration")),
                }
                candidate["metadata_text"] = _candidate_metadata_text(candidate)
                candidates.append(candidate)
                if len(candidates) >= limit:
                    break
        except Exception as e:
            print(f"YouTube API failed (check key/quota): {e}. Falling back to yt-dlp.")

    if not candidates:
        links = get_youtube_links(query, limit=limit, _allow_api=False)
        for link in links:
            video_id = link.split("watch?v=")[-1].split("&")[0]
            candidate = {
                "id": f"youtube:{video_id}",
                "source": "youtube",
                "query": query,
                "url": link,
                "page_url": link,
                "title": query,
                "description": "",
                "tags": short_query.split(),
                "thumbnail_url": None,
                "duration": None,
            }
            candidate["metadata_text"] = _candidate_metadata_text(candidate)
            candidates.append(candidate)

    _save_cached_results("youtube", query, limit, candidates)
    return candidates


def search_pexels_candidates(query, limit=5):
    cached = _load_cached_results("pexels", query, limit)
    if cached is not None:
        return cached
    if not PEXELS_API_KEY:
        return []
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        url = "https://api.pexels.com/videos/search"
        res = requests.get(
            url,
            headers=headers,
            params={"query": query, "per_page": limit, "orientation": "landscape"},
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        candidates = []
        for video in data.get("videos", []):
            files = video.get("video_files") or []
            if not files:
                continue
            best_file = max(files, key=lambda item: item.get("width", 0))
            candidate = {
                "id": f"pexels:{video.get('id')}",
                "source": "pexels",
                "query": query,
                "url": best_file.get("link"),
                "page_url": video.get("url"),
                "title": f"Pexels footage for {query}",
                "description": f"Pexels stock footage by {video.get('user', {}).get('name', 'unknown creator')}",
                "tags": query.split(),
                "thumbnail_url": video.get("image"),
                "duration": video.get("duration"),
            }
            candidate["metadata_text"] = _candidate_metadata_text(candidate)
            candidates.append(candidate)
        _save_cached_results("pexels", query, limit, candidates)
        return candidates
    except Exception as e:
        print(f"Pexels API failed: {e}")
        return []


def search_pixabay_candidates(query, limit=5):
    cached = _load_cached_results("pixabay", query, limit)
    if cached is not None:
        return cached
    if not PIXABAY_API_KEY:
        return []
    try:
        res = requests.get(
            "https://pixabay.com/api/videos/",
            params={
                "key": PIXABAY_API_KEY,
                "q": query,
                "per_page": limit,
                "video_type": "film",
                "orientation": "horizontal",
            },
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        candidates = []
        for video in data.get("hits", []):
            variants = video.get("videos", {})
            best = variants.get("large") or variants.get("medium") or variants.get("small")
            if not best:
                continue
            candidate = {
                "id": f"pixabay:{video.get('id')}",
                "source": "pixabay",
                "query": query,
                "url": best.get("url"),
                "page_url": video.get("pageURL"),
                "title": f"Pixabay footage for {query}",
                "description": video.get("tags", ""),
                "tags": [tag.strip() for tag in (video.get("tags") or "").split(",") if tag.strip()],
                "thumbnail_url": video.get("videos", {}).get("tiny", {}).get("thumbnail") or video.get("picture_id"),
                "duration": video.get("duration"),
            }
            candidate["metadata_text"] = _candidate_metadata_text(candidate)
            candidates.append(candidate)
        _save_cached_results("pixabay", query, limit, candidates)
        return candidates
    except Exception as e:
        print(f"Pixabay API failed: {e}")
        return []


def search_pexels(query, limit=1):
    return [candidate["url"] for candidate in search_pexels_candidates(query, limit=limit) if candidate.get("url")]


def search_pixabay(query, limit=1):
    return [candidate["url"] for candidate in search_pixabay_candidates(query, limit=limit) if candidate.get("url")]


def search_coverr(query, limit=1):
    """Searches Coverr for high-quality cinematic B-roll."""
    if not COVERR_API_KEY:
        return []
    try:
        url = f"https://api.coverr.co/videos?query={query}&api_key={COVERR_API_KEY}"
        res = requests.get(url, timeout=10).json()
        return [video["urls"]["mp4"] for video in res.get("hits", [])[:limit]]
    except Exception as e:
        print(f"Coverr API failed: {e}")
        return []


def search_unsplash_photo(query, limit=1):
    """Fallback to a high-res photo for Ken Burns effect if no video is found."""
    if not UNSPLASH_ACCESS_KEY:
        return []
    try:
        res = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": limit, "client_id": UNSPLASH_ACCESS_KEY},
            timeout=10,
        )
        res.raise_for_status()
        data = res.json()
        return [photo["urls"]["full"] for photo in data.get("results", [])]
    except Exception as e:
        print(f"Unsplash API failed: {e}")
        return []


def search_nasa(query, limit=1):
    """Searches NASA Image and Video Library (Great for space/science)."""
    try:
        res = requests.get(
            "https://images-api.nasa.gov/search",
            params={"q": query, "media_type": "video", "page_size": limit},
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
        links = []
        for item in data.get("collection", {}).get("items", []):
            href = item.get("href")
            if not href:
                continue
            try:
                media_res = requests.get(href, timeout=10).json()
                for media_url in media_res:
                    if media_url.endswith("~orig.mp4") or media_url.endswith(".mp4"):
                        links.append(media_url)
                        break
            except Exception:
                continue
        return links
    except Exception as e:
        print(f"NASA API failed: {e}")
        return []


def search_internet_archive(query, limit=1):
    """Searches Internet Archive for public domain footage (Great for history)."""
    try:
        res = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f"{query} AND mediatype:(movies)",
                "fl[]": "identifier,title,description",
                "rows": limit,
                "output": "json",
            },
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
        results = []
        for doc in data.get("response", {}).get("docs", []):
            identifier = doc["identifier"]
            results.append(f"https://archive.org/download/{identifier}/{identifier}.mp4")
        return results
    except Exception as e:
        print(f"Internet Archive failed: {e}")
        return []


def get_youtube_links(query, limit=2, _allow_api=True):
    """Search YouTube via API (preferred) or yt-dlp fallback."""
    if _allow_api:
        candidates = search_youtube_candidates(query, limit=limit)
        links = [candidate["url"] for candidate in candidates if candidate.get("url")]
        if links:
            return links[:limit]

    short_query = " ".join(query.split()[:6])
    print(f"Using yt-dlp fallback for search (slower). Query: '{short_query}'")
    _ensure_dir(RAW_VIDEOS_DIR)
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
        entries = [entry for entry in info["entries"] if entry]
        links = []
        for entry in entries[:limit]:
            vid_id = entry.get("id") or entry.get("url", "").strip().split("watch?v=")[-1].split("&")[0]
            if not vid_id:
                continue
            if vid_id not in USED_VIDEO_IDS:
                USED_VIDEO_IDS.append(vid_id)
                links.append(f"https://www.youtube.com/watch?v={vid_id}")
        return links
    except Exception as e:
        print(f"Search failed for '{query}': {e}")
        return []


def search_multi_source_candidates(queries, per_query_limit=15, total_limit=20):
    """
    Fetch candidate metadata from YouTube, Pexels, and Pixabay in parallel.
    Returns de-duplicated candidates with rich metadata for semantic ranking.
    """
    if isinstance(queries, str):
        queries = [queries]

    cleaned_queries = [query.strip() for query in queries if query and query.strip()]
    if not cleaned_queries:
        return []

    youtube_limit = max(1, math.ceil(per_query_limit * 0.5))
    pexels_limit = max(1, math.ceil(per_query_limit * 0.3))
    pixabay_limit = max(1, per_query_limit - youtube_limit - pexels_limit)

    futures = []
    candidates = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=min(12, len(cleaned_queries) * 3)) as executor:
        for query in cleaned_queries:
            futures.append(executor.submit(search_youtube_candidates, query, youtube_limit))
            futures.append(executor.submit(search_pexels_candidates, query, pexels_limit))
            futures.append(executor.submit(search_pixabay_candidates, query, pixabay_limit))

        for future in as_completed(futures):
            try:
                for candidate in future.result() or []:
                    url = candidate.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    candidate["metadata_text"] = candidate.get("metadata_text") or _candidate_metadata_text(candidate)
                    candidates.append(candidate)
                    if len(candidates) >= total_limit:
                        break
            except Exception:
                continue
            if len(candidates) >= total_limit:
                break

    return candidates[:total_limit]


def get_best_visual_source(query, limit=1):
    """
    Three-Tier fallback strategy to find the best visual source.
    Returns a tuple: (list_of_links, source_name)
    """
    vault_match = check_local_vault(query)
    if vault_match:
        return [vault_match], "local_vault"

    if os.path.exists(RAW_VIDEOS_DIR):
        for filename in os.listdir(RAW_VIDEOS_DIR):
            if filename.endswith((".mp4", ".jpg", ".jpeg", ".png")):
                query_words = query.lower().split()
                if any(word in filename.lower() for word in query_words if len(word) > 3):
                    return [os.path.join(RAW_VIDEOS_DIR, filename)], "local_manual"

    if any(token in query.lower() for token in ["history", "archive", "war", "1920s", "1940s", "1950s", "1960s"]):
        links = search_internet_archive(query, limit=limit)
        if links:
            return links, "internet_archive"
    if any(token in query.lower() for token in ["nasa", "space", "rocket", "science"]):
        links = search_nasa(query, limit=limit)
        if links:
            return links, "nasa"

    links = search_coverr(query, limit=limit)
    if links:
        return links, "coverr"

    links = search_pexels(query, limit=limit)
    if links:
        return links, "pexels"

    links = search_pixabay(query, limit=limit)
    if links:
        return links, "pixabay"

    links = get_youtube_links(query, limit=limit)
    if links:
        return links, "youtube"

    print("No video found. Falling back to high-resolution photo search...")
    photo_links = search_unsplash_photo(query, limit=limit)
    if photo_links:
        return photo_links, "unsplash_photo"

    return [], None


def download_video(url, progress_callback=None):
    """
    Download media to data/raw_videos.
    Supports local files, direct mp4/image URLs, and YouTube URLs.
    Returns the local file path.
    """
    if os.path.exists(url):
        return url

    _ensure_dir(RAW_VIDEOS_DIR)

    if url.startswith("http") and not ("youtube.com/watch" in url or "youtu.be/" in url):
        try:
            return _download_direct_media(url, progress_callback=progress_callback)
        except Exception as e:
            print(f"Direct media download failed for {url}: {e}")

    vid_id = "unknown"
    if "watch?v=" in url:
        vid_id = url.split("watch?v=")[-1].split("&")[0].strip()
    elif "youtu.be/" in url:
        vid_id = url.split("youtu.be/")[-1].split("?")[0].strip()

    if vid_id != "unknown":
        for name in os.listdir(RAW_VIDEOS_DIR):
            if name.startswith(vid_id) and name.endswith(".mp4"):
                if progress_callback:
                    progress_callback(1.0)
                return os.path.join(RAW_VIDEOS_DIR, name)

    out_tmpl = os.path.join(RAW_VIDEOS_DIR, "%(id)s.%(ext)s")

    def progress_hook(data):
        if not progress_callback:
            return
        if data["status"] == "downloading":
            try:
                total = data.get("total_bytes") or data.get("total_bytes_estimate")
                downloaded = data.get("downloaded_bytes", 0)
                if total:
                    progress_callback(downloaded / total)
            except Exception:
                pass
        elif data["status"] == "finished":
            progress_callback(1.0)

    base_opts = {
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "nopart": True,
        "socket_timeout": 30,
        "retries": 5,
        "merge_output_format": "mp4",
        "js_runtimes": {"node": {}, "deno": {}},
        "progress_hooks": [progress_hook],
        "remote_components": ["ejs:github"],
        "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
    }

    po_token = os.getenv("YOUTUBE_PO_TOKEN")
    if po_token:
        base_opts["extractor_args"]["youtube"]["po_token"] = [po_token]
        if os.getenv("YOUTUBE_VISITOR_DATA"):
            base_opts["extractor_args"]["youtube"]["visitor_data"] = [os.getenv("YOUTUBE_VISITOR_DATA")]

    cookies_path = os.path.join(_PROJECT_ROOT, "cookies.txt")
    if os.path.exists(cookies_path):
        base_opts["cookiefile"] = cookies_path

    strategies = [
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "best[height<=480][ext=mp4]/best",
    ]

    for index, fmt in enumerate(strategies):
        ydl_opts = base_opts.copy()
        ydl_opts["format"] = fmt
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                ext = info.get("ext", "mp4")
                final_id = info.get("id", vid_id)
                return os.path.join(RAW_VIDEOS_DIR, f"{final_id}.{ext}")
        except Exception as e:
            if index < len(strategies) - 1:
                print(f"Attempt {index + 1} failed for {url}: {e}. Retrying with fallback...")
                continue
            print(f"Download failed for {url}: {e}")
            return None


def check_youtube() -> tuple[bool, str]:
    """Return (success, message) for YouTube/yt-dlp status."""
    try:
        links = get_youtube_links("test", limit=1)
        return True, "OK" if links else "No results (check network)"
    except Exception as e:
        return False, str(e)[:80]
