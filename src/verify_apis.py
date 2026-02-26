import os
import requests
from dotenv import load_dotenv

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False

# Load environment variables from .env file
load_dotenv()

def test_youtube_api():
    """Tests the YouTube Data API v3 key and quota."""
    print("--- 1. Testing YouTube Data API v3 ---")
    if not HAS_GOOGLE_API:
        print("🟡 SKIPPED: 'google-api-python-client' not installed.")
        return
        
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("🟡 SKIPPED: YOUTUBE_API_KEY not found in .env file.")
        return

    try:
        youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        request = youtube.search().list(q="test", part="id", maxResults=1, type="video")
        request.execute()
        print("✅ YouTube API: Connected & Key is valid.")
    except HttpError as e:
        if e.resp.status == 403:
            print("❌ YouTube API FAILED: Quota Exceeded or API not enabled.")
            print("   💡 Go to Google Cloud Console, find your key, and check the API usage dashboard.")
        else:
            print(f"❌ YouTube API FAILED: {e}")
    except Exception as e:
        print(f"❌ YouTube API FAILED with unexpected error: {e}")

def test_pexels_api():
    """Tests the Pexels API key."""
    print("\n--- 2. Testing Pexels API ---")
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("🟡 SKIPPED: PEXELS_API_KEY not found in .env file.")
        return

    headers = {"Authorization": api_key}
    url = "https://api.pexels.com/v1/curated?per_page=1"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            print("✅ Pexels API: Connected & Key is valid.")
        else:
            print(f"❌ Pexels API FAILED: Status {res.status_code} - {res.json().get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Pexels API FAILED with connection error: {e}")

def test_pixabay_api():
    """Tests the Pixabay API key."""
    print("\n--- 3. Testing Pixabay API ---")
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("🟡 SKIPPED: PIXABAY_API_KEY not found in .env file.")
        return
    
    url = f"https://pixabay.com/api/videos/?key={api_key}&q=test&per_page=1"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200 and 'hits' in res.json():
            print("✅ Pixabay API: Connected & Key is valid.")
        else:
            print(f"❌ Pixabay API FAILED: Status {res.status_code} - {res.text}")
    except Exception as e:
        print(f"❌ Pixabay API FAILED with connection error: {e}")

def test_coverr_api():
    """Tests the Coverr API key."""
    print("\n--- 4. Testing Coverr API ---")
    api_key = os.getenv("COVERR_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("🟡 SKIPPED: COVERR_API_KEY not found in .env file.")
        return
    
    url = f"https://api.coverr.co/videos?api_key={api_key}&query=test"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            print("✅ Coverr API: Connected & Key is valid.")
        else:
            print(f"❌ Coverr API FAILED: Status {res.status_code} - {res.text}")
    except Exception as e:
        print(f"❌ Coverr API FAILED with connection error: {e}")

if __name__ == "__main__":
    print("🚀 Running VidRush API Connectivity & Quota Test...")
    test_youtube_api()
    test_pexels_api()
    test_coverr_api()
    print("\nTest complete.")