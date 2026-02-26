import os
import sys
from cloud_worker import rank_on_cloud

# Setup paths to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # --- This script is now a test for the new Cloud Ranking Engine ---
    
    # Example candidates from Pexels/Pixabay
    # In a real run, these would be fetched by the stock_retriever
    candidate_videos = [
        {
            "source": "pexels",
            "id": "1448721",
            "url": "https://videos.pexels.com/video-files/1448721/1448721-hd_1280_720_25fps.mp4",
            "duration": 15,
        },
        {
            "source": "pexels",
            "id": "854253",
            "url": "https://videos.pexels.com/video-files/854253/854253-hd_1280_720_25fps.mp4",
            "duration": 20,
        },
        {
            "source": "pixabay",
            "id": "19986",
            "url": "https://cdn.pixabay.com/video/2018/11/22/19986-298726799_large.mp4",
            "duration": 10,
        }
    ]
    
    prompt = "A person walking on a beach during sunset"
    
    print(f"📝 Prompt: {prompt}")
    print(f"☁️ Sending {len(candidate_videos)} candidates to RunPod for ranking...")

    try:
        ranked_results = rank_on_cloud(candidates=candidate_videos, prompt=prompt)
        
        if ranked_results:
            print("\n--- ✅ Cloud Ranking Results ---")
            for i, result in enumerate(ranked_results):
                print(f"#{i+1}: {result['source']}:{result['id']} (Score: {result.get('score', 0):.4f})")
            
            print(f"\n🏆 Best Match: {ranked_results[0]['url']}")
        else:
            print("⚠️ Cloud ranking failed or returned no results.")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()