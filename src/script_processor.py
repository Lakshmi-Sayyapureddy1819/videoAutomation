import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_for_visuals(narration_chunk):
    """
    Converts abstract narration into concrete, searchable visual prompts.
    """
    prompt = f"""
    You are a professional documentary cinematographer. 
    Convert the following narration into 3 specific, highly-searchable visual keywords for stock footage APIs like Pexels.
    Focus on lighting, era, and action. Avoid abstract words.
    
    Narration: "{narration_chunk}"
    
    Format: keyword1, keyword2, keyword3
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo for cost saving
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ OpenAI Rewriter failed: {e}")
        return narration_chunk # Fallback to original if API fails

if __name__ == "__main__":
    test_text = "The weight of the industrial revolution changed how humans lived forever."
    print(f"Original: {test_text}")
    print(f"Optimized Keywords: {rewrite_for_visuals(test_text)}")