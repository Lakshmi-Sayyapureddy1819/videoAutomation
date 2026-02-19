"""
Use OpenAI to turn a user script/prompt into optimized YouTube search queries
so we find exactly the right videos (cinematic, 4K, etc.).
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def tune_script_for_search(user_script: str, num_queries: int = 5, model: str = "gpt-4o-mini") -> list[str]:
    """
    Send user script to OpenAI; return a list of optimized YouTube search queries
    (cinematic, 4K, high quality) to find the best matching footage.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
        return [user_script.strip()] if user_script.strip() else []

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a video researcher. Given a script or scene description, output ONLY a JSON array of "
                        "YouTube search query strings that would find the best, highest-quality footage. "
                        "Prefer queries that include: 4K, cinematic, drone, POV, no commentary, stock footage, "
                        "when relevant. Return 1–5 short search phrases. Output nothing but the JSON array, e.g. "
                        '["query1", "query2"].'
                    ),
                },
                {"role": "user", "content": user_script[:4000]},
            ],
            max_tokens=300,
        )
        text = (response.choices[0].message.content or "").strip()
        # Parse JSON array from response (may be wrapped in markdown code block)
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        import json
        queries = json.loads(text)
        if isinstance(queries, list) and queries:
            return [str(q).strip() for q in queries[:num_queries] if q]
    except Exception:
        pass
    return [user_script.strip()] if user_script.strip() else []


def generate_narration_script(user_prompt: str, target_duration_min: int = 10, model: str = "gpt-4o-mini") -> str:
    """
    Ask OpenAI to write a short narration script for the video (for voiceover and subtitles).
    User can edit this before generating the video.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
        return user_prompt.strip() if user_prompt.strip() else "No script."
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a documentary narrator. Write a single, engaging narration script "
                        f"for a {target_duration_min}-minute video. Use clear short sentences. "
                        "Output ONLY the script text, no titles or instructions. "
                        "Keep it suitable for voiceover and on-screen subtitles."
                    ),
                },
                {"role": "user", "content": user_prompt[:3000]},
            ],
            max_tokens=1500,
        )
        text = (response.choices[0].message.content or "").strip()
        return text if text else user_prompt.strip()
    except Exception:
        return user_prompt.strip()


def check_openai() -> tuple[bool, str]:
    """Return (success, message) for OpenAI API status."""
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
        return False, "Not configured (set OPENAI_API_KEY in .env)"
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}], max_tokens=5)
        return True, "OK"
    except Exception as e:
        return False, str(e)[:80]
