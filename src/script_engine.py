"""
AI writer: turn a short prompt into a 10-scene visual plan for a documentary.
Uses OpenAI GPT-4o (OPENAI_API_KEY in .env).
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def expand_script(user_prompt: str, num_scenes: int = 10, model: str = "gpt-4o") -> list[str]:
    """Turns a short prompt into a list of distinct visual scene descriptions for a ~10-minute video."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or str(api_key).strip().startswith("your_"):
        return [user_prompt.strip()] if user_prompt.strip() else []

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a documentary director. Expand the prompt into exactly "
                        f"{num_scenes} distinct visual scene descriptions for a 10-minute video. "
                        "Output one scene per line. Each line should be a short search-friendly description "
                        "(e.g. 'NASA rocket launch 4K', 'astronaut walking on moon'). No numbering or bullets."
                    ),
                },
                {"role": "user", "content": (user_prompt or "")[:3000]},
            ],
            max_tokens=1500,
        )
        content = (response.choices[0].message.content or "").strip()
        scenes = [s.strip() for s in content.split("\n") if s.strip() and len(s.strip()) > 10]
        return scenes[:num_scenes] if scenes else [user_prompt.strip()]
    except Exception:
        return [user_prompt.strip()] if user_prompt.strip() else []
