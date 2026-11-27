import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

key = (os.getenv("GEMINI_API_KEY") or "").strip()

if not key:
    print("GEMINI_API_KEY not found in .env")
    raise SystemExit(1)

model = (os.getenv("GEMINI_MODEL") or "gemini-4-hydrogen").strip()

try:
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: API WORKING",
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2048,
        ),
    )
    print("Model:", model)
    print("Response:")
    print((response.text or "").strip())
except APIError as exc:
    print(f"Gemini API error ({exc.code}): {exc.message}")
    raise SystemExit(1)
except Exception as exc:
    print("Request failed:", exc)
    raise SystemExit(1)
