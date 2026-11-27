import os
from pathlib import Path
from dotenv import load_dotenv
import requests

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

key = os.getenv("OPENROUTER_API_KEY")

if not key:
    print("API key not found in .env")
    exit()

url = "https://openrouter.ai/api/v1/chat/completions"

payload = {
    "model": "nvidia/nemotron-3.5-lightning:free",
    "messages": [
        {"role": "user", "content": "Reply with exactly: API WORKING"}
    ],
    "max_tokens": 30
}

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    print("HTTP Status:", response.status_code)
    print("Response:")
    print(response.text)

except Exception as e:
    print("Request failed:", e)