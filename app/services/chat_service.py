import httpx
from app.prompts.templates import get_prompt
import os

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "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"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

MODEL = "deepseek-chat"


async def get_chat_response(user_prompt: str, chat_type: str):
    system_prompt = get_prompt(chat_type)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(API_URL, headers=HEADERS, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
