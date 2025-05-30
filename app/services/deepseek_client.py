# app/services/deepseek_client.py
import re
from collections import deque

import httpx
from langdetect import detect

from app.core.config import settings
from app.prompts.templates import get_prompt
from app.services.food_info import get_food_info
from app.services.disease_matcher import (
    get_probable_diseases,
    get_disease_explanation,
)

# Greeting sets per language
GREETINGS = {
    "English": {"hi", "hello", "hey"},
    "French": {"salut", "bonjour", "coucou"},
}

# Responses per language
GREETING_RESPONSES = {
    "English": "Hi! How can I help you today?",
    "French": "Bonjour ! Comment puis-je vous aider aujourd’hui ?",
}

# In-memory chat history (swap for Redis in production)
_chat_history: dict[str, deque] = {}


def detect_language(text: str) -> str:
    """
    Detect language using langdetect. Defaults to English, handles French.
    """
    try:
        code = detect(text)
        return {"fr": "French"}.get(code[:2], "English")
    except Exception:
        return "English"


async def _handle_food(query: str, lang: str) -> str:
    """
    Lookup nutrition info for a food.
    """
    info = get_food_info(query)
    if info:
        return (
            f"**{info['name']}** per 100g:\n"
            f"- Calories: {info['calories']} kcal\n"
            f"- Fat: {info['fat_g']} g (Sat: {info['saturated_fat_g']} g)\n"
            f"- Carbs: {info['carbs_g']} g (Sugars: {info['sugars_g']} g)\n"
            f"- Protein: {info['protein_g']} g\n"
            f"- Fiber: {info['fiber_g']} g"
        )
    return {
        "English": f"Sorry, I don't have data on '{query}'. Try another food.",
        "French": f"Désolé, pas de données sur '{query}'. Essayez un autre aliment.",
    }.get(lang, "Sorry, no data found.")


def _handle_symptom(text: str) -> str:
    """
    Return top probable diseases for symptoms.
    """
    probable = get_probable_diseases(text)
    lines = [
        f"**{d['disease']} ({d['probability']})**\n- {d['reason']}" for d in probable
    ]
    return (
        "Based on your symptoms, the most probable conditions are:\n\n"
        + "\n".join(lines)
        + "\n\nThis is not a medical diagnosis. Please consult a professional."
    )


async def _handle_explore(raw: str, lang: str) -> str:
    """
    Provide a concise overview (max 5 sentences).
    """
    prompt = (
        f"Provide a concise (max 5 sentences) overview of '{raw}': definition, symptoms, "
        f"treatments, and red flags. Respond in {lang}."
    )
    async with httpx.AsyncClient(timeout=settings.chat_timeout) as client:
        resp = await client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a knowledgeable medical assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def get_response(session_id: str, user_input: str, chat_type: str) -> str:
    """
    Entry point: handles greetings, specialized modes, and generic chat.
    """
    text = user_input.strip()
    lang = detect_language(text)

    # Greeting
    if text.lower() in GREETINGS.get(lang, {}):
        return GREETING_RESPONSES.get(lang, GREETING_RESPONSES["English"])

    # Dispatch
    if chat_type == "food":
        return await _handle_food(text, lang)
    if chat_type == "symptom":
        return _handle_symptom(text)
    if chat_type == "explore":
        return await _handle_explore(text, lang)

    # Generic LLM chat via Deepseek
    system_prompt = f"{get_prompt(chat_type)}\nRespond in {lang}."
    history = _chat_history.setdefault(
        session_id, deque(maxlen=settings.chat_history_max)
    )
    history.append({"role": "user", "content": text})
    messages = [{"role": "system", "content": system_prompt}] + list(history)

    async with httpx.AsyncClient(timeout=settings.chat_timeout) as client:
        resp = await client.post(
            settings.deepseek_api_url,
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            json={"model": settings.model_name, "messages": messages},
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]

    history.append({"role": "assistant", "content": reply})
    return reply
