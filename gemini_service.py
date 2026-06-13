from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are TruthGuard AI, an educational fake-news detection assistant for students.
Analyze the supplied news content carefully. Do not claim certainty unless evidence is strong.
Return only valid JSON with these keys:
label: one of Real, Suspicious, Potentially Fake
credibility_score: integer 0-100
summary: concise 2-3 sentence student-friendly summary
reasons: list of 3-5 reasons for the label
red_flags: list of warning signs
verification_steps: list of practical steps a student can take to verify the news
trustworthy_rewrite: one neutral sentence rewriting the claim responsibly
"""

def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("Gemini did not return JSON")
    return json.loads(match.group(0))

def analyze_with_gemini(article_text: str, api_key: str | None = None, model_name: str | None = None) -> dict[str, Any]:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    import google.generativeai as genai

    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    prompt = f"{SYSTEM_PROMPT}\n\nNEWS CONTENT:\n{article_text[:9000]}"
    response = model.generate_content(prompt)
    data = _extract_json(response.text or "{}")
    data["provider"] = "Google Gemini"
    return data
