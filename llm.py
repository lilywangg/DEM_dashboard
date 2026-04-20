"""
llm.py — LLM calling logic (Gemini API) and citation counting
"""

from __future__ import annotations
import hashlib, os, re
from pathlib import Path

import diskcache

# ── Gemini cache ───────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent
_gemini_cache = diskcache.Cache(str(BASE / ".gemini_cache"))
_GEMINI_MODEL = "gemini-2.5-flash"


def _call_gemini(prompt: str) -> str:
    """
    Query Gemini 2.5 Flash API. Caches responses by prompt hash so repeat queries are free. Retries up to 3x on transient errors with exponential backoff.
    """
    key = hashlib.md5(prompt.encode()).hexdigest()
    if key in _gemini_cache:
        return _gemini_cache[key]

    import time as _t
    last_err = ""
    for attempt in range(3):
        try:
            from google import genai as _genai
            client   = _genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
            response = client.models.generate_content(
                model    = _GEMINI_MODEL,
                contents = prompt,
                config   = {"temperature": 0.0},
            )
            text = "".join(
                p.text for p in response.candidates[0].content.parts
                if hasattr(p, "text")
            ).strip()
            _gemini_cache[key] = text
            return text
        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                _t.sleep(3 * (attempt + 1))  # 3s, 6s backoff

    text = (
        f"⚠  Gemini API error after 3 attempts: {last_err}\n\n"
        "Ensure GEMINI_API_KEY is set in .env and `pip install google-genai` is done."
    )
    _gemini_cache[key] = text
    return text


def _count_citations(text: str) -> int:
    """
    Count grounding citations in LLM output. Looks for: E-NNN, [Evidence:, [Stats:, [Context:
    """
    return len(re.findall(r'E-\d+|\[Evidence:|\[Stats:|\[Context:', text))