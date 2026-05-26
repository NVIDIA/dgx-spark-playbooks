"""Nemotron Mini client for recommendation explanations.

Talks to a local Ollama server (the one started by assets/setup.sh) using
the `nemotron-mini` model. Returns short natural-language explanations of
why a set of items was recommended for a user.

The serving stack matches the one configured by setup.sh and used by
assets/app.py (the web UI) — Ollama on port 11434, model `nemotron-mini`.
Both can be overridden with OLLAMA_URL and NEMOTRON_MODEL.
"""

import os
from typing import Optional

import requests

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
NEMOTRON_MODEL = os.environ.get('NEMOTRON_MODEL', 'nemotron-mini')


def build_explanation_prompt(user_history: list[dict], recommendations: list[dict]) -> str:
    """Build a prompt for Nemotron to output a strict single-sentence rationale.

    Output must match the schema:
        "Based on your <attr-1>, <attr-2>, and <attr-3>, these picks share
        <characteristic-1>, <characteristic-2>, and <characteristic-3>."

    Hard rules enforce no preamble, no greeting, no price, no generic phrasing.
    """
    history_titles = [h.get('title', 'Unknown') for h in user_history[-5:]]
    rec_titles = [r.get('title', 'Unknown') for r in recommendations]
    history_str = "\n".join(f"- {t}" for t in history_titles)
    recs_str = "\n".join(f"- {t}" for t in rec_titles)

    return (
        "Customer's recent dresses:\n"
        f"{history_str}\n\n"
        "Recommended dresses:\n"
        f"{recs_str}\n\n"
        "Output 3 numbered lines. Each line states a SPECIFIC style pattern "
        "observed across the customer's history and (optionally) shows how the "
        "recommendations match that pattern. The voice is impersonal — describe "
        "the pattern factually, do not address 'you' or 'your'.\n\n"
        "Two acceptable line formats:\n"
        "  Format A — observation only:\n"
        "    <history pattern>\n"
        "  Format B — observation + match:\n"
        "    <history pattern> → <how the recs match>\n\n"
        "Use the literal Unicode arrow → (NOT '->' or '=>' or hyphens) for the "
        "transition in Format B.\n\n"
        "Use SPECIFIC style values, not the abstract category word. Examples of "
        "allowed values: A-line, sheath, wrap, bodycon, fit-and-flare, maxi, midi, "
        "mini, shift, V-neck, scoop, halter, off-shoulder, square neck, flutter "
        "sleeve, cap sleeve, puff sleeve, short sleeve, lace, chiffon, satin, "
        "smocked, ruffled, tiered, pleated, floral, polka-dot, striped, tie-dye, "
        "color-block, solid.\n\n"
        "EXAMPLES of the format — these are SAMPLE bullets only. You MUST "
        "generate DIFFERENT content tailored to the actual items above:\n"
        "  [1] Short flowy A-line cuts are common across history → recs continue this\n"
        "  [2] V-neck and square-neck necklines dominate past picks\n"
        "  [3] Floral and ruffle detailing recur → 3 of 4 recs share both\n"
        "  [1] Maxi lengths appear repeatedly → recs lean short and midi\n"
        "  [2] Past purchases skew tie-dye → 2 picks feature similar prints\n"
        "  [3] Smocked bodices and elastic waists recurring → all 4 recs match\n"
        "  [1] Polka-dot and solid blacks recur in history → 3 of 4 recs share\n"
        "  [2] Halter and off-shoulder dominant in past picks\n"
        "  [3] Cap-sleeve and short-sleeve recur → recs match in 4 of 4\n\n"
        "Rules:\n"
        "  - Output EXACTLY 3 lines. Not 2, not 4 — exactly 3.\n"
        "  - Each line starts with [1], [2], or [3] respectively. Nothing else.\n"
        "  - Each line follows Format A (observation only) or Format B (observation → match).\n"
        "  - Mix the formats across the 3 lines for variety.\n"
        "  - Vary the wording per call — do NOT repeat the exact example bullets.\n"
        "  - When using Format B, use the Unicode arrow → (never '->' or '=>').\n"
        "  - 8-15 words per line. Concise and factual.\n"
        "  - NEVER use 'you' or 'your'. Stay impersonal.\n"
        "  - Identify patterns that GENUINELY appear in the customer's history.\n"
        "  - Use specific values (e.g. 'A-line' not 'silhouette').\n"
        "  - No greetings, no 'Sure', no 'Here', no preamble.\n"
        "  - DO NOT mention occasions/events/vibes — no 'casual', 'work', "
        "'cocktail', 'brunch', 'wedding', 'summer', 'evening', 'perfect for' etc.\n"
        "  - No vague phrases ('matches your style', 'fits your taste').\n"
        "  - NEVER wrap item names in square brackets [].\n"
        "  - DO NOT include item titles in the bullets (no 'on dresses like XYZ').\n"
        "  - DO NOT meta-reference the data — no 'in the customer's history and "
        "recommendations', 'across both collections', 'in customer history'.\n"
        "  - NEVER output words like 'Cut/silhouette:', 'Pairing verb:', "
        "'Trait A:' — those are prompt structure, not output.\n"
        "  - No trailing colons.\n"
        "  - No price, ratings, shopping logistics.\n"
        "  - Tailor every line to the actual items above — don't reuse the example traits.\n"
    )


def generate_explanation(
    user_history: list[dict],
    recommendations: list[dict],
    ollama_url: str = OLLAMA_URL,
    model: str = NEMOTRON_MODEL,
    timeout: float = 10.0,
) -> Optional[str]:
    """Generate a recommendation explanation via Ollama / Nemotron Mini."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise fashion recommendation assistant."},
            {"role": "user", "content": build_explanation_prompt(user_history, recommendations)},
        ],
        "stream": False,
        # 0.5 was too rigid — model locked to specific example bullets and
        # truncated to 2 lines. 0.65 keeps format adherence + enough variety
        # to vary content per call. num_predict caps output so the model
        # finishes all 3 bullets rather than stopping mid-stream.
        "options": {"temperature": 0.65, "top_p": 0.9, "num_predict": 300},
    }
    try:
        resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return None  # Ollama not running — caller falls back to template
    except Exception as e:
        return f"[Explanation unavailable: {e}]"


def get_explanation_or_fallback(
    user_history: list[dict],
    recommendations: list[dict],
    fallback: str,
) -> tuple[str, bool]:
    """Try Nemotron via Ollama; fall back to caller-provided template if unreachable.

    Returns (explanation, is_llm_generated).
    """
    explanation = generate_explanation(user_history, recommendations)
    if explanation is not None:
        return explanation, True
    return fallback, False


if __name__ == "__main__":
    history = [
        {"title": "Women's Floral Summer Maxi Dress", "price": 34.99},
        {"title": "Elegant Lace Cocktail Dress", "price": 52.00},
        {"title": "Casual V-Neck T-Shirt Dress", "price": 24.99},
    ]
    recs = [
        {"title": "Boho Beach Wrap Dress", "price": 29.99, "score": 0.85},
        {"title": "Vintage A-Line Swing Dress", "price": 39.99, "score": 0.82},
        {"title": "Sleeveless Midi Sundress", "price": 32.00, "score": 0.79},
    ]
    print(f"Calling {NEMOTRON_MODEL} at {OLLAMA_URL}...")
    explanation = generate_explanation(history, recs)
    if explanation:
        print(f"\nExplanation:\n{explanation}")
    else:
        print(f"\nOllama not reachable at {OLLAMA_URL}. Start it with `ollama serve`.")
