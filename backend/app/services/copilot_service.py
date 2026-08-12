"""
OGB — OrbitalGuard
Provider-agnostic AI Copilot service.

The copilot receives ONLY structured JSON from the vision and orbital
pipelines — never raw images or TLEs.  It must never invent orbital values.
"""
from __future__ import annotations

import os
from typing import Optional

from app.core.config import (
    COPILOT_MAX_TOKENS,
    COPILOT_MODEL,
    COPILOT_PROVIDER,
)
from app.models.copilot import CopilotMessage, CopilotRequest, CopilotResponse

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are OGB Copilot, the AI decision-support assistant for the OrbitalGuard \
mission-control system.

CRITICAL RULES — follow without exception:
1. You receive structured JSON summaries from the vision and orbital pipelines. \
   You must NEVER calculate orbital mechanics, invent distances, velocities, \
   Time of Closest Approach (TCA), or any other orbital value. If a field is \
   null or missing, say so explicitly — do not guess.
2. Keep visual and orbital outputs distinct. A visual detection means "an object \
   was seen in an image." An orbital close-approach means "a tracked object has a \
   projected close approach." These are NOT the same thing unless a correlation \
   step has confirmed it — and no such step exists in the current system version.
3. OGB is decision-support only. You can detect, analyze, explain, prioritize, \
   and recommend. You CANNOT control spacecraft, execute maneuvers, or make \
   autonomous decisions. Always make clear that the human operator decides.
4. Be concise, professional, and precise. Mission-control tone. No speculation.
5. If you do not have enough data to answer reliably, say so.
"""


def _build_context_block(request: CopilotRequest) -> str:
    """Serialise available pipeline outputs into the prompt context."""
    parts: list[str] = []
    if request.detection_context:
        import json
        parts.append(
            "=== VISION DETECTION (structured output) ===\n"
            + json.dumps(request.detection_context, indent=2)
        )
    if request.orbital_context:
        import json
        parts.append(
            "=== ORBITAL STATE (structured output) ===\n"
            + json.dumps(request.orbital_context, indent=2)
        )
    if request.risk_context:
        import json
        parts.append(
            "=== RISK ASSESSMENT (structured output) ===\n"
            + json.dumps(request.risk_context, indent=2)
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_gemini(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
    )
    # Convert OpenAI-style message list to Gemini history + last user turn
    history = []
    last_user_message = ""
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        if msg == messages[-1] and role == "user":
            last_user_message = msg["content"]
        else:
            history.append({"role": role, "parts": [msg["content"]]})

    chat = gemini_model.start_chat(history=history)
    response = chat.send_message(last_user_message)
    return response.text


_PROVIDERS: dict[str, callable] = {
    "gemini": _call_gemini,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_copilot(request: CopilotRequest) -> CopilotResponse:
    """
    Invoke the configured AI copilot provider with the operator's message
    and any available structured pipeline context.
    """
    provider_name = COPILOT_PROVIDER
    if provider_name not in _PROVIDERS:
        raise RuntimeError(
            f"Unknown copilot provider '{provider_name}'. "
            f"Available: {list(_PROVIDERS.keys())}"
        )

    context_block = _build_context_block(request)
    system_with_context = _SYSTEM_PROMPT
    if context_block:
        system_with_context += (
            "\n\nCurrent mission context (use this, do not invent values):\n"
            + context_block
        )

    # Build message list from history + current message
    messages: list[dict] = [
        {"role": msg.role, "content": msg.content}
        for msg in request.history
        if msg.role in ("user", "assistant")
    ]
    messages.append({"role": "user", "content": request.message})

    call_fn = _PROVIDERS[provider_name]
    reply = call_fn(
        messages=messages,
        system_prompt=system_with_context,
        model=COPILOT_MODEL,
        max_tokens=COPILOT_MAX_TOKENS,
    )

    return CopilotResponse(
        reply=reply,
        provider=provider_name,
        model=COPILOT_MODEL,
    )
