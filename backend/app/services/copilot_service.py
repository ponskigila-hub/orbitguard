"""
OGB — OrbitalGuard
Provider-agnostic AI Copilot service.

The copilot receives ONLY structured JSON from the vision and orbital
pipelines — never raw images or TLEs.  It must never invent orbital values.

Function-calling tools
──────────────────────
calculate_risk_score  — wraps the real risk_service.calculate_risk_full().
                        The LLM may call this when the operator supplies
                        d_min_km, v_rel_km_s, and delta_t_epoch_days in chat.
                        The result is passed back to the model; only then
                        may the model report a numeric risk score.

propagate_tle         — NOT IMPLEMENTED in V1. Stubbed with a clear error.
                        Full SGP4 propagation requires the V2 orbital pipeline
                        (not yet built). The model will inform the operator.
"""
from __future__ import annotations

import json
import os
from typing import Any

from app.core.config import (
    CLASS_NAMES,
    COPILOT_MAX_TOKENS,
    COPILOT_MODEL,
    COPILOT_PROVIDER,
    HARD_BODY_RADIUS_KM,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_LOW_CONFIDENCE_THRESHOLD,
)
from app.models.copilot import CopilotRequest, CopilotResponse, ToolCallRecord
from app.services.risk_service import calculate_risk_full

# ---------------------------------------------------------------------------
# Tool definitions — declared once, shared between providers
# ---------------------------------------------------------------------------

# Schema for Gemini's function-calling declaration
_TOOL_DECLARATIONS = [
    {
        "name": "calculate_risk_score",
        "description": (
            "Calculate the OGB risk priority score using the formula "
            "min(1, (R/d_min) * log10(v_rel+1) * exp(-Δt/7)). "
            "Call this ONLY when the operator has supplied numeric values for "
            "min_separation_km, relative_velocity_km_s, and tle_age_days. "
            "Never call it with invented or guessed numbers — ask the operator "
            "for any missing parameters first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "min_separation_km": {
                    "type": "number",
                    "description": (
                        "Minimum separation distance between the objects in km. "
                        "Must come from the operator or orbital data, not guessed."
                    ),
                },
                "relative_velocity_km_s": {
                    "type": "number",
                    "description": "Relative velocity between the objects in km/s.",
                },
                "tle_age_days": {
                    "type": "number",
                    "description": (
                        "Age of the TLE epoch in days. Use 0 for a freshly "
                        "downloaded TLE. Default is 0 if not specified by operator."
                    ),
                },
                "hard_body_radius_km": {
                    "type": "number",
                    "description": (
                        f"Hard-body radius of the object in km. "
                        f"Defaults to {HARD_BODY_RADIUS_KM} km if not specified."
                    ),
                },
            },
            "required": ["min_separation_km", "relative_velocity_km_s", "tle_age_days"],
        },
    },
    {
        "name": "propagate_tle",
        "description": (
            "Propagate a TLE (Two-Line Element set) to a given timestamp using "
            "SGP4 to obtain position and velocity vectors. "
            "NOTE: This tool is NOT available in V1 — the SGP4 orbital pipeline "
            "has not been implemented yet. Calling it will return an error. "
            "Inform the operator and ask them to supply d_min and v_rel directly "
            "if they want a risk calculation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tle_line1": {"type": "string", "description": "TLE line 1"},
                "tle_line2": {"type": "string", "description": "TLE line 2"},
                "timestamp_utc": {
                    "type": "string",
                    "description": "ISO-8601 UTC timestamp to propagate to",
                },
            },
            "required": ["tle_line1", "tle_line2", "timestamp_utc"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor — the ONLY place where numeric results are produced
# ---------------------------------------------------------------------------

def _execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a named tool with the arguments extracted by the model.
    Returns a structured dict that is sent back to the model as a tool result.
    """
    if name == "calculate_risk_score":
        hard_body_radius_km = float(args.get("hard_body_radius_km", HARD_BODY_RADIUS_KM))
        d_min_km = float(args["min_separation_km"])
        v_rel_km_s = float(args["relative_velocity_km_s"])
        delta_t_days = float(args["tle_age_days"])
        return calculate_risk_full(
            hard_body_radius_km=hard_body_radius_km,
            d_min_km=d_min_km,
            v_rel_km_s=v_rel_km_s,
            delta_t_epoch_days=delta_t_days,
        )

    if name == "propagate_tle":
        return {
            "error": "propagate_tle is not available in V1.",
            "reason": (
                "The SGP4 orbital pipeline has not been implemented yet. "
                "Please supply d_min_km, v_rel_km_s, and tle_age_days directly "
                "from your own orbital analysis software, and I can run the "
                "risk calculation from those numbers."
            ),
            "status": "NOT_IMPLEMENTED",
        }

    return {"error": f"Unknown tool '{name}'", "status": "ERROR"}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
You are OGB Copilot, the AI decision-support assistant for OrbitalGuard (OGB), \
a spacecraft situational-awareness system.

═══════════════════════════════════════════════════
HARD RULES — never violate these under any circumstances
═══════════════════════════════════════════════════
1. NEVER invent orbital data. You must never state a distance, velocity, \
Time of Closest Approach (TCA), or collision probability that did not come \
from a tool call result in this turn. If orbital_context is null or absent, \
say: "No orbital data is available in this session (V2 pipeline not active)."
2. You may report a risk score ONLY as the direct output of the \
`calculate_risk_score` tool call. Never output a risk number that you computed \
yourself or guessed — use the tool, return its exact result.
3. When the operator provides partial orbital data (e.g. only d_min), ask \
conversationally for the missing parameters before calling the tool. List \
exactly what is missing: min_separation_km, relative_velocity_km_s, tle_age_days.
4. If the operator asks about risk with NO numbers at all, do not silently decline \
or make up a plausible answer. Instead tell them exactly what you need: \
"To calculate the risk score I need: (1) minimum separation in km, \
(2) relative velocity in km/s, (3) TLE age in days (or 0 for a fresh TLE)."
5. Visual ≠ orbital. A visual detection from the camera is not the same as a \
tracked orbital close approach. Do not conflate them.
6. OGB is decision-support only. You can detect, analyse, calculate (via tools), \
explain, prioritise, and recommend. You cannot control spacecraft or execute \
manoeuvres. The human operator makes all decisions.
7. If a field is null, missing, or not in the provided context, say so explicitly.

═══════════════════════════════════════════════════
WHAT YOU CAN AND SHOULD DO
═══════════════════════════════════════════════════
A. EXPLAIN OBJECT CLASSES — Use your general knowledge to explain what a detected \
class *is* (e.g. "CHEOPS is an ESA exoplanet-characterization satellite"). \
Clearly separate general knowledge from detection-specific data. Frame as: \
"Based on general knowledge: [explanation]. Based on this detection: [data]."

B. EXPLAIN CONFIDENCE — Use confidence_advisory from context. If "LOW", flag it \
proactively: "Confidence is below the 0.50 threshold — recommend visual \
verification before acting."

C. CALCULATE RISK SCORES — When the operator supplies the required orbital \
numbers (min separation, relative velocity, TLE age), call `calculate_risk_score`. \
Report the exact result from the tool. Frame it: "Using the risk engine: \
score=[X], category=[Y]. This is a priority indicator, not a collision probability."

D. ANSWER CAPABILITY QUESTIONS — You know these facts:
  • Detector: YOLOv8n, {len(CLASS_NAMES)} classes, mAP@50=0.8156
  • Confidence threshold: {YOLO_CONFIDENCE_THRESHOLD} (advisory: {YOLO_LOW_CONFIDENCE_THRESHOLD})
  • Classes: {', '.join(CLASS_NAMES)}
  • Risk formula: min(1, (R/d_min) * log10(v_rel+1) * exp(-Δt/7))
  • Default hard-body radius: {HARD_BODY_RADIUS_KM} km
  • SGP4/TLE propagation: NOT AVAILABLE in V1 — supply d_min and v_rel directly.

E. FRAME RISK IN PLAIN LANGUAGE — After a tool call: explain the category \
(LOW/MEDIUM/HIGH/CRITICAL), what it means operationally, and suggest next steps \
while making clear the operator decides.

═══════════════════════════════════════════════════
RESPONSE STYLE
═══════════════════════════════════════════════════
Talk like a knowledgeable colleague explaining something out loud, not like a \
form being filled out. Specific rules:

- DEFAULT: flowing conversational sentences. For a single detection with one or \
two facts, write a natural sentence — not a bulleted spec sheet. \
Example: "I found one object — a Proba-3 CSC (`proba_3_csc`) — in the lower-centre \
of the frame, taking up about 24% of the image. Confidence is 57.9%, which is fine, \
though this class has historically been trickier for the detector (66.6% mAP@50 in \
testing vs. 81.6% overall), so worth a second look if this detection matters."
- USE LISTS only when structure genuinely helps: multiple distinct detections, \
multi-step recommendations, or capability comparisons. Never use a bullet list for a \
single object's basic facts.
- USE BOLD only to highlight something an operator needs to act on — a LOW-confidence \
flag, a CRITICAL risk category, a missing data warning. Not as section headers on \
every response.
- Lead with the operationally relevant fact first.
- Be concise. Don't pad responses. One tight paragraph beats four short bullets.
- Never hedge with "I think" — either you have the data or you don't, and you say which.
- Use markdown naturally (backticks for class names, bold for warnings), knowing it \
will render properly in the chat UI.
"""


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------

def _build_context_block(request: CopilotRequest) -> str:
    """Serialise available pipeline outputs into the prompt context."""
    parts: list[str] = []
    if request.detection_context:
        det = request.detection_context
        if isinstance(det, dict) and det.get("copilot_context"):
            parts.append(
                "=== VISION DETECTION — ENRICHED CONTEXT ===\n"
                + json.dumps(det["copilot_context"], indent=2)
            )
        else:
            parts.append(
                "=== VISION DETECTION (structured output) ===\n"
                + json.dumps(det, indent=2)
            )
    if request.orbital_context:
        parts.append(
            "=== ORBITAL STATE (structured output) ===\n"
            + json.dumps(request.orbital_context, indent=2)
        )
    if request.risk_context:
        parts.append(
            "=== RISK ASSESSMENT (structured output) ===\n"
            + json.dumps(request.risk_context, indent=2)
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini provider — with function-calling agentic loop
# ---------------------------------------------------------------------------

def _call_gemini(
    messages: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
) -> tuple[str, list[ToolCallRecord]]:
    """
    Call Gemini with function-calling support.

    Returns (reply_text, tool_calls_made).

    The agentic loop:
    1. Send messages + tool declarations to Gemini.
    2. If the model returns a function_call part, execute the real tool.
    3. Append the tool result as a 'function' role message.
    4. Send again; repeat until the model returns a text response.
    5. Return the final text + a record of every tool invocation.
    """
    try:
        import google.generativeai as genai  # type: ignore
        from google.generativeai.types import FunctionDeclaration, Tool  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)

    # Build tool declarations from our schema dicts
    tool_decls = [
        FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        )
        for t in _TOOL_DECLARATIONS
    ]
    tools = [Tool(function_declarations=tool_decls)]

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
        tools=tools,
    )

    # Convert message list to Gemini history + last user turn
    history = []
    last_user_message = ""
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        if msg == messages[-1] and role == "user":
            last_user_message = msg["content"]
        else:
            history.append({"role": role, "parts": [msg["content"]]})

    chat = gemini_model.start_chat(history=history)
    tool_calls_made: list[ToolCallRecord] = []

    # Agentic loop — at most 5 tool calls per turn to prevent runaway loops
    for _iteration in range(5):
        response = chat.send_message(last_user_message)
        candidate = response.candidates[0]

        # Collect any function_call parts from this response
        fc_parts = [
            part for part in candidate.content.parts
            if hasattr(part, "function_call") and part.function_call.name
        ]

        if not fc_parts:
            # Model returned a plain text response — done
            return response.text, tool_calls_made

        # Execute all requested tools and build the tool-result message
        tool_results = []
        for part in fc_parts:
            fc = part.function_call
            args = dict(fc.args)
            result = _execute_tool(fc.name, args)
            tool_calls_made.append(
                ToolCallRecord(tool_name=fc.name, arguments=args, result=result)
            )
            tool_results.append(
                {
                    "function_response": {
                        "name": fc.name,
                        "response": result,
                    }
                }
            )

        # Feed the tool results back to the model
        last_user_message = tool_results  # type: ignore[assignment]

    # Safety exit — should not be reached in normal operation
    return "Unable to complete the calculation after multiple tool calls. Please try rephrasing your request.", tool_calls_made


_PROVIDERS: dict[str, Any] = {
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
            "\n\nCurrent mission context (use this data, do not invent values):\n"
            + context_block
        )

    messages: list[dict] = [
        {"role": msg.role, "content": msg.content}
        for msg in request.history
        if msg.role in ("user", "assistant")
    ]
    messages.append({"role": "user", "content": request.message})

    call_fn = _PROVIDERS[provider_name]
    reply, tool_calls = call_fn(
        messages=messages,
        system_prompt=system_with_context,
        model=COPILOT_MODEL,
        max_tokens=COPILOT_MAX_TOKENS,
    )

    return CopilotResponse(
        reply=reply,
        provider=provider_name,
        model=COPILOT_MODEL,
        tool_calls=tool_calls,
    )
