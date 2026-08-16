"""
OGB — OrbitalGuard
Pydantic models for AI Copilot chat.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CopilotMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class DetectionHistorySummary(BaseModel):
    """Compact summary of a past detection for session history context."""
    class_name: str
    confidence: float
    timestamp: str


class CopilotRequest(BaseModel):
    """
    Operator sends a question plus structured context (detection JSON,
    orbital JSON, risk JSON).  The copilot must never receive raw images
    or TLEs — only structured outputs from the vision and orbital pipelines.
    """
    message: str
    history: List[CopilotMessage] = []
    detection_context: Optional[dict] = None    # DetectionResponse dict
    orbital_context: Optional[dict] = None      # OrbitalState dict (V2+)
    risk_context: Optional[dict] = None         # RiskAssessment dict (V2+)
    session_history: List[DetectionHistorySummary] = []  # compact prior detections


class ToolCallRecord(BaseModel):
    """Record of a single function-calling tool invocation by the model."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]


class CopilotResponse(BaseModel):
    reply: str
    provider: str
    model: str
    # Non-empty when the model used function-calling to produce this answer
    tool_calls: List[ToolCallRecord] = []
