/**
 * OGB — OrbitalGuard
 * API client — talks to the FastAPI backend.
 */

import type { CopilotRequest, CopilotResult, DetectResult } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// POST /api/v1/copilot
// ---------------------------------------------------------------------------

export async function copilotChat(req: CopilotRequest): Promise<CopilotResult> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/copilot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch {
    return {
      ok: false,
      status: 0,
      message:
        "Cannot reach the OGB backend. Is it running? Check NEXT_PUBLIC_API_URL.",
    };
  }

  if (!response.ok) {
    let message = `Copilot backend returned ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) message = body.detail;
    } catch {
      // ignore JSON parse failure
    }
    return { ok: false, status: response.status, message };
  }

  const data = await response.json();
  return { ok: true, data };
}

// ---------------------------------------------------------------------------
// POST /api/v1/detect
// ---------------------------------------------------------------------------

export async function detectObjects(file: File): Promise<DetectResult> {
  const form = new FormData();
  form.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/detect`, {
      method: "POST",
      body: form,
    });
  } catch (err) {
    return {
      ok: false,
      status: 0,
      message: "Cannot reach the OGB backend. Is it running on port 8000?",
    };
  }

  if (!response.ok) {
    let message = `Server returned ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) message = body.detail;
    } catch {
      // ignore JSON parse failure
    }
    return { ok: false, status: response.status, message };
  }

  const data = await response.json();
  return { ok: true, data };
}
