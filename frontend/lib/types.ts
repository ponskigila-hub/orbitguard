/**
 * OGB — OrbitalGuard
 * Shared TypeScript types for the detection and copilot APIs.
 */

// ---------------------------------------------------------------------------
// Copilot
// ---------------------------------------------------------------------------

export interface CopilotMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface CopilotRequest {
  message: string;
  history?: CopilotMessage[];
  detection_context?: DetectionResponse;
  orbital_context?: Record<string, unknown>; // V2+
  risk_context?: Record<string, unknown>;    // V2+
}

export interface CopilotResponse {
  reply: string;
  provider: string;
  model: string;
}

export type CopilotResult =
  | { ok: true; data: CopilotResponse }
  | { ok: false; status: number; message: string };

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

export interface BoundingBox {
  x_center: number;
  y_center: number;
  width: number;
  height: number;
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bounding_box: BoundingBox;
  x1_px: number | null;
  y1_px: number | null;
  x2_px: number | null;
  y2_px: number | null;
}

export interface DetectionResponse {
  image_width: number;
  image_height: number;
  detections: Detection[];
  detection_count: number;
  model_version: string;
  inference_latency_ms: number | null;
  summary: string | null;
}

export interface ApiError {
  detail: string;
}

export type DetectResult =
  | { ok: true; data: DetectionResponse }
  | { ok: false; status: number; message: string };
