/**
 * OGB — OrbitalGuard
 * Shared TypeScript types for the detection, copilot, and orbital APIs.
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

export interface ToolCallRecord {
  tool_name: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
}

export interface CopilotResponse {
  reply: string;
  provider: string;
  model: string;
  tool_calls: ToolCallRecord[];
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

// ---------------------------------------------------------------------------
// Orbital Intelligence (V2)
// ---------------------------------------------------------------------------

export interface OrbitalAnalyzeRequest {
  tle_line1: string;
  tle_line2: string;
  timestamp_utc?: string;
  target_tle_line1?: string;
  target_tle_line2?: string;
  conjunction_window_hours?: number;
}

export interface PropagatedState {
  ok: boolean;
  position_km?: number[];
  velocity_km_s?: number[];
  epoch_utc?: string;
  tle_age_days?: number;
  propagated_at_utc?: string;
  error?: string;
}

export interface ConjunctionResult {
  ok: boolean;
  tca_utc?: string;
  d_min_km?: number;
  v_rel_km_s?: number;
  position_sat1_km?: number[];
  position_sat2_km?: number[];
  tle_age_days_sat1?: number;
  tle_age_days_sat2?: number;
  risk?: Record<string, unknown>;
  coarse_samples?: number;
  window_hours?: number;
  error?: string;
}

export interface OrbitalAnalyzeResponse {
  propagated_state: PropagatedState;
  conjunction?: ConjunctionResult;
  analysis_type: string;
}

export type OrbitalResult =
  | { ok: true; data: OrbitalAnalyzeResponse }
  | { ok: false; status: number; message: string };

/** Derived type used in the Threat Center list. */
export interface ThreatEntry {
  id: string;                        // human label (e.g. "Object A")
  result: OrbitalAnalyzeResponse;
  risk_category?: string;            // from conjunction.risk.risk_category
  risk_score?: number;
  d_min_km?: number;
  tle_age_days?: number;
}

/** Maps risk category to Tailwind text colour class. */
export const RISK_COLOR: Record<string, string> = {
  CRITICAL: "text-red-400",
  HIGH:     "text-orange-400",
  MEDIUM:   "text-yellow-400",
  LOW:      "text-green-400",
};

/** Maps risk category to emoji indicator. */
export const RISK_EMOJI: Record<string, string> = {
  CRITICAL: "🔴",
  HIGH:     "🟠",
  MEDIUM:   "🟡",
  LOW:      "🟢",
};
