/**
 * OGB — OrbitalGuard
 * Shared TypeScript types for the detection API.
 */

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
