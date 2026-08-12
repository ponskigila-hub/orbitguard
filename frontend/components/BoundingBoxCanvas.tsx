/**
 * OGB — OrbitalGuard
 * BoundingBoxCanvas — draws detections over the uploaded image.
 *
 * Uses a <canvas> overlaid on top of the <img> at the same rendered size.
 * Boxes are drawn in the detected class colour with label + confidence.
 */
"use client";

import { useEffect, useRef } from "react";
import type { Detection } from "@/lib/types";

/** Deterministic colour per class index — mission-control palette */
const CLASS_COLORS = [
  "#3b82f6", // 0 cheops          — blue
  "#ef4444", // 1 debris           — red
  "#f97316", // 2 double_start     — orange
  "#a855f7", // 3 earth_obs_sat_1  — purple
  "#06b6d4", // 4 lisa_pathfinder  — cyan
  "#84cc16", // 5 proba_2          — lime
  "#eab308", // 6 proba_3_csc      — yellow
  "#f43f5e", // 7 proba_3_ocs      — rose
  "#10b981", // 8 smart_1          — emerald
  "#6366f1", // 9 soho             — indigo
  "#ec4899", // 10 xmm_newton      — pink
];

function getColor(classId: number): string {
  return CLASS_COLORS[classId % CLASS_COLORS.length];
}

interface Props {
  imageSrc: string;
  detections: Detection[];
  naturalWidth: number;
  naturalHeight: number;
}

export default function BoundingBoxCanvas({
  imageSrc,
  detections,
  naturalWidth,
  naturalHeight,
}: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  /** Re-draw boxes whenever detections or image dimensions change. */
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || naturalWidth === 0 || naturalHeight === 0) return;

    const draw = () => {
      const renderedW = img.clientWidth;
      const renderedH = img.clientHeight;
      canvas.width = renderedW;
      canvas.height = renderedH;

      const scaleX = renderedW / naturalWidth;
      const scaleY = renderedH / naturalHeight;

      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, renderedW, renderedH);

      for (const det of detections) {
        const x1 = (det.x1_px ?? 0) * scaleX;
        const y1 = (det.y1_px ?? 0) * scaleY;
        const x2 = (det.x2_px ?? naturalWidth) * scaleX;
        const y2 = (det.y2_px ?? naturalHeight) * scaleY;
        const w = x2 - x1;
        const h = y2 - y1;

        const color = getColor(det.class_id);
        const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, w, h);

        // Label background
        ctx.font = "bold 11px 'JetBrains Mono', monospace, sans-serif";
        const textW = ctx.measureText(label).width + 8;
        const labelY = y1 > 18 ? y1 - 2 : y1 + h + 14;
        ctx.fillStyle = color;
        ctx.fillRect(x1, labelY - 13, textW, 16);

        // Label text
        ctx.fillStyle = "#000000";
        ctx.fillText(label, x1 + 4, labelY);
      }
    };

    if (img.complete) {
      draw();
    } else {
      img.onload = draw;
    }

    // Redraw on window resize
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [imageSrc, detections, naturalWidth, naturalHeight]);

  return (
    <div className="relative w-full">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        src={imageSrc}
        alt="Spacecraft camera feed"
        className="w-full rounded border border-[#2d3748] block"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />
    </div>
  );
}
