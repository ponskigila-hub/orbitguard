/**
 * OGB — OrbitalGuard
 * MVP screen — Camera Analysis + AI Copilot side-by-side.
 *
 * Layout:
 *  ┌────────────────────────────────┬──────────────────┐
 *  │  Camera Analysis               │  AI Copilot      │
 *  │  Upload → /detect → bbox+table │  Chat panel      │
 *  └────────────────────────────────┴──────────────────┘
 */
"use client";

import { useState, useCallback } from "react";
import { Shield, RefreshCw } from "lucide-react";
import BoundingBoxCanvas from "@/components/BoundingBoxCanvas";
import DetectionSummary from "@/components/DetectionSummary";
import DropZone from "@/components/DropZone";
import StatusBar, { Status } from "@/components/StatusBar";
import CopilotPanel from "@/components/CopilotPanel";
import { detectObjects } from "@/lib/api";
import type { DetectionResponse } from "@/lib/types";

export default function MvpPage() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [status, setStatus] = useState<Status>({ type: "idle" });

  const handleFile = useCallback(async (file: File) => {
    const objectUrl = URL.createObjectURL(file);

    const img = new window.Image();
    img.src = objectUrl;
    await new Promise<void>((resolve) => {
      img.onload = () => {
        setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
        resolve();
      };
    });

    setImageSrc(objectUrl);
    setResult(null);
    setStatus({ type: "loading" });

    const res = await detectObjects(file);

    if (res.ok) {
      setResult(res.data);
      const n = res.data.detection_count;
      setStatus({
        type: "success",
        message: `${n} object${n !== 1 ? "s" : ""} detected · ${res.data.inference_latency_ms?.toFixed(0) ?? "—"} ms`,
      });
    } else {
      setStatus({ type: "error", message: res.message });
    }
  }, []);

  function reset() {
    setImageSrc(null);
    setResult(null);
    setStatus({ type: "idle" });
    setNaturalSize({ w: 0, h: 0 });
  }

  return (
    <div className="min-h-screen bg-[#060d18] text-[#e2e8f0] flex flex-col">
      {/* ── Top nav ──────────────────────────────────────────────────────── */}
      <header className="border-b border-[#1e2535] bg-[#060d18] flex-shrink-0">
        <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield size={20} className="text-[#3b82f6]" strokeWidth={1.5} />
            <span className="font-mono text-sm font-bold tracking-widest text-[#e2e8f0] uppercase">
              OGB
            </span>
            <span className="text-[#2d3748] font-mono text-sm">|</span>
            <span className="text-[#a0aec0] font-mono text-xs tracking-wider uppercase">
              Camera Analysis
            </span>
            <span className="text-[#2d3748] font-mono text-sm hidden sm:block">
              /
            </span>
            <span className="text-[#a0aec0] font-mono text-xs tracking-wider uppercase hidden sm:block">
              AI Copilot
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-[#0d1117] border border-[#2d3748] text-[#4a5568]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#3b82f6] animate-pulse" />
              LIVE
            </span>
          </div>
        </div>
      </header>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-screen-2xl w-full mx-auto px-6 py-6">
        {/* Two-column MVP layout: Camera Analysis | Copilot */}
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-6 items-start">

          {/* ══ LEFT COLUMN: Camera Analysis ════════════════════════════════ */}
          <div className="flex flex-col gap-4 min-w-0">
            {/* Section header */}
            <div>
              <h1 className="font-mono text-sm font-bold text-[#e2e8f0] tracking-widest uppercase">
                Spacecraft Camera Feed
              </h1>
              <p className="text-[#4a5568] text-xs font-mono mt-0.5">
                Upload an image to run YOLOv8n debris / object detection.
                Vision output only — orbital context requires V2 pipeline.
              </p>
            </div>

            {/* Status bar */}
            {status.type !== "idle" && (
              <StatusBar status={status} />
            )}

            <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
              {/* Image + bbox overlay */}
              <div className="flex flex-col gap-3">
                {imageSrc ? (
                  <>
                    <div className="relative">
                      <BoundingBoxCanvas
                        imageSrc={imageSrc}
                        detections={result?.detections ?? []}
                        naturalWidth={naturalSize.w || result?.image_width || 640}
                        naturalHeight={naturalSize.h || result?.image_height || 640}
                      />
                      {status.type === "loading" && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded">
                          <div className="flex flex-col items-center gap-3">
                            <div className="w-8 h-8 border-2 border-[#3b82f6] border-t-transparent rounded-full animate-spin" />
                            <span className="text-[#a0aec0] text-xs font-mono">
                              Running YOLOv8n…
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                    <button
                      onClick={reset}
                      className="flex items-center gap-2 self-start text-xs font-mono text-[#4a5568] hover:text-[#a0aec0] transition-colors"
                    >
                      <RefreshCw size={13} />
                      Analyse another image
                    </button>
                  </>
                ) : (
                  <DropZone onFile={handleFile} disabled={status.type === "loading"} />
                )}
              </div>

              {/* Detection sidebar: system info + summary + raw JSON */}
              <div className="flex flex-col gap-3">
                {/* System info */}
                <div className="rounded border border-[#2d3748] bg-[#0d1117] px-4 py-3">
                  <p className="text-[#4a5568] text-xs font-mono uppercase tracking-widest mb-3">
                    System
                  </p>
                  <div className="space-y-1.5">
                    <InfoRow label="Model" value="YOLOv8n" />
                    <InfoRow label="Classes" value="11" />
                    <InfoRow label="Input" value="640 × 640 px" />
                    <InfoRow label="Pipeline" value="Vision only (MVP)" />
                    <InfoRow
                      label="Backend"
                      value={
                        <span className="flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-[#3b82f6]" />
                          FastAPI
                        </span>
                      }
                    />
                  </div>
                </div>

                {result && (
                  <DetectionSummary
                    detections={result.detections}
                    latencyMs={result.inference_latency_ms}
                    modelVersion={result.model_version}
                  />
                )}

                {result && (
                  <details className="rounded border border-[#2d3748] bg-[#0d1117] overflow-hidden">
                    <summary className="px-4 py-2 text-xs font-mono text-[#4a5568] cursor-pointer hover:text-[#a0aec0] transition-colors select-none bg-[#161b27]">
                      Raw API response
                    </summary>
                    <pre className="px-4 py-3 text-[10px] font-mono text-[#718096] overflow-auto max-h-64 whitespace-pre-wrap">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            </div>
          </div>

          {/* ══ RIGHT COLUMN: AI Copilot ═════════════════════════════════════ */}
          {/* sticky so the panel stays in viewport while the left column scrolls */}
          <div className="flex flex-col xl:sticky xl:top-6" style={{ height: "calc(100vh - 7rem)" }}>
            {/* Section header */}
            <div className="mb-3 flex-shrink-0">
              <h2 className="font-mono text-sm font-bold text-[#e2e8f0] tracking-widest uppercase">
                AI Copilot
              </h2>
              <p className="text-[#4a5568] text-xs font-mono mt-0.5">
                Ask questions grounded in detection data. No orbital data in MVP — Copilot will say so explicitly.
              </p>
            </div>
            <div className="flex-1 min-h-0">
              <CopilotPanel detectionContext={result} />
            </div>
          </div>

        </div>
      </main>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="border-t border-[#1e2535] flex-shrink-0">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between">
          <p className="text-[#2d3748] text-xs font-mono">
            OGB v0.1.0 — IBM Bob AI Builders Challenge 2025
          </p>
          <p className="text-[#2d3748] text-xs font-mono">
            Decision-support only. Operator makes all decisions.
          </p>
        </div>
      </footer>
    </div>
  );
}

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[#4a5568] text-xs font-mono">{label}</span>
      <span className="text-[#a0aec0] text-xs font-mono">{value}</span>
    </div>
  );
}
