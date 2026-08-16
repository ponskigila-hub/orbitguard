/**
 * OGB — OrbitalGuard
 * MVP screen — Camera Analysis + AI Copilot side-by-side.
 *
 * Features:
 *  - Session detection history (in-memory, feature #1)
 *  - Skeleton loader + bbox animation (feature #2)
 *  - Confidence threshold slider (feature #3)
 *  - Export detection report: JSON + PDF (feature #4)
 *
 * Layout:
 *  ┌────────────────────────────────┬──────────────────┐
 *  │  Camera Analysis               │  AI Copilot      │
 *  │  Upload → /detect → bbox+table │  Chat panel      │
 *  └────────────────────────────────┴──────────────────┘
 */
"use client";

import { useState, useCallback, useRef } from "react";
import Link from "next/link";
import { Shield, RefreshCw, Satellite } from "lucide-react";
import BoundingBoxCanvas from "@/components/BoundingBoxCanvas";
import DetectionSummary from "@/components/DetectionSummary";
import DropZone from "@/components/DropZone";
import StatusBar, { Status } from "@/components/StatusBar";
import CopilotPanel from "@/components/CopilotPanel";
import ExportButton from "@/components/ExportButton";
import { detectObjects } from "@/lib/api";
import type { DetectionResponse, DetectionHistoryEntry } from "@/lib/types";

/** Simple incrementing ID for history entries. */
let _historySeq = 0;

export default function MvpPage() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [status, setStatus] = useState<Status>({ type: "idle" });

  // Feature #1 — session detection history (in-memory, cleared on page reload)
  const [detectionHistory, setDetectionHistory] = useState<DetectionHistoryEntry[]>([]);

  // Feature #3 — confidence threshold slider
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);

  const handleFile = useCallback(async (file: File) => {
    const objectUrl = URL.createObjectURL(file);

    const img = new window.Image();
    img.src = objectUrl;
    const nSize = await new Promise<{ w: number; h: number }>((resolve) => {
      img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    });

    setImageSrc(objectUrl);
    setNaturalSize(nSize);
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
      // Feature #1 — append to session history
      _historySeq += 1;
      setDetectionHistory((prev) => [
        ...prev,
        {
          id: `det-${_historySeq}`,
          timestamp: new Date().toISOString(),
          imageSrc: objectUrl,
          naturalSize: nSize,
          result: res.data,
        },
      ]);
    } else {
      setStatus({ type: "error", message: res.message });
    }
  }, []);

  function reset() {
    setImageSrc(null);
    setResult(null);
    setStatus({ type: "idle" });
    setNaturalSize({ w: 0, h: 0 });
    // Note: detectionHistory is intentionally NOT cleared on reset —
    // it is session memory, only cleared on full page reload.
  }

  /** Load a history entry back as the current context. */
  function loadHistoryEntry(entry: DetectionHistoryEntry) {
    setImageSrc(entry.imageSrc);
    setNaturalSize(entry.naturalSize);
    setResult(entry.result);
    setStatus({
      type: "success",
      message: `History: ${entry.result.detection_count} object${entry.result.detection_count !== 1 ? "s" : ""} · ${new Date(entry.timestamp).toLocaleTimeString()}`,
    });
  }

  // Filtered detections based on confidence threshold slider
  const filteredDetections = result
    ? result.detections.filter((d) => d.confidence >= confidenceThreshold)
    : [];

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
          <div className="flex items-center gap-3">
            <Link
              href="/orbital"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-[#0d1117] border border-[#2d4a7a] text-[#7aa2d4] hover:bg-[#1a2847] hover:text-[#c5d8f7] transition-colors"
            >
              <Satellite size={13} strokeWidth={1.5} />
              Orbital Intelligence (V2)
            </Link>
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

            {/* Feature #1 — Session history strip */}
            {detectionHistory.length > 0 && (
              <SessionHistoryStrip
                history={detectionHistory}
                currentImageSrc={imageSrc}
                onSelect={loadHistoryEntry}
              />
            )}

            <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
              {/* Image + bbox overlay */}
              <div className="flex flex-col gap-3">
                {status.type === "loading" && !imageSrc ? (
                  /* Feature #2 — Skeleton loader (no image yet, loading state) */
                  <SkeletonLoader />
                ) : imageSrc ? (
                  <>
                    <div className="relative">
                      <BoundingBoxCanvas
                        imageSrc={imageSrc}
                        detections={filteredDetections}
                        naturalWidth={naturalSize.w || result?.image_width || 640}
                        naturalHeight={naturalSize.h || result?.image_height || 640}
                        animate={status.type === "success"}
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

                {/* Feature #3 — Confidence threshold slider */}
                {result && (
                  <ConfidenceSlider
                    value={confidenceThreshold}
                    onChange={setConfidenceThreshold}
                    totalDetections={result.detections.length}
                    filteredCount={filteredDetections.length}
                    backendThreshold={0.25}
                  />
                )}

                {result && (
                  <DetectionSummary
                    detections={filteredDetections}
                    latencyMs={result.inference_latency_ms}
                    modelVersion={result.model_version}
                  />
                )}

                {/* Feature #4 — Export buttons */}
                {result && (
                  <ExportButton
                    result={result}
                    filteredDetections={filteredDetections}
                    imageSrc={imageSrc}
                    detectionHistory={detectionHistory}
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

                {/* Empty state — no result yet, image not uploading */}
                {!result && status.type === "idle" && (
                  <div className="rounded border border-[#2d3748] bg-[#0d1117] px-4 py-8 flex flex-col items-center gap-2 text-center">
                    <span className="text-[#2d3748] text-2xl">📡</span>
                    <p className="text-[#4a5568] text-xs font-mono">
                      No detection loaded
                    </p>
                    <p className="text-[#2d3748] text-xs font-mono">
                      Upload an image to run analysis
                    </p>
                  </div>
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
              <CopilotPanel
                detectionContext={result}
                detectionHistory={detectionHistory}
              />
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

// ---------------------------------------------------------------------------
// Feature #1 — Session history strip
// ---------------------------------------------------------------------------

interface SessionHistoryStripProps {
  history: DetectionHistoryEntry[];
  currentImageSrc: string | null;
  onSelect: (entry: DetectionHistoryEntry) => void;
}

function SessionHistoryStrip({ history, currentImageSrc, onSelect }: SessionHistoryStripProps) {
  return (
    <div className="rounded border border-[#2d3748] bg-[#0d1117] overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#161b27] border-b border-[#2d3748]">
        <span className="text-[#4a5568] text-[10px] font-mono uppercase tracking-widest">
          Session History
        </span>
        <span className="text-[#4a5568] text-[10px] font-mono">
          {history.length} detection{history.length !== 1 ? "s" : ""} this session
        </span>
      </div>
      <div className="flex items-center gap-2 px-3 py-2 overflow-x-auto ogb-scrollbar">
        {history.map((entry, idx) => {
          const isActive = entry.imageSrc === currentImageSrc;
          const topClass = entry.result.detections[0]?.class_name ?? "—";
          const topConf = entry.result.detections[0]
            ? `${(entry.result.detections[0].confidence * 100).toFixed(0)}%`
            : "";
          const time = new Date(entry.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          });
          return (
            <button
              key={entry.id}
              onClick={() => onSelect(entry)}
              title={`#${idx + 1} · ${entry.result.detection_count} objects · ${time}`}
              className={[
                "flex-shrink-0 flex flex-col items-center gap-1 px-2 py-1.5 rounded border text-[10px] font-mono transition-colors",
                isActive
                  ? "border-[#3b82f6] bg-[#1a2847] text-[#7aa2d4]"
                  : "border-[#2d3748] bg-[#0f1624] text-[#4a5568] hover:border-[#3b82f6] hover:text-[#a0aec0]",
              ].join(" ")}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={entry.imageSrc}
                alt={`Detection ${idx + 1}`}
                className="w-12 h-10 object-cover rounded"
              />
              <span className="text-[#a0aec0]">#{idx + 1}</span>
              <span className="truncate max-w-[52px]">{topClass}</span>
              {topConf && <span>{topConf}</span>}
              <span className="text-[#2d3748]">{time}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Feature #2 — Skeleton loader
// ---------------------------------------------------------------------------

function SkeletonLoader() {
  return (
    <div className="flex flex-col gap-3 animate-pulse" aria-label="Loading detection...">
      {/* Image area skeleton */}
      <div className="w-full rounded border border-[#2d3748] bg-[#0d1117] overflow-hidden">
        <div
          className="w-full bg-[#161b27] ogb-skeleton"
          style={{ paddingTop: "62.5%" }} // 16:10 aspect ratio placeholder
        />
      </div>
      {/* Status skeleton */}
      <div className="flex items-center gap-3">
        <div className="h-3 w-32 rounded bg-[#1e2535] ogb-skeleton" />
        <div className="h-3 w-16 rounded bg-[#1e2535] ogb-skeleton" />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Feature #3 — Confidence threshold slider
// ---------------------------------------------------------------------------

interface ConfidenceSliderProps {
  value: number;
  onChange: (v: number) => void;
  totalDetections: number;
  filteredCount: number;
  backendThreshold: number;
}

function ConfidenceSlider({
  value,
  onChange,
  totalDetections,
  filteredCount,
  backendThreshold,
}: ConfidenceSliderProps) {
  return (
    <div className="rounded border border-[#2d3748] bg-[#0d1117] px-4 py-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[#4a5568] text-xs font-mono uppercase tracking-widest">
          Confidence Filter
        </span>
        <span className="text-[#3b82f6] text-xs font-mono font-bold">
          ≥ {(value * 100).toFixed(0)}%
        </span>
      </div>
      <input
        type="range"
        min={backendThreshold}
        max={1.0}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-[#2d3748] accent-[#3b82f6]"
        aria-label="Confidence threshold"
      />
      <div className="flex items-center justify-between text-[10px] font-mono">
        <span className="text-[#4a5568]">
          Backend floor: {(backendThreshold * 100).toFixed(0)}%
        </span>
        <span className="text-[#a0aec0]">
          Showing {filteredCount} / {totalDetections} detection{totalDetections !== 1 ? "s" : ""}
        </span>
      </div>
      {value < backendThreshold && (
        <p className="text-[#f59e0b] text-[10px] font-mono">
          ⚠ Slider floor limited to backend minimum ({(backendThreshold * 100).toFixed(0)}%) — detections below that were not returned.
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: InfoRow
// ---------------------------------------------------------------------------

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
