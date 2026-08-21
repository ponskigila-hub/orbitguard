/**
 * OGB — OrbitalGuard
 * DetectionSummary — tabular list of detected objects below the image.
 */
"use client";

import type { Detection } from "@/lib/types";

const CLASS_COLORS = [
  "#3b82f6","#ef4444","#f97316","#a855f7","#06b6d4",
  "#84cc16","#eab308","#f43f5e","#10b981","#6366f1","#ec4899",
];

function confidenceBadge(conf: number) {
  if (conf >= 0.75) return "text-green-400";
  if (conf >= 0.50) return "text-yellow-400";
  return "text-orange-400";
}

interface Props {
  detections: Detection[];
  latencyMs: number | null;
  modelVersion: string;
}

export default function DetectionSummary({ detections, latencyMs, modelVersion }: Props) {
  // Aggregate by class
  const counts: Record<string, { count: number; color: string; maxConf: number; classId: number }> = {};
  for (const d of detections) {
    if (!counts[d.class_name]) {
      counts[d.class_name] = {
        count: 0,
        color: CLASS_COLORS[d.class_id % CLASS_COLORS.length],
        maxConf: 0,
        classId: d.class_id,
      };
    }
    counts[d.class_name].count += 1;
    counts[d.class_name].maxConf = Math.max(counts[d.class_name].maxConf, d.confidence);
  }

  const rows = Object.entries(counts).sort((a, b) => b[1].maxConf - a[1].maxConf);

  return (
    <div className="rounded border border-[#2d3748] bg-[#0d1117] overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#161b27] border-b border-[#2d3748]">
        <span className="text-[#a0aec0] text-xs font-mono uppercase tracking-widest">
          Detection Results
        </span>
        <span className="text-[#4a5568] text-xs font-mono">
          {modelVersion}
        </span>
      </div>

      {detections.length === 0 ? (
        <div className="px-4 py-6 text-center text-[#4a5568] text-sm font-mono">
          No objects detected above confidence threshold.
        </div>
      ) : (
        <>
          {/* Aggregate summary */}
          <div className="px-4 py-3 border-b border-[#1e2535]">
            <p className="text-[#e2e8f0] text-sm font-mono">
              <span className="text-[#3b82f6] font-bold">{detections.length}</span>{" "}
              object{detections.length !== 1 ? "s" : ""} detected
              {latencyMs != null && (
                <span className="text-[#4a5568] ml-2">· {latencyMs.toFixed(0)} ms</span>
              )}
            </p>
          </div>

          {/* Per-class rows */}
          <table className="w-full text-sm font-mono">
            <thead>
              <tr className="text-[#4a5568] text-xs border-b border-[#1e2535]">
                <th className="text-left px-4 py-2 font-normal">Class</th>
                <th className="text-right px-4 py-2 font-normal">Count</th>
                <th className="text-right px-4 py-2 font-normal">Max conf.</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([name, info]) => (
                <tr
                  key={name}
                  className="border-b border-[#1a2236] last:border-0 hover:bg-[#111827] transition-colors"
                >
                  <td className="px-4 py-2 flex items-center gap-2">
                    <span
                      className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: info.color }}
                    />
                    <span className="text-[#e2e8f0]">{name}</span>
                  </td>
                  <td className="px-4 py-2 text-right text-[#a0aec0]">{info.count}</td>
                  <td className={`px-4 py-2 text-right ${confidenceBadge(info.maxConf)}`}>
                    {(info.maxConf * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Vision-only disclaimer */}
          <div className="px-4 py-2 border-t border-[#1e2535] bg-[#0d1117]">
            <p className="text-[#4a5568] text-xs font-mono">
              ⓘ Vision output only — no distance, velocity, or collision data.
              
            </p>
          </div>
        </>
      )}
    </div>
  );
}
