/**
 * OGB — OrbitalGuard
 * OrbitalAnalysisForm — paste TLE lines and run analysis.
 * Matches the dark mission-control style of the rest of the app.
 */
"use client";

import { useState } from "react";
import { Satellite, ChevronDown, ChevronUp } from "lucide-react";
import { analyzeOrbital } from "@/lib/api";
import type { OrbitalAnalyzeResponse } from "@/lib/types";

interface Props {
  onResult: (result: OrbitalAnalyzeResponse) => void;
  onError: (msg: string) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
}

const _PLACEHOLDER_L1 =
  "1 88888U          80275.98708465  .00073094  13844-3  66816-4 0    87";
const _PLACEHOLDER_L2 =
  "2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518  1058";

export default function OrbitalAnalysisForm({
  onResult,
  onError,
  loading,
  setLoading,
}: Props) {
  const [line1, setLine1] = useState("");
  const [line2, setLine2] = useState("");
  const [ts, setTs] = useState("");
  const [showConj, setShowConj] = useState(false);
  const [tgtLine1, setTgtLine1] = useState("");
  const [tgtLine2, setTgtLine2] = useState("");
  const [windowHrs, setWindowHrs] = useState("24");

  async function submit() {
    const l1 = line1.trim();
    const l2 = line2.trim();
    if (l1.length !== 69 || l2.length !== 69) {
      onError(
        `TLE lines must be exactly 69 characters each. Got L1=${l1.length}, L2=${l2.length}.`
      );
      return;
    }
    setLoading(true);
    const res = await analyzeOrbital({
      tle_line1: l1,
      tle_line2: l2,
      timestamp_utc: ts.trim() || undefined,
      target_tle_line1: showConj ? tgtLine1.trim() || undefined : undefined,
      target_tle_line2: showConj ? tgtLine2.trim() || undefined : undefined,
      conjunction_window_hours: parseFloat(windowHrs) || 24,
    });
    setLoading(false);
    if (res.ok) {
      onResult(res.data);
    } else {
      onError(res.message);
    }
  }

  return (
    <div className="rounded border border-[#2d3748] bg-[#0d1117] overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2 bg-[#161b27] border-b border-[#2d3748]">
        <Satellite size={14} className="text-[#3b82f6]" strokeWidth={1.5} />
        <span className="text-[#a0aec0] text-xs font-mono uppercase tracking-widest">
          TLE Analysis
        </span>
      </div>

      <div className="px-4 py-4 space-y-3">
        {/* Primary TLE */}
        <div>
          <label className="block text-[#4a5568] text-xs font-mono mb-1 uppercase tracking-widest">
            Primary Object — TLE Line 1
          </label>
          <input
            className={_inputClass}
            value={line1}
            onChange={(e) => setLine1(e.target.value)}
            placeholder={_PLACEHOLDER_L1}
            disabled={loading}
            spellCheck={false}
          />
        </div>
        <div>
          <label className="block text-[#4a5568] text-xs font-mono mb-1 uppercase tracking-widest">
            Primary Object — TLE Line 2
          </label>
          <input
            className={_inputClass}
            value={line2}
            onChange={(e) => setLine2(e.target.value)}
            placeholder={_PLACEHOLDER_L2}
            disabled={loading}
            spellCheck={false}
          />
        </div>
        <div>
          <label className="block text-[#4a5568] text-xs font-mono mb-1 uppercase tracking-widest">
            Timestamp UTC (ISO-8601, optional)
          </label>
          <input
            className={_inputClass}
            value={ts}
            onChange={(e) => setTs(e.target.value)}
            placeholder="2025-08-01T12:00:00Z  (blank = now)"
            disabled={loading}
            spellCheck={false}
          />
        </div>

        {/* Conjunction toggle */}
        <button
          onClick={() => setShowConj(!showConj)}
          className="flex items-center gap-1.5 text-xs font-mono text-[#7aa2d4] hover:text-[#c5d8f7] transition-colors"
          disabled={loading}
        >
          {showConj ? (
            <ChevronUp size={13} />
          ) : (
            <ChevronDown size={13} />
          )}
          {showConj ? "Hide" : "Add"} conjunction target (optional)
        </button>

        {showConj && (
          <div className="space-y-3 pl-3 border-l border-[#2d3748]">
            <div>
              <label className="block text-[#4a5568] text-xs font-mono mb-1 uppercase tracking-widest">
                Target Object — TLE Line 1
              </label>
              <input
                className={_inputClass}
                value={tgtLine1}
                onChange={(e) => setTgtLine1(e.target.value)}
                placeholder="1 XXXXX…"
                disabled={loading}
                spellCheck={false}
              />
            </div>
            <div>
              <label className="block text-[#4a5568] text-xs font-mono mb-1 uppercase tracking-widest">
                Target Object — TLE Line 2
              </label>
              <input
                className={_inputClass}
                value={tgtLine2}
                onChange={(e) => setTgtLine2(e.target.value)}
                placeholder="2 XXXXX…"
                disabled={loading}
                spellCheck={false}
              />
            </div>
            <div>
              <label className="block text-[#4a5568] text-xs font-mono mb-1 uppercase tracking-widest">
                Analysis Window (hours)
              </label>
              <input
                className={`${_inputClass} w-32`}
                value={windowHrs}
                onChange={(e) => setWindowHrs(e.target.value)}
                type="number"
                min={0.5}
                max={168}
                disabled={loading}
              />
            </div>
          </div>
        )}

        <button
          onClick={submit}
          disabled={loading || !line1.trim() || !line2.trim()}
          className={[
            "w-full py-2 rounded text-xs font-mono border transition-colors",
            loading || !line1.trim() || !line2.trim()
              ? "border-[#2d3748] text-[#4a5568] cursor-not-allowed bg-transparent"
              : "border-[#3b82f6] text-[#3b82f6] hover:bg-[#3b82f6]/10 cursor-pointer",
          ].join(" ")}
        >
          {loading ? "Propagating…" : "Analyse →"}
        </button>
      </div>
    </div>
  );
}

const _inputClass = [
  "w-full bg-[#161b27] border border-[#2d3748] rounded",
  "px-3 py-1.5 text-xs font-mono text-[#e2e8f0] placeholder-[#2d3748]",
  "focus:outline-none focus:border-[#3b82f6] focus:ring-1 focus:ring-[#3b82f6]/30",
  "transition-colors",
].join(" ");
