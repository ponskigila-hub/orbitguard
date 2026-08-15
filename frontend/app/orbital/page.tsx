/**
 * OGB — OrbitalGuard
 * /orbital — Orbital Intelligence page (V2).
 *
 * Layout:
 *  ┌──────────────────────────────────────┬───────────────────────────┐
 *  │  TLE Analysis Form (left)            │  Threat Details (right)   │
 *  └──────────────────────────────────────┴───────────────────────────┘
 *  ┌──────────────────────────────────────────────────────────────────┐
 *  │  Threat Center (bottom — all analyzed objects, sortable)          │
 *  └──────────────────────────────────────────────────────────────────┘
 */
"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import { Shield, AlertTriangle, ChevronLeft } from "lucide-react";
import OrbitalAnalysisForm from "@/components/OrbitalAnalysisForm";
import ThreatCenter from "@/components/ThreatCenter";
import ThreatDetails from "@/components/ThreatDetails";
import type { OrbitalAnalyzeResponse, ThreatEntry } from "@/lib/types";

let _threatCounter = 0;

function makeThreatEntry(result: OrbitalAnalyzeResponse): ThreatEntry {
  _threatCounter += 1;
  const risk = result.conjunction?.risk;
  return {
    id: `Object ${_threatCounter}`,
    result,
    risk_category: (risk?.risk_category as string) ?? undefined,
    risk_score: (risk?.risk_score as number) ?? undefined,
    d_min_km: result.conjunction?.d_min_km ?? undefined,
    tle_age_days: result.propagated_state.tle_age_days ?? undefined,
  };
}

export default function OrbitalPage() {
  const [threats, setThreats] = useState<ThreatEntry[]>([]);
  const [selected, setSelected] = useState<ThreatEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleResult = useCallback((result: OrbitalAnalyzeResponse) => {
    const entry = makeThreatEntry(result);
    setThreats((prev) => [entry, ...prev]);
    setSelected(entry);
    setError(null);
  }, []);

  const handleError = useCallback((msg: string) => {
    setError(msg);
  }, []);

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
              Orbital Intelligence
            </span>
            <span className="text-[#2d3748] font-mono text-sm hidden sm:block">
              /
            </span>
            <span className="text-[#a0aec0] font-mono text-xs tracking-wider uppercase hidden sm:block">
              V2 Pipeline
            </span>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex items-center gap-1.5 text-xs font-mono text-[#4a5568] hover:text-[#a0aec0] transition-colors"
            >
              <ChevronLeft size={13} />
              Camera Analysis
            </Link>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-[#0d1117] border border-[#2d3748] text-[#4a5568]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#3b82f6] animate-pulse" />
              SGP4
            </span>
          </div>
        </div>
      </header>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-screen-2xl w-full mx-auto px-6 py-6 space-y-6">
        {/* Page header */}
        <div>
          <h1 className="font-mono text-sm font-bold text-[#e2e8f0] tracking-widest uppercase">
            Orbital Intelligence
          </h1>
          <p className="text-[#4a5568] text-xs font-mono mt-0.5">
            SGP4 TLE propagation (python-sgp4, Brandon Rhodes) · scipy minimize_scalar
            close-approach · risk engine (V2). Vision and orbital pipelines are
            separate — do not conflate camera detections with orbital tracks.
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <div className="flex items-start gap-2 px-4 py-3 rounded border border-[#7f1d1d] bg-[#1a0909]">
            <AlertTriangle size={14} className="text-red-400 flex-shrink-0 mt-0.5" />
            <p className="text-red-400 text-xs font-mono">{error}</p>
          </div>
        )}

        {/* Two-column: Form | Details */}
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_1fr] gap-4">
          <div>
            <h2 className="font-mono text-xs font-bold text-[#a0aec0] tracking-widest uppercase mb-2">
              TLE Analysis
            </h2>
            <OrbitalAnalysisForm
              onResult={handleResult}
              onError={handleError}
              loading={loading}
              setLoading={setLoading}
            />
          </div>
          <div>
            <h2 className="font-mono text-xs font-bold text-[#a0aec0] tracking-widest uppercase mb-2">
              {selected ? `Details — ${selected.id}` : "Select an object from the Threat Center"}
            </h2>
            {selected ? (
              <ThreatDetails result={selected.result} />
            ) : (
              <div className="rounded border border-[#2d3748] bg-[#0d1117] px-4 py-10 text-center">
                <p className="text-[#4a5568] text-xs font-mono">
                  Analyse a TLE to see details here.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Threat Center — full width */}
        <div>
          <h2 className="font-mono text-xs font-bold text-[#a0aec0] tracking-widest uppercase mb-2">
            Threat Center
          </h2>
          <ThreatCenter
            threats={threats}
            onSelect={setSelected}
            selected={selected}
          />
        </div>
      </main>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="border-t border-[#1e2535] flex-shrink-0">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between">
          <p className="text-[#2d3748] text-xs font-mono">
            OGB v2.0 — Orbital Intelligence · SGP4 · Conjunction Analysis
          </p>
          <p className="text-[#2d3748] text-xs font-mono">
            Decision-support only. Operator makes all decisions.
          </p>
        </div>
      </footer>
    </div>
  );
}
