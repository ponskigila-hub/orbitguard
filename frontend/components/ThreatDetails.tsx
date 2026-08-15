/**
 * OGB — OrbitalGuard
 * ThreatDetails — full result panel for one orbital analysis.
 * Shows: state vector, TLE age, conjunction result, risk score.
 */
"use client";

import type { OrbitalAnalyzeResponse } from "@/lib/types";
import { RISK_COLOR, RISK_EMOJI } from "@/lib/types";

interface Props {
  result: OrbitalAnalyzeResponse;
}

export default function ThreatDetails({ result }: Props) {
  const state = result.propagated_state;
  const conj = result.conjunction;

  return (
    <div className="rounded border border-[#2d3748] bg-[#0d1117] overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 bg-[#161b27] border-b border-[#2d3748] flex items-center justify-between">
        <span className="text-[#a0aec0] text-xs font-mono uppercase tracking-widest">
          Analysis Result
        </span>
        <span className="text-[#4a5568] text-xs font-mono">
          {result.analysis_type === "propagation_and_conjunction"
            ? "Propagation + Conjunction"
            : "Propagation only"}
        </span>
      </div>

      <div className="px-4 py-4 space-y-4">
        {/* Propagated state */}
        <Section title="Propagated State (ECI)">
          {state.ok ? (
            <>
              <Row label="Position (km)" value={
                state.position_km
                  ? `[${state.position_km.map((x) => x.toFixed(3)).join(", ")}]`
                  : "—"
              } />
              <Row label="Velocity (km/s)" value={
                state.velocity_km_s
                  ? `[${state.velocity_km_s.map((x) => x.toFixed(6)).join(", ")}]`
                  : "—"
              } />
              <Row label="Propagated at" value={state.propagated_at_utc ?? "—"} />
              <Row label="TLE epoch" value={state.epoch_utc ?? "—"} />
              <Row
                label="TLE age"
                value={
                  state.tle_age_days != null
                    ? `${state.tle_age_days.toFixed(2)} days`
                    : "—"
                }
                highlight={
                  state.tle_age_days != null && Math.abs(state.tle_age_days) > 7
                    ? "warn"
                    : undefined
                }
              />
            </>
          ) : (
            <p className="text-red-400 text-xs font-mono">
              Propagation error: {state.error}
            </p>
          )}
        </Section>

        {/* Conjunction result */}
        {conj && (
          <Section title="Conjunction Analysis">
            {conj.ok ? (
              <>
                <Row label="TCA" value={conj.tca_utc ?? "—"} />
                <Row
                  label="d_min"
                  value={conj.d_min_km != null ? `${conj.d_min_km.toFixed(3)} km` : "—"}
                  highlight={
                    conj.d_min_km != null && conj.d_min_km < 1.0 ? "crit" : undefined
                  }
                />
                <Row
                  label="v_rel"
                  value={conj.v_rel_km_s != null ? `${conj.v_rel_km_s.toFixed(3)} km/s` : "—"}
                />
                <Row
                  label="TLE age (sat 1)"
                  value={conj.tle_age_days_sat1 != null ? `${conj.tle_age_days_sat1.toFixed(2)} d` : "—"}
                />
                <Row
                  label="TLE age (sat 2)"
                  value={conj.tle_age_days_sat2 != null ? `${conj.tle_age_days_sat2.toFixed(2)} d` : "—"}
                />
                <Row
                  label="Coarse samples"
                  value={conj.coarse_samples?.toString() ?? "—"}
                />

                {/* Risk block */}
                {conj.risk && (
                  <RiskBlock risk={conj.risk} />
                )}
              </>
            ) : (
              <p className="text-red-400 text-xs font-mono">
                Conjunction error: {conj.error}
              </p>
            )}
          </Section>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="text-[#4a5568] text-[10px] font-mono uppercase tracking-widest mb-2">
        {title}
      </p>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function Row({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: "warn" | "crit";
}) {
  const valueClass =
    highlight === "crit"
      ? "text-red-400 font-bold"
      : highlight === "warn"
      ? "text-yellow-400"
      : "text-[#a0aec0]";
  return (
    <div className="flex items-start justify-between gap-4">
      <span className="text-[#4a5568] text-xs font-mono flex-shrink-0">{label}</span>
      <span className={`text-xs font-mono text-right ${valueClass}`}>{value}</span>
    </div>
  );
}

function RiskBlock({ risk }: { risk: Record<string, unknown> }) {
  const score = risk.risk_score as number | undefined;
  const cat = risk.risk_category as string | undefined;
  const emoji = cat ? RISK_EMOJI[cat] ?? "●" : "●";
  const color = cat ? RISK_COLOR[cat] ?? "text-[#a0aec0]" : "text-[#a0aec0]";

  return (
    <div className="mt-2 pt-2 border-t border-[#2d3748]">
      <p className="text-[#4a5568] text-[10px] font-mono uppercase tracking-widest mb-2">
        Risk Assessment
      </p>
      <div className="flex items-center gap-3 px-3 py-2 rounded bg-[#161b27] border border-[#2d3748]">
        <span className="text-lg leading-none">{emoji}</span>
        <div>
          <p className={`text-sm font-mono font-bold ${color}`}>
            {cat ?? "UNKNOWN"}
          </p>
          {score != null && (
            <p className="text-[#718096] text-xs font-mono">
              score = {score.toFixed(6)}
            </p>
          )}
        </div>
      </div>
      <p className="text-[#4a5568] text-[10px] font-mono mt-1.5">
        Priority indicator only. Operator makes all decisions.
      </p>
    </div>
  );
}
