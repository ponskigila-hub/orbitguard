/**
 * OGB — OrbitalGuard
 * ThreatCenter — list of analyzed objects, color-coded by risk, sortable.
 */
"use client";

import { useState } from "react";
import { ArrowUp, ArrowDown, Minus } from "lucide-react";
import type { ThreatEntry } from "@/lib/types";
import { RISK_COLOR, RISK_EMOJI } from "@/lib/types";

interface Props {
  threats: ThreatEntry[];
  onSelect: (t: ThreatEntry) => void;
  selected: ThreatEntry | null;
}

type SortKey = "risk" | "dmin" | "age";
type SortDir = "asc" | "desc";

const RISK_ORDER: Record<string, number> = {
  CRITICAL: 4,
  HIGH: 3,
  MEDIUM: 2,
  LOW: 1,
};

export default function ThreatCenter({ threats, onSelect, selected }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("risk");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  const sorted = [...threats].sort((a, b) => {
    let cmp = 0;
    if (sortKey === "risk") {
      const ra = RISK_ORDER[a.risk_category ?? ""] ?? 0;
      const rb = RISK_ORDER[b.risk_category ?? ""] ?? 0;
      cmp = ra - rb;
    } else if (sortKey === "dmin") {
      cmp = (a.d_min_km ?? Infinity) - (b.d_min_km ?? Infinity);
    } else if (sortKey === "age") {
      cmp = (a.tle_age_days ?? 0) - (b.tle_age_days ?? 0);
    }
    return sortDir === "desc" ? -cmp : cmp;
  });

  return (
    <div className="rounded border border-[#2d3748] bg-[#0d1117] overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#161b27] border-b border-[#2d3748]">
        <span className="text-[#a0aec0] text-xs font-mono uppercase tracking-widest">
          Threat Center
        </span>
        <span className="text-[#4a5568] text-xs font-mono">
          {threats.length} object{threats.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Column headers */}
      <div className="grid grid-cols-[auto_1fr_auto_auto] gap-2 px-4 py-2 border-b border-[#1e2535] text-[#4a5568] text-[10px] font-mono uppercase tracking-widest">
        <span>Risk</span>
        <span>Object</span>
        <SortHeader label="d_min" active={sortKey === "dmin"} dir={sortDir} onClick={() => toggleSort("dmin")} />
        <SortHeader label="TLE age" active={sortKey === "age"} dir={sortDir} onClick={() => toggleSort("age")} />
      </div>

      {/* List */}
      {sorted.length === 0 ? (
        <div className="px-4 py-8 text-center text-[#4a5568] text-xs font-mono">
          No objects analysed yet. Use the TLE Analysis form above.
        </div>
      ) : (
        <div className="divide-y divide-[#1e2535]">
          {sorted.map((t) => (
            <ThreatRow
              key={t.id}
              threat={t}
              active={selected?.id === t.id}
              onClick={() => onSelect(t)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SortHeader({
  label,
  active,
  dir,
  onClick,
}: {
  label: string;
  active: boolean;
  dir: SortDir;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "flex items-center gap-0.5 text-[10px] font-mono uppercase tracking-widest",
        active ? "text-[#7aa2d4]" : "text-[#4a5568] hover:text-[#a0aec0]",
        "transition-colors",
      ].join(" ")}
    >
      {label}
      {active ? (
        dir === "desc" ? (
          <ArrowDown size={10} />
        ) : (
          <ArrowUp size={10} />
        )
      ) : (
        <Minus size={10} className="opacity-30" />
      )}
    </button>
  );
}

function ThreatRow({
  threat,
  active,
  onClick,
}: {
  threat: ThreatEntry;
  active: boolean;
  onClick: () => void;
}) {
  const cat = threat.risk_category ?? "UNKNOWN";
  const emoji = RISK_EMOJI[cat] ?? "●";
  const color = RISK_COLOR[cat] ?? "text-[#a0aec0]";

  return (
    <button
      onClick={onClick}
      className={[
        "w-full grid grid-cols-[auto_1fr_auto_auto] gap-2 px-4 py-2.5",
        "text-left transition-colors",
        active
          ? "bg-[#1a2847]"
          : "hover:bg-[#0f1420]",
      ].join(" ")}
    >
      {/* Risk indicator */}
      <span className="text-sm leading-none self-center">{emoji}</span>

      {/* Object ID + category */}
      <div className="min-w-0">
        <p className="text-xs font-mono text-[#e2e8f0] truncate">{threat.id}</p>
        <p className={`text-[10px] font-mono ${color}`}>{cat}</p>
      </div>

      {/* d_min */}
      <span className="text-xs font-mono text-[#718096] self-center">
        {threat.d_min_km != null ? `${threat.d_min_km.toFixed(1)} km` : "—"}
      </span>

      {/* TLE age */}
      <span className="text-xs font-mono text-[#718096] self-center">
        {threat.tle_age_days != null ? `${Math.abs(threat.tle_age_days).toFixed(1)} d` : "—"}
      </span>
    </button>
  );
}
