/**
 * OGB — OrbitalGuard
 * StatusBar — top status / error banner.
 */
"use client";

import { AlertTriangle, CheckCircle, Loader } from "lucide-react";

export type Status =
  | { type: "idle" }
  | { type: "loading" }
  | { type: "success"; message: string }
  | { type: "error"; message: string };

interface Props {
  status: Status;
}

export default function StatusBar({ status }: Props) {
  if (status.type === "idle") return null;

  const configs = {
    loading: {
      icon: <Loader size={15} className="animate-spin text-[#3b82f6]" />,
      text: "text-[#a0aec0]",
      bg: "bg-[#0f1624] border-[#2d3748]",
      label: "Analysing image…",
    },
    success: {
      icon: <CheckCircle size={15} className="text-green-400" />,
      text: "text-green-400",
      bg: "bg-[#0a1a0f] border-[#166534]",
      label: status.type === "success" ? status.message : "",
    },
    error: {
      icon: <AlertTriangle size={15} className="text-red-400" />,
      text: "text-red-400",
      bg: "bg-[#1a0909] border-[#7f1d1d]",
      label: status.type === "error" ? status.message : "",
    },
  } as const;

  const cfg = configs[status.type as keyof typeof configs];

  return (
    <div
      className={`flex items-center gap-2 px-4 py-2 rounded border text-sm font-mono ${cfg.bg} ${cfg.text}`}
      role={status.type === "error" ? "alert" : "status"}
    >
      {cfg.icon}
      <span>{cfg.label}</span>
    </div>
  );
}
