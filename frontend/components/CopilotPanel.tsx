/**
 * OGB — OrbitalGuard
 * CopilotPanel — AI Copilot chat panel, grounded in detection context.
 *
 * Enforces spec rules in the UI:
 *  - Passes detection JSON as context on every message so answers are grounded.
 *  - Shows a banner when no detection has run yet (orbital fields N/A in MVP).
 *  - Displays backend errors (503, 500, network) visibly, not silently.
 *  - Suggested question chips guide operators during live demos.
 */
"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { Bot, Send, AlertTriangle, Loader } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { copilotChat } from "@/lib/api";
import type { CopilotMessage, DetectionResponse, ToolCallRecord } from "@/lib/types";

interface Props {
  detectionContext: DetectionResponse | null;
}

interface DisplayMessage {
  role: "user" | "assistant" | "error";
  content: string;
  toolCalls?: ToolCallRecord[];
}

const WELCOME: DisplayMessage = {
  role: "assistant",
  content:
    "OGB Copilot online. Upload and analyse an image to ground my responses in real detection data. I will not invent orbital data — if it is not in the detection output, I will say so.",
};

// Suggested questions shown as chips above the input.
// Pre-detection chips are shown before any image is analysed.
// Post-detection chips appear once a detection result is loaded.
const SUGGESTED_PRE: string[] = [
  "What can you detect?",
  "How confident is the detector?",
  "What happens after I upload an image?",
];

const SUGGESTED_POST: string[] = [
  "What did you detect?",
  "How confident are you?",
  "What should I do next?",
  "Tell me about this object class.",
];

export default function CopilotPanel({ detectionContext }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([WELCOME]);
  const [history, setHistory] = useState<CopilotMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [chipsUsed, setChipsUsed] = useState<Set<string>>(new Set());
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // When a new detection arrives, inject a brief system note
  const prevDetectionRef = useRef<DetectionResponse | null>(null);
  useEffect(() => {
    if (
      detectionContext &&
      detectionContext !== prevDetectionRef.current
    ) {
      prevDetectionRef.current = detectionContext;
      const count = detectionContext.detection_count;
      const classes = [
        ...new Set(detectionContext.detections.map((d) => d.class_name)),
      ].join(", ");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Detection complete — ${count} object${count !== 1 ? "s" : ""} found${classes ? `: ${classes}` : ""}. Context loaded. Ask me anything about the results.`,
        },
      ]);
    }
  }, [detectionContext]);

  // Reset used chips when a new detection loads
  useEffect(() => {
    if (detectionContext) setChipsUsed(new Set());
  }, [detectionContext]);

  function useChip(question: string) {
    setInput(question);
    setChipsUsed((prev) => new Set(prev).add(question));
    inputRef.current?.focus();
  }

  async function send() {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: DisplayMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    const result = await copilotChat({
      message: text,
      history,
      detection_context: detectionContext ?? undefined,
      // orbital_context and risk_context are V2 — omit entirely in MVP
      // (backend will see null and copilot will say "no orbital data available")
    });

    setLoading(false);

    if (result.ok) {
      const assistantMsg: DisplayMessage = {
        role: "assistant",
        content: result.data.reply,
        toolCalls: result.data.tool_calls?.length ? result.data.tool_calls : undefined,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      // Accumulate history for multi-turn context
      setHistory((prev) => [
        ...prev,
        { role: "user", content: text },
        { role: "assistant", content: result.data.reply },
      ]);
    } else {
      setMessages((prev) => [
        ...prev,
        { role: "error", content: result.message },
      ]);
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    // Ctrl/Cmd+Enter to send; plain Enter adds newline
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex flex-col h-full rounded border border-[#2d3748] bg-[#0d1117] overflow-hidden">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-4 py-2 bg-[#161b27] border-b border-[#2d3748] flex-shrink-0">
        <Bot size={15} className="text-[#3b82f6]" strokeWidth={1.5} />
        <span className="text-[#a0aec0] text-xs font-mono uppercase tracking-widest">
          AI Copilot
        </span>
        <span className="ml-auto text-[#4a5568] text-xs font-mono">
          Gemini · vision-grounded
        </span>
        {/* Orbital data availability badge */}
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-[#0d1117] border border-[#2d3748] text-[#4a5568]">
          <span className="w-1.5 h-1.5 rounded-full bg-[#4a5568]" />
          Orbital: N/A (V2)
        </span>
      </div>

      {/* ── No-detection notice ─────────────────────────────────────────── */}
      {!detectionContext && (
        <div className="px-4 py-2 border-b border-[#1e2535] bg-[#0f1420]">
          <p className="text-[#4a5568] text-xs font-mono">
            ⓘ No detection loaded — run Camera Analysis first to ground
            copilot responses in real data.
          </p>
        </div>
      )}

      {/* ── Message history ─────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-0 ogb-scrollbar">
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}

        {/* Typing indicator */}
        {loading && (
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full bg-[#161b27] border border-[#2d3748] flex items-center justify-center flex-shrink-0">
              <Loader size={10} className="text-[#3b82f6] animate-spin" />
            </div>
            <div className="flex gap-1 items-center px-3 py-2 rounded bg-[#161b27] border border-[#2d3748]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#4a5568] animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-[#4a5568] animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-[#4a5568] animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Input ───────────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 border-t border-[#2d3748] bg-[#0d1117] p-3">
        {/* Suggested question chips */}
        <div className="flex flex-wrap gap-1.5 mb-2">
          {(detectionContext ? SUGGESTED_POST : SUGGESTED_PRE).map((q) => (
            <button
              key={q}
              onClick={() => useChip(q)}
              disabled={loading || chipsUsed.has(q)}
              className={[
                "px-2 py-1 rounded text-[10px] font-mono border transition-colors",
                loading || chipsUsed.has(q)
                  ? "border-[#1e2535] text-[#2d3748] cursor-not-allowed"
                  : "border-[#2d4a7a] text-[#7aa2d4] hover:bg-[#1a2847] hover:text-[#c5d8f7] cursor-pointer",
              ].join(" ")}
            >
              {q}
            </button>
          ))}
        </div>
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
            rows={2}
            placeholder={
              detectionContext
                ? "Ask about the detection… (Ctrl+Enter to send)"
                : "Run a detection first to get grounded answers…"
            }
            className={[
              "flex-1 resize-none bg-[#161b27] border border-[#2d3748] rounded",
              "px-3 py-2 text-xs font-mono text-[#e2e8f0] placeholder-[#4a5568]",
              "focus:outline-none focus:border-[#3b82f6] focus:ring-1 focus:ring-[#3b82f6]/30",
              "transition-colors leading-relaxed",
              loading ? "opacity-50" : "",
            ].join(" ")}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className={[
              "flex-shrink-0 w-9 h-9 rounded border flex items-center justify-center",
              "transition-colors",
              loading || !input.trim()
                ? "border-[#2d3748] text-[#4a5568] cursor-not-allowed"
                : "border-[#3b82f6] text-[#3b82f6] hover:bg-[#3b82f6]/10 cursor-pointer",
            ].join(" ")}
            aria-label="Send message"
          >
            <Send size={14} strokeWidth={1.5} />
          </button>
        </div>
        <p className="text-[#2d3748] text-[10px] font-mono mt-1.5">
          Decision-support only · Orbital data requires V2 pipeline ·
          Operator makes all decisions
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: individual message bubble
// ---------------------------------------------------------------------------

function MessageBubble({ msg }: { msg: DisplayMessage }) {
  const isUser = msg.role === "user";
  const isError = msg.role === "error";

  if (isError) {
    return (
      <div className="flex items-start gap-2">
        <div className="w-6 h-6 rounded-full bg-[#1a0909] border border-[#7f1d1d] flex items-center justify-center flex-shrink-0 mt-0.5">
          <AlertTriangle size={10} className="text-red-400" />
        </div>
        <div className="flex-1 px-3 py-2 rounded bg-[#1a0909] border border-[#7f1d1d]">
          <p className="text-red-400 text-xs font-mono leading-relaxed">
            {msg.content}
          </p>
        </div>
      </div>
    );
  }

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] px-3 py-2 rounded bg-[#1a2847] border border-[#2d4a7a]">
          <p className="text-[#c5d8f7] text-xs font-mono leading-relaxed whitespace-pre-wrap">
            {msg.content}
          </p>
        </div>
      </div>
    );
  }

  // Assistant
  return (
    <div className="flex items-start gap-2">
      <div className="w-6 h-6 rounded-full bg-[#161b27] border border-[#2d3748] flex items-center justify-center flex-shrink-0 mt-0.5">
        <Bot size={10} className="text-[#3b82f6]" />
      </div>
      <div className="flex-1 min-w-0">
        {/* Tool-call badges — shown when this response used function-calling */}
        {msg.toolCalls && msg.toolCalls.length > 0 && (
          <div className="mb-1.5 space-y-1">
            {msg.toolCalls.map((tc, i) => (
              <ToolCallBadge key={i} tc={tc} />
            ))}
          </div>
        )}
        <div className="px-3 py-2 rounded bg-[#161b27] border border-[#2d3748]">
          <div className="ogb-md text-[#e2e8f0] text-xs font-mono leading-relaxed">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tool-call badge — shows what the risk engine returned
// ---------------------------------------------------------------------------

function ToolCallBadge({ tc }: { tc: ToolCallRecord }) {
  const result = tc.result as Record<string, unknown>;
  const isError = result.status === "ERROR" || result.status === "NOT_IMPLEMENTED";

  return (
    <div
      className={[
        "flex items-start gap-2 px-3 py-2 rounded border text-[10px] font-mono",
        isError
          ? "bg-[#1a0a00] border-[#7f3d00] text-[#f59e0b]"
          : "bg-[#0a1a0d] border-[#1a4a2a] text-[#4ade80]",
      ].join(" ")}
    >
      <span className="flex-shrink-0 mt-0.5">{isError ? "⚠️" : "🔧"}</span>
      <div className="min-w-0">
        <span className="text-[#6ee7b7] font-bold">{tc.tool_name}</span>
        {!isError && typeof result.risk_score === "number" && (
          <span className="ml-2">
            score={" "}
            <span className="text-white font-bold">
              {(result.risk_score as number).toFixed(4)}
            </span>
            {" "}· category={" "}
            <span
              className={
                result.risk_category === "CRITICAL"
                  ? "text-red-400 font-bold"
                  : result.risk_category === "HIGH"
                  ? "text-orange-400 font-bold"
                  : result.risk_category === "MEDIUM"
                  ? "text-yellow-400 font-bold"
                  : "text-green-400 font-bold"
              }
            >
              {result.risk_category as string}
            </span>
          </span>
        )}
        {isError && (
          <span className="ml-2 text-[#f59e0b]">
            {(result.error as string) ?? "Tool unavailable"}
          </span>
        )}
      </div>
    </div>
  );
}
