// One agent step in the live timeline: who ran, which tools it called, what it produced.
import type { TraceEntry } from "../api";
import { formatMs } from "./ui";

const agentStyles: Record<string, string> = {
  planner: "border-l-sky-500",
  executor: "border-l-indigo-500",
  reviewer: "border-l-amber-500",
  finalize: "border-l-emerald-500",
};

const agentBadge: Record<string, string> = {
  planner: "bg-sky-100 text-sky-800",
  executor: "bg-indigo-100 text-indigo-800",
  reviewer: "bg-amber-100 text-amber-800",
  finalize: "bg-emerald-100 text-emerald-800",
};

/** The executor's prompt contains "Your current step (2 of 4): ..."; show just that line. */
function stepLabel(entry: TraceEntry): string | null {
  if (entry.agent !== "executor") return null;
  return entry.input.split("\n").find((line) => line.startsWith("Your current step")) ?? null;
}

export default function TraceCard({ entry }: { entry: TraceEntry }) {
  const step = stepLabel(entry);
  const tokens =
    entry.input_tokens || entry.output_tokens ? `${entry.input_tokens ?? 0} in / ${entry.output_tokens ?? 0} out` : null;

  return (
    <div className={`rounded-lg border border-slate-200 border-l-4 bg-white p-4 ${agentStyles[entry.agent] ?? ""}`}>
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span className={`rounded px-2 py-0.5 font-semibold uppercase ${agentBadge[entry.agent] ?? "bg-slate-100"}`}>
          {entry.agent}
        </span>
        {entry.model !== "-" && <span>{entry.provider}/{entry.model}</span>}
        <span>{formatMs(entry.latency_ms)}</span>
        {tokens && <span>{tokens} tokens</span>}
      </div>

      {step && <p className="mt-2 text-sm font-medium text-slate-800">{step.replace("Your current step", "Step")}</p>}

      {entry.tool_calls.length > 0 && (
        <ul className="mt-3 space-y-2">
          {entry.tool_calls.map((call, i) => (
            <li key={i} className="rounded-md bg-slate-50 p-2 text-xs">
              <details>
                <summary className="cursor-pointer">
                  <span className={`font-mono font-semibold ${call.error ? "text-red-700" : "text-slate-800"}`}>
                    {call.name}
                  </span>
                  <span className="font-mono text-slate-500">({JSON.stringify(call.args)})</span>
                  <span className="ml-2 text-slate-400">{formatMs(call.latency_ms)}</span>
                  {call.error && <span className="ml-2 font-semibold text-red-700">error</span>}
                </summary>
                <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap text-slate-700">{call.output}</pre>
              </details>
            </li>
          ))}
        </ul>
      )}

      {entry.agent !== "finalize" && (
        <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap font-sans text-sm text-slate-800">
          {entry.output}
        </pre>
      )}

      {entry.error && (
        <p className="mt-2 rounded bg-red-50 px-2 py-1 text-xs text-red-800"><span className="font-semibold">Error:</span> {entry.error}</p>
      )}

      {entry.agent !== "finalize" && (
        <details className="mt-2 text-xs text-slate-500">
          <summary className="cursor-pointer">Prompt input</summary>
          <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap">{entry.input}</pre>
        </details>
      )}
    </div>
  );
}
