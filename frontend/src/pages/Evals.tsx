// Latest benchmark results: summary table, simple bar charts, failure analysis, per-task results.
import { useEffect, useState } from "react";
import { api, type EvalReport, type EvalSummary } from "../api";
import { Card, ErrorBox } from "../components/ui";

function Bars({ title, rows, value, format, max }: {
  title: string;
  rows: EvalSummary[];
  value: (s: EvalSummary) => number;
  format: (n: number) => string;
  max?: number;
}) {
  const top = max ?? Math.max(...rows.map(value), 1e-9);
  return (
    <Card>
      <h3 className="mb-3 text-sm font-semibold text-slate-700">{title}</h3>
      <div className="space-y-2">
        {rows.map((s) => (
          <div key={`${s.provider}/${s.model}`} className="grid grid-cols-[9rem_1fr_4.5rem] items-center gap-2 text-xs">
            <span className="truncate text-slate-600" title={s.model}>{s.model}</span>
            <div className="h-4 rounded bg-slate-100">
              <div className="h-4 rounded bg-indigo-500" style={{ width: `${(value(s) / top) * 100}%` }} />
            </div>
            <span className="text-right font-mono">{format(value(s))}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

const pct = (n: number) => `${Math.round(n * 100)}%`;

export default function Evals() {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.latestEvals().then(setReport).catch((e: Error) => setError(e.message));
  }, []);

  if (error)
    return <ErrorBox message={`${error}. Run the benchmark first: python -m evals.run_evals`} />;
  if (!report) return <p className="text-slate-500">Loading…</p>;

  const models = report.summary;
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Benchmark results</h1>
        <p className="text-sm text-slate-500">
          {report.results.length} task runs · generated {new Date(report.generated_at).toLocaleString()}
        </p>
      </div>

      <Card className="overflow-x-auto p-0">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
            <tr>
              {["Model", "Success", "Avg latency", "Avg steps", "Avg retries", "Tool error rate", "Est. cost"].map((h) => (
                <th key={h} className="px-4 py-2">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {models.map((s) => (
              <tr key={`${s.provider}/${s.model}`}>
                <td className="px-4 py-2 font-medium">{s.provider}/{s.model}</td>
                <td className="px-4 py-2">{pct(s.success_rate)}</td>
                <td className="px-4 py-2">{s.avg_latency_s.toFixed(1)} s</td>
                <td className="px-4 py-2">{s.avg_steps.toFixed(1)}</td>
                <td className="px-4 py-2">{s.avg_retries.toFixed(2)}</td>
                <td className="px-4 py-2">{pct(s.tool_error_rate)}</td>
                <td className="px-4 py-2">${s.est_cost_usd.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Bars title="Success rate" rows={models} value={(s) => s.success_rate} format={pct} max={1} />
        <Bars title="Average latency per task" rows={models} value={(s) => s.avg_latency_s} format={(n) => `${n.toFixed(1)} s`} />
      </div>

      <Card>
        <h3 className="mb-3 text-sm font-semibold text-slate-700">Most common failures</h3>
        <div className="grid gap-4 md:grid-cols-3">
          {Object.entries(report.failure_analysis).map(([model, failures]) => (
            <div key={model}>
              <p className="mb-1 text-xs font-semibold text-slate-600">{model}</p>
              {failures.length === 0 ? (
                <p className="text-xs text-emerald-700">No failures</p>
              ) : (
                <ul className="space-y-1 text-xs">
                  {failures.map((f) => (
                    <li key={f.type} className="flex justify-between">
                      <span>{f.type}</span><span className="font-mono">{f.count}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      </Card>

      <details>
        <summary className="cursor-pointer text-sm font-semibold text-slate-700">All task results</summary>
        <Card className="mt-2 overflow-x-auto p-0">
          <table className="w-full text-xs">
            <thead className="bg-slate-50 text-left uppercase text-slate-500">
              <tr>{["Task", "Model", "Result", "Check", "Latency", "Steps", "Failure"].map((h) => <th key={h} className="px-3 py-2">{h}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {report.results.map((r, i) => (
                <tr key={i}>
                  <td className="px-3 py-1.5 font-mono">{r.task_id}</td>
                  <td className="px-3 py-1.5">{r.model}</td>
                  <td className={`px-3 py-1.5 font-semibold ${r.passed ? "text-emerald-700" : "text-red-700"}`}>{r.passed ? "pass" : "fail"}</td>
                  <td className="px-3 py-1.5">{r.check}</td>
                  <td className="px-3 py-1.5">{r.latency_s.toFixed(1)} s</td>
                  <td className="px-3 py-1.5">{r.steps}</td>
                  <td className="px-3 py-1.5 text-slate-500">{r.failure_type ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </details>
    </div>
  );
}
