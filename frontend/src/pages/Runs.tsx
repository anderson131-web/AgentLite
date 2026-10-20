// Run history, newest first.
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Run } from "../api";
import { Card, ErrorBox, formatMs, StatusBadge } from "../components/ui";

export default function Runs() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.listRuns().then(setRuns).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorBox message={error} />;
  if (!runs) return <p className="text-slate-500">Loading…</p>;
  if (runs.length === 0) return <p className="text-slate-500">No runs yet. <Link className="text-indigo-600" to="/">Start one.</Link></p>;

  return (
    <Card className="overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
          <tr>
            <th className="px-4 py-2">Task</th>
            <th className="px-4 py-2">Model</th>
            <th className="px-4 py-2">Status</th>
            <th className="px-4 py-2 text-right">Steps</th>
            <th className="px-4 py-2 text-right">Time</th>
            <th className="px-4 py-2">Started</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {runs.map((run) => (
            <tr key={run.id} className="hover:bg-slate-50">
              <td className="max-w-md px-4 py-2">
                <Link to={`/runs/${run.id}`} className="line-clamp-2 text-indigo-700 hover:underline">
                  {run.mode === "job" && <span className="mr-1 rounded bg-indigo-100 px-1 text-xs">job</span>}
                  {run.task}
                </Link>
              </td>
              <td className="px-4 py-2 whitespace-nowrap text-slate-600">{run.model}</td>
              <td className="px-4 py-2"><StatusBadge status={run.status} /></td>
              <td className="px-4 py-2 text-right">{run.plan.length || "–"}</td>
              <td className="px-4 py-2 text-right whitespace-nowrap">{formatMs(run.total_latency_ms)}</td>
              <td className="px-4 py-2 whitespace-nowrap text-slate-500">{new Date(run.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
