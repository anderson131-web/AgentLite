// One run: live trace timeline (via Server-Sent Events) and the final answer.
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type Run, type TraceEntry } from "../api";
import TraceCard from "../components/TraceCard";
import { Card, ErrorBox, formatMs, secondaryButtonClass, StatusBadge } from "../components/ui";

export default function RunDetail() {
  const { id = "" } = useParams();
  const [run, setRun] = useState<Run | null>(null);
  const [trace, setTrace] = useState<TraceEntry[]>([]);
  const [error, setError] = useState("");
  const [now, setNow] = useState(Date.now());
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setTrace([]);
    api.getRun(id).then(setRun).catch((e: Error) => setError(e.message));

    // One code path for live and finished runs: the stream replays past steps,
    // then sends new ones, then a final "done" event with the whole run.
    const source = new EventSource(api.streamUrl(id));
    source.addEventListener("trace", (event) => {
      const entry = JSON.parse((event as MessageEvent).data) as TraceEntry;
      setTrace((current) => [...current, entry]);
      setRun((current) => (current && current.status === "queued" ? { ...current, status: "running" } : current));
    });
    source.addEventListener("done", (event) => {
      setRun(JSON.parse((event as MessageEvent).data) as Run);
      source.close();
    });
    return () => source.close();
  }, [id]);

  const active = run && (run.status === "queued" || run.status === "running");
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);

  if (error) return <ErrorBox message={error} />;
  if (!run) return <p className="text-slate-500">Loading…</p>;

  const elapsedS = Math.round((now - new Date(run.created_at).getTime()) / 1000);
  const totalMs = trace.reduce((sum, e) => sum + e.latency_ms, 0);

  async function copyAnswer() {
    await navigator.clipboard.writeText(run?.final_answer ?? "");
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold break-words">{run.task}</h1>
          <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-slate-500">
            <StatusBadge status={run.status} />
            <span>{run.provider}/{run.model}</span>
            <span>· {active ? `${elapsedS}s elapsed` : `${formatMs(totalMs)} total`}</span>
            {run.retries ? <span>· {run.retries} revision(s)</span> : null}
          </p>
        </div>
        <Link to="/" className={secondaryButtonClass}>New run</Link>
      </div>

      {run.error && <ErrorBox message={run.error} />}

      {run.final_answer && (
        <Card className="border-emerald-300">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="font-semibold text-emerald-800">Final answer</h2>
            <button className={secondaryButtonClass} onClick={copyAnswer}>{copied ? "Copied" : "Copy"}</button>
          </div>
          <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">{run.final_answer}</pre>
        </Card>
      )}

      <h2 className="pt-2 text-sm font-semibold uppercase tracking-wide text-slate-500">Trace</h2>
      <div className="space-y-3">
        {trace.map((entry, i) => <TraceCard key={i} entry={entry} />)}
        {active && (
          <div className="flex items-center gap-2 rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-500">
            <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
            {run.status === "queued" ? "Waiting for the model…" : "Working on the next step…"}
          </div>
        )}
      </div>
    </div>
  );
}
