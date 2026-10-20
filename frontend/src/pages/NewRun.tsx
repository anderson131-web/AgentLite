// Start a run: a general task, or Job Research (posting + resume).
import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api, type ModelsResponse, type RunRequest } from "../api";
import { buttonClass, Card, ErrorBox, inputClass, secondaryButtonClass } from "../components/ui";

const EXAMPLES = [
  "Find the top 3 Python web frameworks, compare their GitHub stars, and save a summary to frameworks.md",
  "Which product category had the highest total revenue in the sales database?",
  "What is LangGraph and which company maintains it?",
];

export default function NewRun() {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [mode, setMode] = useState<"general" | "job">("general");
  const [task, setTask] = useState("");

  // Job Research fields
  const [postingSource, setPostingSource] = useState<"url" | "text">("url");
  const [postingUrl, setPostingUrl] = useState("");
  const [postingText, setPostingText] = useState("");
  const [resume, setResume] = useState<{ filename: string; text: string } | null>(null);
  const [ingestMessage, setIngestMessage] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .models()
      .then((m) => {
        setModels(m);
        setProvider(m.default.provider);
        setModel(m.default.model);
      })
      .catch((e: Error) => setError(`Could not load models: ${e.message}`));
  }, []);

  const providerModels = models?.providers[provider] ?? [];

  async function handleResume(file: File | undefined) {
    if (!file) return;
    setError("");
    try {
      setResume(await api.uploadResume(file));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function handleIngest() {
    setIngestMessage("Indexing…");
    try {
      const stats = await api.ingest();
      setIngestMessage(`Indexed ${stats.chunks} chunks from ${stats.files} files.`);
    } catch (e) {
      setIngestMessage((e as Error).message);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    const body: RunRequest = { mode, provider, model };
    if (mode === "general") body.task = task;
    else {
      if (postingSource === "url") body.job_posting_url = postingUrl;
      else body.job_posting_text = postingText;
      if (resume) body.resume_text = resume.text;
    }
    try {
      const { run_id } = await api.createRun(body);
      navigate(`/runs/${run_id}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  const canSubmit =
    !busy && model && (mode === "general" ? task.trim() : postingSource === "url" ? postingUrl.trim() : postingText.trim());

  return (
    <form onSubmit={submit} className="space-y-4">
      <div className="flex gap-2">
        {(["general", "job"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`rounded-md px-4 py-2 text-sm font-medium ${
              mode === m ? "bg-indigo-600 text-white" : "bg-white text-slate-700 ring-1 ring-slate-300 hover:bg-slate-100"
            }`}
          >
            {m === "general" ? "General task" : "Job Research"}
          </button>
        ))}
      </div>

      <Card className="space-y-4">
        {mode === "general" ? (
          <div className="space-y-2">
            <label className="text-sm font-medium" htmlFor="task">Task</label>
            <textarea
              id="task"
              rows={4}
              className={inputClass}
              placeholder="Describe a multi-step task…"
              value={task}
              onChange={(e) => setTask(e.target.value)}
            />
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((example) => (
                <button key={example} type="button" onClick={() => setTask(example)}
                  className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-700 hover:bg-slate-200">
                  {example.length > 60 ? example.slice(0, 60) + "…" : example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="rounded-md bg-indigo-50 px-3 py-2 text-sm text-indigo-900">
              The agents research the company, compare the posting with your resume, and draft talking points and a
              cover letter. AgentLite never submits applications.
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-4 text-sm">
                <span className="font-medium">Job posting</span>
                {(["url", "text"] as const).map((s) => (
                  <label key={s} className="flex items-center gap-1">
                    <input type="radio" checked={postingSource === s} onChange={() => setPostingSource(s)} />
                    {s === "url" ? "URL" : "Paste text"}
                  </label>
                ))}
              </div>
              {postingSource === "url" ? (
                <input className={inputClass} placeholder="https://company.com/careers/job-123"
                  value={postingUrl} onChange={(e) => setPostingUrl(e.target.value)} />
              ) : (
                <textarea rows={8} className={inputClass} placeholder="Paste the job posting here"
                  value={postingText} onChange={(e) => setPostingText(e.target.value)} />
              )}
              <p className="text-xs text-slate-500">Sites that need a login (e.g. LinkedIn) won't load. Paste the text instead.</p>
            </div>

            <div className="space-y-2">
              <span className="text-sm font-medium">Resume</span>
              <input type="file" accept=".pdf,.md,.txt" onChange={(e) => handleResume(e.target.files?.[0])}
                className="block text-sm file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm hover:file:bg-slate-200" />
              {resume ? (
                <p className="text-xs text-emerald-700">
                  Loaded {resume.filename} ({resume.text.length.toLocaleString()} characters).
                </p>
              ) : (
                <div className="flex items-center gap-3 text-xs text-slate-500">
                  <span>No file? The agents will search your documents in ./docs instead.</span>
                  <button type="button" className={secondaryButtonClass} onClick={handleIngest}>Re-index docs</button>
                  {ingestMessage && <span>{ingestMessage}</span>}
                </div>
              )}
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-end gap-3 border-t border-slate-100 pt-4">
          <label className="text-sm">
            <span className="mb-1 block font-medium">Provider</span>
            <select className={inputClass} value={provider} onChange={(e) => {
              setProvider(e.target.value);
              setModel(models?.providers[e.target.value]?.[0] ?? "");
            }}>
              {Object.keys(models?.providers ?? {}).map((p) => <option key={p}>{p}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium">Model</span>
            <select className={inputClass} value={model} onChange={(e) => setModel(e.target.value)}>
              {providerModels.length === 0 && <option value="">No models found</option>}
              {providerModels.map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
          <button type="submit" disabled={!canSubmit} className={`${buttonClass} ml-auto`}>
            {busy ? "Starting…" : "Run"}
          </button>
        </div>
      </Card>

      {error && <ErrorBox message={error} />}
    </form>
  );
}
