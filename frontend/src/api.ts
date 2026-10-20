// Typed wrappers around the FastAPI endpoints. Shapes mirror api/schemas.py.

export type ToolCall = {
  name: string;
  args: Record<string, unknown>;
  output: string;
  error: boolean;
  latency_ms: number;
};

export type TraceEntry = {
  agent: "planner" | "executor" | "reviewer" | "finalize" | string;
  input: string;
  output: string;
  tool_calls: ToolCall[];
  provider: string;
  model: string;
  input_tokens: number | null;
  output_tokens: number | null;
  latency_ms: number;
  error: string | null;
  timestamp: string;
};

export type RunStatus = "queued" | "running" | "completed" | "failed";

export type Run = {
  id: string;
  task: string;
  mode: "general" | "job";
  provider: string;
  model: string;
  status: RunStatus;
  plan: string[];
  final_answer: string | null;
  retries: number | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
  trace: TraceEntry[];
  total_latency_ms?: number | null;
  trace_count?: number | null;
};

export type RunRequest = {
  mode: "general" | "job";
  task?: string;
  provider?: string;
  model?: string;
  job_posting_url?: string;
  job_posting_text?: string;
  resume_text?: string;
};

export type ModelsResponse = {
  default: { provider: string; model: string };
  providers: Record<string, string[]>;
};

export type EvalSummary = {
  provider: string;
  model: string;
  tasks: number;
  success_rate: number;
  avg_latency_s: number;
  avg_steps: number;
  avg_retries: number;
  tool_error_rate: number;
  est_cost_usd: number;
};

export type EvalResult = {
  task_id: string;
  category: string;
  provider: string;
  model: string;
  passed: boolean;
  check: string;
  latency_s: number;
  steps: number;
  retries: number;
  tool_calls: number;
  tool_errors: number;
  failure_type: string | null;
};

export type EvalReport = {
  generated_at: string;
  summary: EvalSummary[];
  results: EvalResult[];
  failure_analysis: Record<string, { type: string; count: number }[]>;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      // FastAPI sends {detail: "..."} for HTTPException and {detail: [...]} for validation errors.
      message = typeof body.detail === "string" ? body.detail : body.detail?.[0]?.msg ?? message;
    } catch {
      /* body wasn't JSON */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  models: () => request<ModelsResponse>("/models"),
  createRun: (body: RunRequest) => request<{ run_id: string; status: string }>("/runs", json(body)),
  getRun: (id: string) => request<Run>(`/runs/${id}`),
  listRuns: () => request<Run[]>("/runs"),
  ingest: () => request<{ files: number; chunks: number }>("/ingest", { method: "POST" }),
  latestEvals: () => request<EvalReport>("/evals/latest"),
  uploadResume: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ filename: string; text: string }>("/uploads/resume", { method: "POST", body: form });
  },
  streamUrl: (id: string) => `/api/runs/${id}/stream`,
};
