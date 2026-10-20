"""Job Research flow: compare a job posting with the user's resume.

Design choice: this flow uses a FIXED plan instead of asking the planner.
It always needs the same four steps, and a 3B planner sometimes skips or merges
them. The executor and reviewer still run normally, so it's the same graph.

Inputs are staged as files in the workspace (job_posting.txt, resume.txt) and
the steps tell the executor exactly which tool to call. That keeps the task
prompt short. Each step's prompt only includes what it needs, which matters
with an 8k context window.

Safety: AgentLite never submits applications. No tool can POST or fill forms,
and the final step says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentlite.config import get_settings
from agentlite.documents import load_text

JOB_FILE = "job_posting.txt"
RESUME_FILE = "resume.txt"
REPORT_FILE = "job_report.md"


@dataclass
class JobResearch:
    task: str
    plan: list[str]


def build_job_research(
    posting_url: str | None = None,
    posting_text: str | None = None,
    resume_text: str | None = None,
    resume_path: Path | None = None,
) -> JobResearch:
    """Stage the inputs in the workspace and return the task plus its fixed plan."""
    if not posting_url and not posting_text:
        raise ValueError("Give a job posting URL or the posting text.")
    workspace = get_settings().workspace_dir
    workspace.mkdir(parents=True, exist_ok=True)

    if posting_text:
        (workspace / JOB_FILE).write_text(posting_text.strip(), encoding="utf-8")
        read_posting = f'Use read_file with path "{JOB_FILE}" to read the job posting.'
        source = "the job posting saved in the workspace"
    else:
        read_posting = f'Use fetch_page with url "{posting_url}" to read the job posting.'
        source = posting_url

    if resume_path is not None:
        resume_text = load_text(resume_path)
    if resume_text:
        (workspace / RESUME_FILE).write_text(resume_text.strip(), encoding="utf-8")
        read_resume = f'Use read_file with path "{RESUME_FILE}" to read the resume.'
    else:
        read_resume = 'Use knowledge_search with query "resume skills experience education" to find the resume.'

    task = (
        f"Job research for {source}: research the company, extract the job requirements, compare them "
        "with my resume (matching skills and gaps), and write interview talking points and a short cover letter. "
        "Do not submit any application."
    )
    plan = [
        f"{read_posting} Then list: company name, job title, required skills, nice-to-have skills, and main responsibilities.",
        "Use web_search to research the company from step 1: what it does, its main products, and any recent news. "
        "Summarize in 3-5 bullet points.",
        f"{read_resume} Compare it with the required and nice-to-have skills from step 1. "
        "List MATCHING SKILLS (with evidence from the resume) and GAPS (skills the resume does not show).",
        "Write the final report in Markdown with these sections: ## Company, ## Role Requirements, "
        "## Matching Skills, ## Gaps, ## Interview Talking Points (5 bullets that connect resume evidence to the role), "
        "## Cover Letter (under 200 words, specific to this company and role). "
        "Only use facts from the earlier steps. This is a draft for the user; do not apply or submit anything.",
    ]
    return JobResearch(task=task, plan=plan)


def save_report(answer: str) -> Path:
    path = get_settings().workspace_dir / REPORT_FILE
    path.write_text(answer, encoding="utf-8")
    return path
