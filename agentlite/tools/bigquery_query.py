"""bigquery_query tool: read-only SQL over Google BigQuery public datasets.

Cost safety (BigQuery bills by bytes scanned, and the free tier is 1 TB/month):
- Every query runs with `maximum_bytes_billed` (default 1 GB). BigQuery itself
  refuses any query that would scan more, so it is never billed.
- A dry run first reports the bytes the query WOULD scan, and we refuse
  before running if it's over the cap, with a message the model can act on.
- Only SELECT/WITH statements are accepted, and results are capped at 50 rows.

Enabled only when GOOGLE_CLOUD_PROJECT is set. Auth uses Application Default
Credentials: `gcloud auth application-default login` locally, or the service
account automatically on Cloud Run.
"""

from __future__ import annotations

from langchain_core.tools import tool

from agentlite.config import get_settings
from agentlite.tools.sql_query import check_query

MAX_ROWS = 50
TIMEOUT_S = 60


@tool(parse_docstring=True)
def bigquery_query(query: str) -> str:
    """Run a read-only GoogleSQL SELECT query on BigQuery public datasets and return the rows.

    Use fully qualified table names in backticks, for example
    `bigquery-public-data.usa_names.usa_1910_current` (columns: state, gender, year, name, number)
    or `bigquery-public-data.github_repos.licenses` (columns: repo_name, license).
    Select only the columns you need and add LIMIT: large scans are refused.

    Args:
        query: One GoogleSQL SELECT statement.
    """
    problem = check_query(query)
    if problem:
        return f"ERROR: {problem}."

    try:
        from google.cloud import bigquery
    except ImportError:
        return "ERROR: google-cloud-bigquery is not installed."

    settings = get_settings()
    cap = settings.bigquery_max_bytes
    try:
        client = bigquery.Client(project=settings.gcp_project)
        dry = client.query(query, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
        scanned = dry.total_bytes_processed or 0
        if scanned > cap:
            return (
                f"ERROR: this query would scan {scanned / 1e9:.2f} GB, over the {cap / 1e9:.2f} GB limit. "
                "Select fewer columns or filter on a partition/cluster column."
            )
        job = client.query(query, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=cap))
        rows = list(job.result(timeout=TIMEOUT_S, max_results=MAX_ROWS + 1))
    except Exception as exc:  # auth, syntax, quota, timeout
        return f"ERROR: BigQuery query failed: {exc}"

    if not rows:
        return "Query returned no rows."
    columns = list(rows[0].keys())
    lines = [" | ".join(columns)]
    lines += [" | ".join(str(row[c]) for c in columns) for row in rows[:MAX_ROWS]]
    if len(rows) > MAX_ROWS:
        lines.append(f"...(more than {MAX_ROWS} rows; add LIMIT or aggregate)")
    lines.append(f"[scanned {scanned / 1e6:.1f} MB]")
    return "\n".join(lines)
