# Deploy AgentLite (API + frontend, one container) to Google Cloud Run, using Gemini.
#
# COST NOTES - read before running:
# - Cloud Run, Cloud Build, Artifact Registry, Secret Manager and BigQuery all have
#   monthly free tiers. This script sets max-instances=1 and scale-to-zero so
#   normal demo use stays inside them.
# - The container image (~1.5 GB) can exceed Artifact Registry's 0.5 GB free
#   storage: roughly $0.10/month until you run teardown.ps1.
# - Make your Gemini API key in a Google Cloud project WITHOUT billing enabled
#   (AI Studio > Get API key > create in a new project). A key from a project
#   without billing can only use the free tier, so it can never be charged.
# - A budget alert EMAILS you; it does not stop spending. teardown.ps1 does.
#
# Usage (PowerShell, from the repo root):
#   .\deploy\gcp\deploy.ps1 -ProjectId my-agentlite-123 -GeminiApiKey "AIza..."
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$GeminiApiKey,
    [string]$Region = "us-central1",          # a Tier 1 region (cheapest, free tier applies)
    [string]$Service = "agentlite",
    [string]$GeminiModel = "gemini-3.8-flash"
)
$ErrorActionPreference = "Stop"

Write-Host "==> Using project $ProjectId in $Region"
gcloud config set project $ProjectId
gcloud config set run/region $Region

Write-Host "==> Enabling APIs (free to enable)"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com `
    secretmanager.googleapis.com bigquery.googleapis.com

Write-Host "==> Storing the Gemini key in Secret Manager (never in env files or images)"
$exists = gcloud secrets list --filter="name:gemini-api-key" --format="value(name)"
if (-not $exists) {
    gcloud secrets create gemini-api-key --replication-policy=automatic
}
$tmp = New-TemporaryFile
Set-Content -Path $tmp -Value $GeminiApiKey -NoNewline
gcloud secrets versions add gemini-api-key --data-file=$tmp
Remove-Item $tmp

# Cloud Run's default service account must be able to read the secret and run BigQuery jobs.
$projectNumber = gcloud projects describe $ProjectId --format="value(projectNumber)"
$serviceAccount = "$projectNumber-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding gemini-api-key `
    --member="serviceAccount:$serviceAccount" --role="roles/secretmanager.secretAccessor" | Out-Null
gcloud projects add-iam-policy-binding $ProjectId `
    --member="serviceAccount:$serviceAccount" --role="roles/bigquery.jobUser" --condition=None | Out-Null

Write-Host "==> Building with Cloud Build and deploying to Cloud Run (takes ~5-10 minutes)"
# --source . builds the Dockerfile remotely, so you don't need Docker installed.
# --no-allow-unauthenticated keeps the service private (see the README to open it up).
gcloud run deploy $Service `
    --source . `
    --region $Region `
    --no-allow-unauthenticated `
    --memory 1Gi --cpu 1 `
    --min-instances 0 --max-instances 1 `
    --timeout 900 `
    --set-env-vars "LLM_PROVIDER=gemini,LLM_MODEL=$GeminiModel,GEMINI_MODEL=$GeminiModel,EMBED_PROVIDER=gemini,EMBED_MODEL=gemini-embedding-001,RUN_WORKERS=2,GOOGLE_CLOUD_PROJECT=$ProjectId,BIGQUERY_MAX_BYTES=1000000000" `
    --set-secrets "GOOGLE_API_KEY=gemini-api-key:latest"

Write-Host ""
Write-Host "==> Deployed. The service is private. Open it through an authenticated local proxy:"
Write-Host "    gcloud run services proxy $Service --region $Region --port 8080"
Write-Host "    then browse to http://localhost:8080"
