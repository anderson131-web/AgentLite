# Remove everything deploy.ps1 created, so nothing keeps costing money.
#
# Usage:
#   .\deploy\gcp\teardown.ps1 -ProjectId my-agentlite-123
#   .\deploy\gcp\teardown.ps1 -ProjectId my-agentlite-123 -DeleteProject   # removes EVERYTHING in the project
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "us-central1",
    [string]$Service = "agentlite",
    [switch]$DeleteProject
)

if ($DeleteProject) {
    Write-Host "==> Deleting the whole project $ProjectId (recoverable for 30 days)"
    gcloud projects delete $ProjectId
    exit
}

Write-Host "==> Deleting the Cloud Run service"
gcloud run services delete $Service --region $Region --project $ProjectId --quiet

Write-Host "==> Deleting the container images (Artifact Registry storage is the only ongoing cost)"
gcloud artifacts repositories delete cloud-run-source-deploy --location $Region --project $ProjectId --quiet

Write-Host "==> Deleting the Gemini key secret"
gcloud secrets delete gemini-api-key --project $ProjectId --quiet

Write-Host "==> Deleting Cloud Build source uploads"
gcloud storage rm --recursive "gs://${ProjectId}_cloudbuild" 2>$null

Write-Host "Done. The budget (if you made one) stays as a harmless alert; delete it in the Billing console if you like."
