# Create a $1 monthly budget with email alerts at 50%, 90% and 100% (actual spend)
# for one project. Alerts go to the billing account's admins by email.
# A budget does NOT cap spending; it only warns you. Run teardown.ps1 to stop costs.
#
# Usage:
#   gcloud billing accounts list          # copy the ACCOUNT_ID (XXXXXX-XXXXXX-XXXXXX)
#   .\deploy\gcp\budget.ps1 -BillingAccount XXXXXX-XXXXXX-XXXXXX -ProjectId my-agentlite-123
param(
    [Parameter(Mandatory = $true)][string]$BillingAccount,
    [Parameter(Mandatory = $true)][string]$ProjectId
)
$ErrorActionPreference = "Stop"

gcloud services enable billingbudgets.googleapis.com --project $ProjectId
$projectNumber = gcloud projects describe $ProjectId --format="value(projectNumber)"

gcloud billing budgets create `
    --billing-account=$BillingAccount `
    --display-name="AgentLite 1 USD" `
    --budget-amount=1USD `
    --filter-projects="projects/$projectNumber" `
    --threshold-rule=percent=0.5 `
    --threshold-rule=percent=0.9 `
    --threshold-rule=percent=1.0

Write-Host "Budget created. Check it at https://console.cloud.google.com/billing/$BillingAccount/budgets"
