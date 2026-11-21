# Create a $1 monthly budget on the subscription with email alerts at 50% and 100% of
# actual spend, plus 100% of forecast spend. A budget only WARNS; it does not stop
# spending. Run teardown.ps1 to stop costs.
#
# Usage:
#   .\deploy\azure\budget.ps1 -Email you@example.com
param(
    [Parameter(Mandatory = $true)][string]$Email,
    [string]$Name = "agentlite-1usd"
)
$ErrorActionPreference = "Stop"

$subscription = az account show --query id -o tsv
$start = (Get-Date -Day 1).ToString("yyyy-MM-01T00:00:00Z")      # budgets must start on the 1st
$end = (Get-Date -Day 1).AddYears(1).ToString("yyyy-MM-01T00:00:00Z")

function Rule([int]$threshold, [string]$type) {
    @{ enabled = $true; operator = "GreaterThanOrEqualTo"; threshold = $threshold;
       thresholdType = $type; contactEmails = @($Email) }
}

$body = @{
    properties = @{
        category = "Cost"
        amount = 1
        timeGrain = "Monthly"
        timePeriod = @{ startDate = $start; endDate = $end }
        notifications = @{
            actual50 = Rule 50 "Actual"
            actual100 = Rule 100 "Actual"
            forecast100 = Rule 100 "Forecasted"
        }
    }
} | ConvertTo-Json -Depth 6

$file = New-TemporaryFile
Set-Content -Path $file -Value $body -Encoding utf8
az rest --method put `
    --url "https://management.azure.com/subscriptions/$subscription/providers/Microsoft.Consumption/budgets/${Name}?api-version=2023-05-01" `
    --body "@$file" | Out-Null
Remove-Item $file
Write-Host "Budget '$Name' created: alerts to $Email. View it under Cost Management > Budgets."
