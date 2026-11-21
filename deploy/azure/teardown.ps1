# Remove everything deploy.ps1 created.
#
# Usage:
#   .\deploy\azure\teardown.ps1
param(
    [string]$ResourceGroup = "agentlite-rg",
    [string]$Location = "eastus",
    [switch]$DeleteBudget
)

# Remember the OpenAI resource name before the group disappears; Azure soft-deletes
# it for 48 hours, and purging frees the name (soft-deleted resources aren't billed).
$openai = az cognitiveservices account list --resource-group $ResourceGroup --query "[].name" -o tsv

Write-Host "==> Deleting resource group $ResourceGroup (Container App, environment, Azure OpenAI)"
az group delete --name $ResourceGroup --yes

foreach ($name in $openai) {
    Write-Host "==> Purging soft-deleted Azure OpenAI resource $name"
    az cognitiveservices account purge --location $Location --resource-group $ResourceGroup --name $name
}

if ($DeleteBudget) {
    $subscription = az account show --query id -o tsv
    az rest --method delete `
        --url "https://management.azure.com/subscriptions/$subscription/providers/Microsoft.Consumption/budgets/agentlite-1usd?api-version=2023-05-01"
    Write-Host "==> Budget deleted"
}
Write-Host "Done. Also delete the public image on GitHub (Packages > agentlite) if you no longer need it."
