# Deploy AgentLite to Azure Container Apps, using Azure OpenAI as the model.
#
# COST NOTES - read before running:
# - Azure OpenAI has NO free tier: you pay per token. With gpt-4.1-nano a demo run
#   costs a fraction of a cent, and the full 18-task eval suite costs a few cents.
# - Container Apps (Consumption plan) includes a monthly free grant (180,000 vCPU-s,
#   360,000 GiB-s, 2M requests). This app scales to zero and caps at 1 replica.
# - We skip Azure Container Registry (Basic is ~$5/month even when idle) and pull a
#   public image from GitHub Container Registry (free) instead.
# - We skip Log Analytics (--logs-destination none), which bills per GB ingested.
#
# Before running:
#   1. Install Docker Desktop and the Azure CLI, then: az login
#   2. Build and push the image to GHCR (free), using a GitHub token with write:packages:
#        docker build -t ghcr.io/<github-user>/agentlite:latest .
#        echo <token> | docker login ghcr.io -u <github-user> --password-stdin
#        docker push ghcr.io/<github-user>/agentlite:latest
#      Then on GitHub: Packages > agentlite > Package settings > Change visibility > Public.
#
# Usage:
#   .\deploy\azure\deploy.ps1 -Image ghcr.io/<github-user>/agentlite:latest
param(
    [Parameter(Mandatory = $true)][string]$Image,
    [string]$ResourceGroup = "agentlite-rg",
    [string]$Location = "eastus",
    [string]$OpenAIName = "agentlite-openai-$((Get-Random -Maximum 99999))",
    [string]$ModelName = "gpt-4.1-nano",       # cheapest non-reasoning chat model with tool calling
    [string]$ModelVersion = "2025-04-14",       # check with: az cognitiveservices account list-models
    [string]$Deployment = "gpt-41-nano",
    [string]$App = "agentlite-api"
)
$ErrorActionPreference = "Stop"

Write-Host "==> Registering resource providers (free)"
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.CognitiveServices --wait

Write-Host "==> Resource group"
az group create --name $ResourceGroup --location $Location | Out-Null

Write-Host "==> Azure OpenAI resource + $ModelName deployment (pay-per-token, small capacity)"
az cognitiveservices account create --name $OpenAIName --resource-group $ResourceGroup `
    --location $Location --kind OpenAI --sku S0 --custom-domain $OpenAIName --yes | Out-Null
az cognitiveservices account deployment create --name $OpenAIName --resource-group $ResourceGroup `
    --deployment-name $Deployment --model-name $ModelName --model-version $ModelVersion `
    --model-format OpenAI --sku-name GlobalStandard --sku-capacity 10 | Out-Null

$endpoint = az cognitiveservices account show --name $OpenAIName --resource-group $ResourceGroup `
    --query properties.endpoint -o tsv
$key = az cognitiveservices account keys list --name $OpenAIName --resource-group $ResourceGroup `
    --query key1 -o tsv

Write-Host "==> Container Apps environment (no Log Analytics, to avoid ingestion charges)"
az containerapp env create --name agentlite-env --resource-group $ResourceGroup `
    --location $Location --logs-destination none | Out-Null

Write-Host "==> Container App (scale to zero, max 1 replica)"
az containerapp create --name $App --resource-group $ResourceGroup --environment agentlite-env `
    --image $Image --target-port 8080 --ingress external `
    --cpu 0.5 --memory 1Gi --min-replicas 0 --max-replicas 1 `
    --secrets "azure-openai-key=$key" `
    --env-vars "LLM_PROVIDER=azure_openai" "LLM_MODEL=$Deployment" `
        "AZURE_OPENAI_API_KEY=secretref:azure-openai-key" "AZURE_OPENAI_ENDPOINT=$endpoint" `
        "AZURE_OPENAI_DEPLOYMENT=$Deployment" "AZURE_OPENAI_API_VERSION=2024-10-21" "RUN_WORKERS=2" | Out-Null

$fqdn = az containerapp show --name $App --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
Write-Host ""
Write-Host "==> Deployed: https://$fqdn"
Write-Host "    The URL is public. Anyone who finds it can spend your Azure OpenAI tokens;"
Write-Host "    run teardown.ps1 when you're done demoing."
Write-Host ""
Write-Host "To run the evals against Azure from your PC, add to .env:"
Write-Host "    AZURE_OPENAI_ENDPOINT=$endpoint"
Write-Host "    AZURE_OPENAI_DEPLOYMENT=$Deployment"
Write-Host "    AZURE_OPENAI_API_KEY=(az cognitiveservices account keys list -n $OpenAIName -g $ResourceGroup --query key1 -o tsv)"
