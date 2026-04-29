# Stop GCP Training VM
# Usage: .\scripts\stop_training_vm.ps1

$projectName = "nt549-494407"
$zone = "asia-east1-a"
$instanceName = "rl-training-vm"

Write-Host "🛑 Stopping GCP Training VM: $instanceName..." -ForegroundColor Cyan

gcloud compute instances stop $instanceName --zone $zone --project $projectName

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ VM stopped successfully. Costs for compute are now paused." -ForegroundColor Green
} else {
    Write-Host "❌ Failed to stop VM." -ForegroundColor Red
}
