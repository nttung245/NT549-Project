# Start GCP Training VM
# Usage: .\scripts\start_training_vm.ps1

$projectName = "nt549-494407"
$zone = "asia-east1-a"
$instanceName = "rl-training-vm"

Write-Host "🚀 Starting GCP Training VM: $instanceName..." -ForegroundColor Cyan

gcloud compute instances start $instanceName --zone $zone --project $projectName

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ VM started successfully." -ForegroundColor Green
    
    # Get External IP
    $ip = gcloud compute instances describe $instanceName --zone $zone --project $projectName --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
    
    Write-Host "`n📊 MLflow Dashboard will be available at: http://$($ip):5000" -ForegroundColor Yellow
    Write-Host "💻 To SSH into the VM: gcloud compute ssh $instanceName --zone $zone --project $projectName" -ForegroundColor Magenta
} else {
    Write-Host "❌ Failed to start VM." -ForegroundColor Red
}
