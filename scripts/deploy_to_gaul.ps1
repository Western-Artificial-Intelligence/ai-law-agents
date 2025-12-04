# PowerShell Script to deploy to GAUL
# Usage: .\scripts\deploy_to_gaul.ps1 -Username <your_gaul_username>

param (
    [Parameter(Mandatory=$true)]
    [string]$Username
)

$HostName = "compute.gaul.csd.uwo.ca"
$RemoteDir = "~/ai-law-agents"

Write-Host "🚀 Deploying to $Username@$HostName..." -ForegroundColor Cyan

# Create remote directory
Write-Host "📂 Creating remote directory..."
ssh "$Username@$HostName" "mkdir -p $RemoteDir"

# Define files/folders to copy (excluding .git, .venv, etc.)
$Items = @("bailiff", "configs", "scripts", "docs", "setup.py", "pyproject.toml", "README.md")

Write-Host "📦 Transferring files (this may take a moment)..."

foreach ($Item in $Items) {
    if (Test-Path $Item) {
        Write-Host "   Uploading $Item..."
        # -r for recursive, -p for preserving modification times
        scp -r -p $Item "$Username@$HostName`:$RemoteDir/"
    } else {
        Write-Warning "   Skipping $Item (not found)"
    }
}

Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "   Next steps:"
Write-Host "   1. ssh $Username@$HostName"
Write-Host "   2. cd ai-law-agents"
Write-Host "   3. chmod +x scripts/*.sh"
Write-Host "   4. ./scripts/setup_gaul.sh"
