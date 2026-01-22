# PowerShell Script to Force Update and Restart GAUL Experiment
# Usage: .\scripts\force_update_gaul.ps1 -Username <your_gaul_username>

param (
    [Parameter(Mandatory=$true)]
    [string]$Username
)

$HostName = "gpu1.gaul.csd.uwo.ca"
$RemoteDir = "~/ai-law-agents"

Write-Host "🚀 Connecting to $Username@$HostName to force update..." -ForegroundColor Cyan

# The command sequence:
# 1. Kill existing python processes
# 2. Go to dir
# 3. Fetch latest code
# 4. Reset hard to origin/gaul-experiments (Wipes local changes/mistakes)
# 5. Run the experiment script in background
$RemoteCommand = "killall python; cd $RemoteDir && git fetch origin && git reset --hard origin/gaul-experiments && ./scripts/run_gaul_experiments.sh"

ssh "$Username@$HostName" $RemoteCommand

Write-Host "✅ Update and Restart Command Sent!" -ForegroundColor Green
Write-Host "   The experiment should be running in the background now."
Write-Host "   You can verify by running: ssh $Username@$HostName 'tail -f $RemoteDir/runs/execution_*.log'"
