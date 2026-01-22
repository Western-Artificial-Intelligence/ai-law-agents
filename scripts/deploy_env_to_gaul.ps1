# PowerShell Script to deploy .env.local to GAUL as .env
# Usage: .\scripts\deploy_env_to_gaul.ps1 -Username <your_gaul_username>

param (
    [Parameter(Mandatory = $true)]
    [string]$Username
)

$HostName = "gpu1.gaul.csd.uwo.ca"
$RemoteDir = "~/ai-law-agents"
$LocalEnv = ".env.local"
$RemoteEnv = "$RemoteDir/.env"

if (-not (Test-Path $LocalEnv)) {
    Write-Error "Missing $LocalEnv. Create it before deploying."
    exit 1
}

function Parse-EnvFile {
    param ([string]$Path)
    $envMap = @{}
    foreach ($line in Get-Content -Path $Path) {
        $trim = $line.Trim()
        if (-not $trim -or $trim.StartsWith("#")) {
            continue
        }
        if ($trim.StartsWith("export ")) {
            $trim = $trim.Substring(7).Trim()
        }
        $idx = $trim.IndexOf("=")
        if ($idx -lt 1) {
            continue
        }
        $key = $trim.Substring(0, $idx).Trim()
        $value = $trim.Substring($idx + 1).Trim()
        if (($value.StartsWith("'") -and $value.EndsWith("'")) -or ($value.StartsWith('"') -and $value.EndsWith('"'))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $envMap[$key] = $value
    }
    return $envMap
}

$envMap = Parse-EnvFile -Path $LocalEnv

if ($envMap.ContainsKey("GROQ_API_KEYS")) {
    try {
        $parsed = $envMap["GROQ_API_KEYS"] | ConvertFrom-Json
    } catch {
        Write-Error "GROQ_API_KEYS must be valid JSON."
        exit 1
    }
    if ($parsed -is [string]) {
        Write-Error "GROQ_API_KEYS must be a JSON list of strings."
        exit 1
    }
    foreach ($item in $parsed) {
        if (-not ($item -is [string]) -or [string]::IsNullOrWhiteSpace($item)) {
            Write-Error "GROQ_API_KEYS must be a JSON list of non-empty strings."
            exit 1
        }
    }
} elseif ($envMap.ContainsKey("GROQ_API_KEY")) {
    if ([string]::IsNullOrWhiteSpace($envMap["GROQ_API_KEY"])) {
        Write-Error "GROQ_API_KEY cannot be empty."
        exit 1
    }
} else {
    Write-Error "Missing GROQ_API_KEYS or GROQ_API_KEY in .env.local."
    exit 1
}

if ($envMap.ContainsKey("GROQ_API_KEY_CONCURRENCY")) {
    try {
        $concurrency = $envMap["GROQ_API_KEY_CONCURRENCY"] | ConvertFrom-Json
    } catch {
        Write-Error "GROQ_API_KEY_CONCURRENCY must be valid JSON."
        exit 1
    }
    if ($null -eq $concurrency -or -not ($concurrency -is [System.Collections.IDictionary])) {
        Write-Error "GROQ_API_KEY_CONCURRENCY must be a JSON object."
        exit 1
    }
    foreach ($value in $concurrency.Values) {
        try {
            $limit = [int]$value
        } catch {
            Write-Error "GROQ_API_KEY_CONCURRENCY values must be integers."
            exit 1
        }
        if ($limit -le 0) {
            Write-Error "GROQ_API_KEY_CONCURRENCY values must be positive integers."
            exit 1
        }
    }
}

Write-Host "Deploying .env.local to $Username@$HostName..." -ForegroundColor Cyan
ssh "$Username@$HostName" "mkdir -p $RemoteDir"
scp -p $LocalEnv "${Username}@${HostName}:${RemoteEnv}"
ssh "$Username@$HostName" "chmod 600 $RemoteEnv"
Write-Host "Deployment complete. Remote .env stored at $RemoteEnv with permissions 600." -ForegroundColor Green
