param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

function Test-TumorMeshServer {
    param([int]$CandidatePort)

    try {
        $status = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$CandidatePort/api/status" `
            -TimeoutSec 2
        return $status.PSObject.Properties.Name -contains "checkpoint_ready"
    }
    catch {
        return $false
    }
}

function Test-PortAvailable {
    param([int]$CandidatePort)

    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $CandidatePort
    )

    try {
        $listener.Start()
        return $true
    }
    catch [System.Net.Sockets.SocketException] {
        return $false
    }
    finally {
        $listener.Stop()
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Khong tim thay .venv. Hay tao moi truong ao va cai requirements.txt truoc."
}

$lastPort = [Math]::Min($Port + 10, 65535)
$selectedPort = $null

for ($candidatePort = $Port; $candidatePort -le $lastPort; $candidatePort++) {
    if (Test-TumorMeshServer -CandidatePort $candidatePort) {
        Write-Host "TumorMesh Studio dang chay tai:"
        Write-Host "http://127.0.0.1:$candidatePort"
        exit 0
    }

    if (Test-PortAvailable -CandidatePort $candidatePort) {
        $selectedPort = $candidatePort
        break
    }
}

if ($null -eq $selectedPort) {
    throw "Khong tim thay cong trong tu $Port den $lastPort."
}

if ($selectedPort -ne $Port) {
    Write-Warning "Cong $Port dang duoc su dung. Chuyen sang cong $selectedPort."
}

Set-Location $projectRoot
Write-Host "Khoi dong TumorMesh Studio tai:"
Write-Host "http://127.0.0.1:$selectedPort"
& $python -m uvicorn webapp.app:app --host 127.0.0.1 --port $selectedPort
