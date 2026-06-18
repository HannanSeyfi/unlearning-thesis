$ErrorActionPreference = 'Stop'

$ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = (Resolve-Path (Join-Path $ToolDir '..\..')).Path
$PidFile = Join-Path $ProjectDir '.local-runtime\jupyter.pid'

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host 'No saved local-runtime process was found.'
    exit 0
}

$processId = [int](Get-Content -LiteralPath $PidFile -Raw)
$process = Get-Process -Id $processId -ErrorAction SilentlyContinue

if ($process) {
    Stop-Process -Id $processId
    Write-Host "Stopped local Colab runtime process $processId."
} else {
    Write-Host "Process $processId is no longer running."
}

Remove-Item -LiteralPath $PidFile -Force
