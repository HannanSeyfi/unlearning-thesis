$ErrorActionPreference = 'Stop'

$ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = (Resolve-Path (Join-Path $ToolDir '..\..')).Path
$PythonExe = 'C:\Users\hanna\anaconda3\envs\google-colab\python.exe'
$JupyterExe = 'C:\Users\hanna\anaconda3\envs\google-colab\Scripts\jupyter-notebook.exe'
$Port = 8888
$LogDir = Join-Path $ProjectDir '.local-runtime'
$StdoutLog = Join-Path $LogDir 'jupyter.stdout.log'
$StderrLog = Join-Path $LogDir 'jupyter.stderr.log'
$UrlFile = Join-Path $ProjectDir 'local_runtime_url.txt'
$PidFile = Join-Path $LogDir 'jupyter.pid'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python environment not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $JupyterExe)) {
    throw "Jupyter Notebook not found: $JupyterExe"
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$existingListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existingListener) {
    throw "Port $Port is already in use. Stop that process before starting this runtime."
}

$Token = & $PythonExe -c "import secrets; print(secrets.token_urlsafe(32))"
$RuntimeUrl = "http://127.0.0.1:${Port}/?token=$Token"

$Arguments = @(
    '--no-browser'
    "--port=$Port"
    '--ServerApp.port_retries=0'
    "--ServerApp.allow_origin=https://colab.research.google.com"
    '--ServerApp.allow_credentials=True'
    "--ServerApp.token=$Token"
)

$process = Start-Process `
    -FilePath $JupyterExe `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

$process.Id | Set-Content -LiteralPath $PidFile -Encoding ascii
$RuntimeUrl | Set-Content -LiteralPath $UrlFile -Encoding ascii
Set-Clipboard -Value $RuntimeUrl

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 500
    if ($process.HasExited) {
        $details = Get-Content -LiteralPath $StderrLog -Raw -ErrorAction SilentlyContinue
        throw "Jupyter exited before becoming ready.`n$details"
    }
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        Write-Host ''
        Write-Host 'Local Colab runtime is ready:' -ForegroundColor Green
        Write-Host $RuntimeUrl -ForegroundColor Cyan
        Write-Host ''
        Write-Host 'The URL was copied to your clipboard and saved to:'
        Write-Host $UrlFile
        Write-Host ''
        Write-Host 'In Colab: Connect -> Connect to local runtime -> paste the URL.'
        exit 0
    }
}

throw "Jupyter did not start within 15 seconds. Check $StderrLog"
