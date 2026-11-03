<#
.SYNOPSIS
    Inicia o servidor local do testbed e abre o navegador.

.PARAMETER Port
    Porta do servidor. Padrao 8010.

.PARAMETER NoBrowser
    Nao abre o navegador automaticamente.

.PARAMETER Reload
    Recarrega o servidor quando o codigo muda (util ao desenvolver).
#>
[CmdletBinding()]
param(
    [int]$Port = 8010,
    [switch]$NoBrowser,
    [switch]$Reload
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot

if ($env:HFT_HOME) { $appHome = $env:HFT_HOME }
else { $appHome = Join-Path $env:LOCALAPPDATA 'HyperFaceTestbed' }

$venvPython = Join-Path $appHome 'venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    Write-Host 'Ambiente nao encontrado. Rode .\setup.ps1 primeiro.' -ForegroundColor Red
    exit 1
}

$url = "http://127.0.0.1:$Port/"
Write-Host ''
Write-Host "Testbed de deteccao facial em $url" -ForegroundColor Cyan
Write-Host 'Ctrl+C encerra o servidor.' -ForegroundColor DarkGray
Write-Host ''

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($target)
        Start-Sleep -Seconds 3
        Start-Process $target
    } -ArgumentList $url | Out-Null
}

Push-Location $projectRoot
try {
    $arguments = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port")
    if ($Reload) { $arguments += '--reload' }
    & $venvPython @arguments
}
finally {
    Pop-Location
}
