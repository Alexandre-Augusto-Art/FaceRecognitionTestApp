<#
.SYNOPSIS
    Prepara o ambiente do testbed: venv, dependencias e modelos.

.DESCRIPTION
    O ambiente virtual e os pesos dos modelos ficam em %LOCALAPPDATA%\HyperFaceTestbed,
    fora da pasta do projeto. O projeto costuma viver em uma pasta do Google Drive,
    onde escrever milhares de arquivos pequenos e lento e consome cota de
    sincronizacao a toa. Os modelos sao sempre re-baixaveis.

    Defina a variavel de ambiente HFT_HOME para escolher outro diretorio.

.PARAMETER Core
    Instala apenas as dependencias core (Haar, LBP, YuNet).

.PARAMETER SkipModels
    Nao baixa os pesos dos modelos.

.EXAMPLE
    .\setup.ps1
    Instalacao completa: venv, dependencias core, extras e todos os modelos.

.EXAMPLE
    .\setup.ps1 -Core -SkipModels
    Instalacao minima, sem extras e sem baixar pesos.
#>
[CmdletBinding()]
param(
    [switch]$Core,
    [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot

if ($env:HFT_HOME) { $appHome = $env:HFT_HOME }
else { $appHome = Join-Path $env:LOCALAPPDATA 'HyperFaceTestbed' }

$venvPath = Join-Path $appHome 'venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

Write-Host ''
Write-Host '=== Testbed de deteccao facial - setup ===' -ForegroundColor Cyan
Write-Host "Projeto : $projectRoot"
Write-Host "Ambiente: $appHome"
Write-Host ''

function Find-BasePython {
    foreach ($candidate in @('C:\Python314\python.exe', 'C:\Python313\python.exe', 'C:\Python312\python.exe')) {
        if (Test-Path $candidate) { return $candidate }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw 'Python nao encontrado. Instale Python 3.11 ou mais recente.'
}

if (-not (Test-Path $venvPython)) {
    $basePython = Find-BasePython
    Write-Host "Criando ambiente virtual com $basePython ..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $appHome | Out-Null
    & $basePython -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar o ambiente virtual.' }
    & $venvPython -m pip install --upgrade pip --quiet
} else {
    Write-Host 'Ambiente virtual ja existe.' -ForegroundColor DarkGray
}

Write-Host ''
Write-Host 'Instalando dependencias core ...' -ForegroundColor Yellow
& $venvPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar as dependencias core.' }

if (-not $Core) {
    Write-Host ''
    Write-Host 'Instalando extras (SCRFD, UltraFace, dlib, MediaPipe) ...' -ForegroundColor Yellow
    & $venvPython -m pip install -r (Join-Path $projectRoot 'requirements-extras.txt')
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Algum extra falhou. O app continua funcionando; os detectores afetados aparecerao como indisponiveis.'
    }
}

if (-not $SkipModels) {
    Write-Host ''
    Write-Host 'Baixando modelos ...' -ForegroundColor Yellow
    Push-Location $projectRoot
    try { & $venvPython -m app.downloader }
    finally { Pop-Location }
}

Write-Host ''
Write-Host 'Pronto. Rode .\run.ps1 para iniciar o app.' -ForegroundColor Green
Write-Host ''
