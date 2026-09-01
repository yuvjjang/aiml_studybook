# render-check.ps1  (render-check.sh 의 Windows 버전)
#
# 챕터 하나를 렌더링해서 빌드 에러 여부만 확인한다. (/note, /merge 커맨드에서 사용)
# PUBLIC_HOST 가 비어있으면 quarto의 dotenv 검증이 실패하므로 기본값을 채워 넣는다.
#
# 사용법:  .\scripts\render-check.ps1 chapters\05_transformer\00_attention_mechanism.qmd

param(
    [Parameter(Mandatory = $true)]
    [string]$QmdPath
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\_common.ps1"

Import-DotEnv
$python = Get-VenvPython
Initialize-Quarto

Push-Location $ProjectDir
try {
    $env:QUARTO_PYTHON = $python
    if (-not $env:PUBLIC_HOST) { $env:PUBLIC_HOST = 'localhost' }
    if (-not $env:PORT) { $env:PORT = '8080' }

    Invoke-Quarto render $QmdPath
}
finally {
    Pop-Location
}
