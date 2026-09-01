# preview.ps1  (preview.sh 의 Windows 버전)
#
# quarto preview 로 로컬 전용 실시간 미리보기 서버를 띄운다.
# .qmd 를 저장하면 자동으로 다시 렌더링되고 브라우저가 새로고침된다.
#
# 사용법:  .\scripts\preview.ps1 [-Port 4200]
#          (Port 미지정 시 .env 의 PREVIEW_PORT, 그것도 없으면 quarto 가 자동 선택)

param(
    [int]$Port
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\_common.ps1"

Import-DotEnv
$python = Get-VenvPython
Initialize-Quarto

# 포트 우선순위: -Port 인자 > .env PREVIEW_PORT > quarto 기본(자동 선택)
if (-not $Port -and $env:PREVIEW_PORT) { $Port = [int]$env:PREVIEW_PORT }

Push-Location $ProjectDir
try {
    $env:QUARTO_PYTHON = $python

    $quartoArgs = @('preview')
    if ($Port) { $quartoArgs += @('--port', "$Port") }

    Write-Host "==> quarto preview 시작 (종료: Ctrl+C)"
    # Ctrl+C 로 끝내는 장기 실행 서버라 종료 코드는 확인하지 않는다.
    $ErrorActionPreference = 'Continue'
    quarto @quartoArgs
}
finally {
    Pop-Location
}
