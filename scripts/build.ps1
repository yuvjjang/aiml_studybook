# build.ps1  (build.sh 의 Windows 버전)
#
# Quarto 책 전체를 _site/ 로 렌더링한다.
#
# 사용법:  .\scripts\build.ps1

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\_common.ps1"

Import-DotEnv
$python = Get-VenvPython
Initialize-Quarto

Push-Location $ProjectDir
try {
    $env:QUARTO_PYTHON = $python

    Write-Host "==> Quarto 렌더링 시작 (시간이 다소 걸립니다)"
    Invoke-Quarto render

    Write-Host "==> 완료: $ProjectDir\_site"
}
finally {
    Pop-Location
}
