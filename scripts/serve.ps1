# serve.ps1
#
# 빌드된 _site/ 를 로컬 정적 서버로 서빙한다. (자동 새로고침 없음 — 빌드 결과 확인용)
# Windows에는 systemd 가 없으므로, 상시 구동 서비스 대신 이 스크립트로 그때그때 띄운다.
#
# 127.0.0.1 에만 바인딩하므로 이 PC 밖에서는 접속되지 않는다(방화벽 예외 불필요).
#
# 사용법:  .\scripts\serve.ps1 [-Port 8080]
#          (Port 미지정 시 .env 의 PORT, 그것도 없으면 8080)

param(
    [int]$Port
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\_common.ps1"

Import-DotEnv
$python = Get-VenvPython

# 포트 우선순위: -Port 인자 > .env PORT > 8080
if (-not $Port) {
    if ($env:PORT) { $Port = [int]$env:PORT } else { $Port = 8080 }
}

$siteDir = Join-Path $ProjectDir '_site'
if (-not (Test-Path $siteDir)) {
    throw "_site 디렉터리가 없습니다. 먼저 빌드하세요:  .\scripts\build.ps1"
}

Write-Host "==> http://localhost:$Port  (종료: Ctrl+C)"
& $python -m http.server $Port --bind 127.0.0.1 --directory $siteDir
