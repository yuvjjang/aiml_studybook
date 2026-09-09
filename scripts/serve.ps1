# serve.ps1
#
# 빌드된 _site/ 를 로컬 정적 서버로 서빙한다. (자동 새로고침 없음 — 빌드 결과 확인용)
# Windows에는 systemd 가 없으므로, 상시 구동 서비스 대신 이 스크립트로 그때그때 띄운다.
#
# 기본값은 127.0.0.1 바인딩이라 이 PC 밖에서는 접속되지 않는다(방화벽 예외 불필요).
# -Lan 을 주면 0.0.0.0 에 바인딩해 같은 공유기에 붙은 스마트폰에서도 볼 수 있다.
# 이때는 Windows 방화벽 인바운드 규칙이 필요하며, 스크립트가 상태를 확인해 알려준다.
#
# 사용법:  .\scripts\serve.ps1 [-Port 8080] [-Lan]
#          (Port 미지정 시 .env 의 PORT, 그것도 없으면 8080)

param(
    [int]$Port,
    [switch]$Lan
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

if (-not $Lan) {
    Write-Host "==> http://localhost:$Port  (종료: Ctrl+C)"
    & $python -m http.server $Port --bind 127.0.0.1 --directory $siteDir
    return
}

# ── LAN 공개 ────────────────────────────────────────────────
# 기본 게이트웨이를 가진 인터페이스의 IPv4 주소가 스마트폰이 찾아올 주소다.
$cfg = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | Select-Object -First 1
if (-not $cfg) { throw "기본 게이트웨이를 가진 네트워크 인터페이스를 찾지 못했습니다." }
$lanIp = $cfg.IPv4Address.IPAddress

Write-Host "==> http://${lanIp}:$Port  (같은 Wi-Fi 의 스마트폰에서 접속, 종료: Ctrl+C)"
Write-Host "    http://localhost:$Port  (이 PC)"

# 방화벽이 막고 있으면 스마트폰에서 조용히 타임아웃만 난다. 미리 짚어 준다.
$profileName = (Get-NetConnectionProfile -InterfaceIndex $cfg.InterfaceIndex).NetworkCategory
$ruleName = "ai-ml-study serve ($Port)"
$hasRule = [bool](Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)

if (-not $hasRule) {
    Write-Host ""
    Write-Host "    [알림] '$($cfg.InterfaceAlias)' 네트워크 프로필이 $profileName 이고 인바운드 예외가 없습니다."
    Write-Host "           스마트폰에서 접속되지 않으면 관리자 PowerShell 에서 아래를 한 번 실행하세요:"
    Write-Host ""
    Write-Host "           New-NetFirewallRule -DisplayName '$ruleName' ``"
    Write-Host "               -Direction Inbound -Action Allow -Protocol TCP ``"
    Write-Host "               -LocalPort $Port -Profile $profileName"
    Write-Host ""
    Write-Host "           되돌릴 때:  Remove-NetFirewallRule -DisplayName '$ruleName'"
    Write-Host ""
}

& $python -m http.server $Port --bind 0.0.0.0 --directory $siteDir
