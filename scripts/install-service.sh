#!/usr/bin/env bash
#
# install-service.sh
# ai-ml-study Quarto 책 HTTP 서버를 systemd 시스템 서비스로 등록한다.
# 터미널 종료·로그아웃·재부팅 후에도 서비스가 유지된다.
#
# 사용법:  ./scripts/install-service.sh [PORT]
#          (PORT 미지정 시 .env 의 PORT, 그것도 없으면 8080)
#
# 접속 호스트 안내는 다음 우선순위로 결정된다:
#   1) .env 의 PUBLIC_HOST (고정 도메인/IP)
#   2) GCP 메타데이터의 외부 IP (자동 조회, VM 재시작 시 변경될 수 있음)
#   3) 내부 IP (hostname -I)
#
set -euo pipefail

# 프로젝트 루트 (이 스크립트 위치 기준)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# .env 로드 (있으면). PUBLIC_HOST, PORT 등을 정의할 수 있다.
PUBLIC_HOST=""
if [[ -f "${PROJECT_DIR}/.env" ]]; then
  set -a; source "${PROJECT_DIR}/.env"; set +a
fi

# 포트 우선순위: CLI 인자 > .env PORT > 8080
PORT="${1:-${PORT:-8080}}"
SERVICE_NAME="ai-ml-study"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON="${PROJECT_DIR}/venv/bin/python"
SITE_DIR="${PROJECT_DIR}/_site"
RUN_USER="$(whoami)"

# 안내에 사용할 접속 호스트 결정.
# 우선순위: .env PUBLIC_HOST > GCP 메타데이터 외부 IP > 내부 IP(hostname -I)
resolve_host() {
  if [[ -n "${PUBLIC_HOST}" ]]; then
    echo "${PUBLIC_HOST}"; return
  fi
  local ext
  ext="$(curl -s --max-time 3 -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip' 2>/dev/null || true)"
  if [[ -n "${ext}" ]]; then
    echo "${ext}"; return
  fi
  hostname -I | awk '{print $1}'
}

echo "==> 프로젝트 : ${PROJECT_DIR}"
echo "==> 포트     : ${PORT}"
echo "==> 실행 유저: ${RUN_USER}"

# 사전 점검
if [[ ! -x "${PYTHON}" ]]; then
  echo "!! venv 파이썬을 찾을 수 없습니다: ${PYTHON}" >&2
  echo "   먼저 venv 를 생성하세요: python -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
if [[ ! -d "${SITE_DIR}" ]]; then
  echo "!! _site 디렉터리가 없습니다. 먼저 빌드하세요:" >&2
  echo "   QUARTO_PYTHON=${PYTHON} quarto render" >&2
  exit 1
fi

# systemd 유닛 생성
echo "==> systemd 유닛 작성: ${UNIT_PATH}"
sudo tee "${UNIT_PATH}" >/dev/null <<EOF
[Unit]
Description=AI-ML Study Quarto Book HTTP Server
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PYTHON} -m http.server ${PORT} --bind 0.0.0.0 --directory ${SITE_DIR}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
sudo systemctl restart "${SERVICE_NAME}.service"

sleep 2
echo
echo "==> 상태"
sudo systemctl status "${SERVICE_NAME}.service" --no-pager | head -8 || true
echo
HOST="$(resolve_host)"
echo "완료. 서비스가 상시 구동됩니다."
echo "  로컬 : http://localhost:${PORT}"
echo "  외부 : http://${HOST}:${PORT}"
if [[ -z "${PUBLIC_HOST}" ]]; then
  echo "  (외부 IP는 GCP 메타데이터에서 자동 조회됨 — VM 재시작 시 변경될 수 있습니다."
  echo "   고정하려면 .env 의 PUBLIC_HOST 에 도메인/고정 IP를 지정하세요.)"
fi
echo "  (외부 접속이 안 되면 방화벽에서 TCP ${PORT} 인바운드를 여세요)"
