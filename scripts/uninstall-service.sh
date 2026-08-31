#!/usr/bin/env bash
#
# uninstall-service.sh
# ai-ml-study systemd 서비스를 중지하고 등록 해제한다.
#
# 사용법:  ./scripts/uninstall-service.sh
#
set -euo pipefail

SERVICE_NAME="ai-ml-study"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ ! -f "${UNIT_PATH}" ]]; then
  echo "등록된 서비스가 없습니다: ${UNIT_PATH}"
  exit 0
fi

echo "==> 서비스 중지 및 비활성화"
sudo systemctl stop "${SERVICE_NAME}.service"    || true
sudo systemctl disable "${SERVICE_NAME}.service" || true

echo "==> 유닛 파일 제거: ${UNIT_PATH}"
sudo rm -f "${UNIT_PATH}"
sudo systemctl daemon-reload
sudo systemctl reset-failed "${SERVICE_NAME}.service" 2>/dev/null || true

echo "완료. 서비스가 제거되었습니다."
