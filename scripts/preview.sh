#!/usr/bin/env bash
#
# preview.sh
# quarto preview 로 로컬 전용 실시간 미리보기 서버를 띄운다.
# .qmd 저장 시 자동으로 다시 렌더링되고 브라우저가 자동 새로고침된다.
#
# 사용법:  ./scripts/preview.sh [PORT]
#          (PORT 미지정 시 .env 의 PREVIEW_PORT, 그것도 없으면 quarto 기본 포트)
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PREVIEW_PORT=""
if [[ -f "${PROJECT_DIR}/.env" ]]; then
  set -a; source "${PROJECT_DIR}/.env"; set +a
fi

if [[ -x "${PROJECT_DIR}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_DIR}/venv/bin/python"
elif [[ -x "${PROJECT_DIR}/venv/Scripts/python.exe" ]]; then
  PYTHON="${PROJECT_DIR}/venv/Scripts/python.exe"
else
  echo "!! venv 파이썬을 찾을 수 없습니다. 먼저 venv를 생성하세요:" >&2
  echo "   Windows      : python -m venv venv && venv/Scripts/pip install -r requirements.txt" >&2
  echo "   macOS/Linux  : python -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# 포트 우선순위: CLI 인자 > .env PREVIEW_PORT > quarto 기본(자동 선택)
PORT_ARG="${1:-${PREVIEW_PORT}}"

ARGS=(preview)
if [[ -n "${PORT_ARG}" ]]; then
  ARGS+=(--port "${PORT_ARG}")
fi

echo "==> quarto preview 시작 (종료: Ctrl+C)"
QUARTO_PYTHON="${PYTHON}" quarto "${ARGS[@]}"
