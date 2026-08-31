#!/usr/bin/env bash
#
# render-check.sh
# 챕터 하나를 렌더링해서 빌드 에러 여부만 확인한다. (/note, /merge 커맨드에서 사용)
# PUBLIC_HOST 가 비어있으면 quarto의 dotenv 검증이 실패하므로 기본값을 채워 넣는다.
#
# 사용법:  ./scripts/render-check.sh <qmd 파일 경로>
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_DIR}"

if [[ -x "${PROJECT_DIR}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_DIR}/venv/bin/python"
elif [[ -x "${PROJECT_DIR}/venv/Scripts/python.exe" ]]; then
  PYTHON="${PROJECT_DIR}/venv/Scripts/python.exe"
else
  echo "!! venv 파이썬이 없습니다: ${PROJECT_DIR}/venv" >&2
  exit 1
fi

export PUBLIC_HOST="${PUBLIC_HOST:-localhost}"
export PORT="${PORT:-8080}"

QUARTO_PYTHON="${PYTHON}" quarto render "$1"
