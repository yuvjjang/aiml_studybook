#!/usr/bin/env bash
#
# build.sh
# Quarto 책을 _site/ 로 렌더링한다. 서비스가 이미 _site/ 를 서빙 중이면
# 렌더 완료 후 자동으로 최신 내용이 반영된다(재시작 불필요).
#
# 사용법:  ./scripts/build.sh
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

echo "==> Quarto 렌더링 시작 (시간이 다소 걸립니다)"
QUARTO_PYTHON="${PYTHON}" quarto render

echo "==> 완료: ${PROJECT_DIR}/_site"
