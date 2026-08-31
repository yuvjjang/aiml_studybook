#!/usr/bin/env bash
#
# service.sh
# 등록된 ai-ml-study 서비스를 제어하는 래퍼.
#
# 사용법:  ./scripts/service.sh {start|stop|restart|status|logs}
#
set -euo pipefail

SERVICE_NAME="ai-ml-study"
CMD="${1:-status}"

case "${CMD}" in
  start)   sudo systemctl start   "${SERVICE_NAME}.service"; echo "시작됨" ;;
  stop)    sudo systemctl stop    "${SERVICE_NAME}.service"; echo "중지됨" ;;
  restart) sudo systemctl restart "${SERVICE_NAME}.service"; echo "재시작됨" ;;
  status)  sudo systemctl status  "${SERVICE_NAME}.service" --no-pager ;;
  logs)    sudo journalctl -u "${SERVICE_NAME}.service" -f --no-pager ;;
  *)
    echo "사용법: $0 {start|stop|restart|status|logs}" >&2
    exit 1
    ;;
esac
