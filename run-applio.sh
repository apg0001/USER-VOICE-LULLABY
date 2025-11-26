#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVC_DIR="${ROOT_DIR}/rvc"
VENV_PY="${RVC_DIR}/env/bin/python"

if [[ -x "${VENV_PY}" ]]; then
  PY_CMD="${VENV_PY}"
else
  PY_CMD="python3"
fi

cd "${ROOT_DIR}"
exec "${PY_CMD}" -m app "$@"

