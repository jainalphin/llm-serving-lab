#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

cd "${PROJECT_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "${VENV_DIR}" --python 3.12
  else
    python3 -m venv "${VENV_DIR}"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${VENV_DIR}/bin/python" -r requirements.txt
else
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -r requirements.txt
fi

echo
echo "Environment is ready."
echo "Activate it with: source .venv/bin/activate"
echo "Run the UI with: PYTHONPATH=. python -m streamlit run app.py"
