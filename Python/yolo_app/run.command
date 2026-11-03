#!/bin/bash
# Double-click (macOS) or run ./yolo_simple/run.command — uses the project's .venv (created on first run).
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "Setting up environment (first run only)…"
  python3 -m venv --system-site-packages .venv && .venv/bin/pip install -q -r requirements.txt || exit 1
fi
exec .venv/bin/python -m yolo_app
