#!/usr/bin/env bash
# Memory Factory Studio - start on macOS / Linux
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
  pip install -q -r requirements.txt   # picks up new packages after a git pull
fi
python app.py "$@"
