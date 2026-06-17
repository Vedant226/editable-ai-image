#!/usr/bin/env bash
# Start the inpaint service on 127.0.0.1:8000 using the project venv.
set -euo pipefail
cd "$(dirname "$0")"
exec ../venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 "$@"
