#!/usr/bin/env bash
set -euo pipefail
# Run in a separate WAM environment: SAM 3 and WSP pin different ftfy versions.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${XPL_ROOT}" -e "${SCRIPT_DIR}/worldscape-policy[server]" \
    -r "${SCRIPT_DIR}/requirements-wam.txt" -r "${SCRIPT_DIR}/requirements-runtime.txt"
python "${SCRIPT_DIR}/preflight.py" --check-runtime
echo "Bind the Flash/Memory checkpoints and start the WAM service; see WAM.md."
