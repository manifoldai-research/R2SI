#!/bin/bash
set -euo pipefail
# XPolicyLab itself, then what the policy imports. torch and torchvision must already be in the environment, built for its CUDA.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${XPL_ROOT}" \
    -r "${SCRIPT_DIR}/requirements.txt" \
    -r "${SCRIPT_DIR}/requirements-runtime.txt" \
    -r "${SCRIPT_DIR}/requirements-test.txt"
echo "For WAM inference, run install_wam.sh in a separate environment and configure its service; see WAM.md."
