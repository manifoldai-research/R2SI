#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]'
    echo 'Reserved WSP training interface (stub); training backend is not connected.'
    exit 0
fi
if [[ "${1:-}" != "RoboDojo" ]]; then
    echo '[WorldScape_R2SI] only bench_name=RoboDojo is supported' >&2
    exit 2
fi
echo 'WSP training backend is not connected (reserved interface / stub). No training job or checkpoint was created.' >&2
exit 2
