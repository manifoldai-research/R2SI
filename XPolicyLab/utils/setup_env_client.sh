#!/bin/bash
set -e

UTILS_DIR="${1}"
yaml_file="${2}"
eval_env_conda_env="${3}"
policy_server_port="${4}"
bench_name="${5}"
task_name="${6}"
env_cfg_type="${7}"
policy_name="${8}"
additional_info="${9}"
ROOT_DIR="${10}"
seed="${11}"
env_gpu_id="${12}"
policy_server_ip="${13:-localhost}"
protocol_override="${14:-}"

# shellcheck source=resolve_eval_env_type.sh
source "${UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)" || exit 1

# Read deploy.yml with the eval env's interpreter. This runs before any of
# the run_*_env_client.sh scripts activate that env, so otherwise the read
# lands on whatever python the launching shell exposes. The eval env needs
# pyyaml regardless: the client imports XPolicyLab.utils.process_data, which
# pulls in load_file -> yaml. Activation is best-effort and confined to the
# subshell; without conda the read falls back to the ambient interpreter.
deploy_meta="$(
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null)" || conda_base=""
        if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${conda_base}/etc/profile.d/conda.sh"
            conda activate "${eval_env_conda_env}" 2>/dev/null || true
        fi
    fi
    python - <<PY
import yaml
with open("${yaml_file}", "r") as f:
    data = yaml.safe_load(f)
print(
    str(data.get("eval_batch", False)).lower(),
    data.get("protocol", "ws"),
)
PY
)" || true

read eval_batch yaml_protocol <<< "${deploy_meta}" || true

if [[ -z "${eval_batch}" ]]; then
    echo "[ERROR] could not read eval_batch/protocol from ${yaml_file}." >&2
    echo "[ERROR] Check the file exists and that pyyaml is installed in ${eval_env_conda_env}." >&2
    exit 1
fi

protocol="${protocol_override:-${yaml_protocol}}"

if [[ -z "${EVAL_ENV_TYPE:-}" ]]; then
    echo "[CLIENT] EVAL_ENV_TYPE=(default sim) -> ${eval_env_mode}"
else
    echo "[CLIENT] EVAL_ENV_TYPE=${EVAL_ENV_TYPE} -> ${eval_env_mode}"
fi

COMMON_ARGS=(
    "${eval_batch}"
    "${eval_env_conda_env}"
    "${policy_server_port}"
    "${bench_name}"
    "${task_name}"
    "${env_cfg_type}"
    "${policy_name}"
    "${additional_info}"
    "${ROOT_DIR}"
    "${seed}"
    "${env_gpu_id}"
    "${policy_server_ip}"
)

if [[ "${eval_env_mode}" == "debug" ]]; then
    bash "${UTILS_DIR}/run_debug_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
elif [[ "${eval_env_mode}" == "sim" ]]; then
    bash "${UTILS_DIR}/run_sim_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
elif [[ "${eval_env_mode}" == "real_world" ]]; then
    echo -e "\033[31m[WARN] EVAL_ENV_TYPE=real: real-world evaluation is not supported in the open-source release; continuing to real env client.\033[0m" >&2
    bash "${UTILS_DIR}/run_real_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
else
    echo "[ERROR] Unknown eval env mode: ${eval_env_mode}" >&2
    exit 1
fi
