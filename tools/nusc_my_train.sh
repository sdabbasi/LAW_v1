#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="$1"  # e.g. law/default
CONFIG="projects/configs/${EXP_NAME}.py"
GPUS="$2"
PORT="${PORT:-59230}"

# Create a lexicographically sortable timestamped run directory.
# This makes older experiments appear first (top) and newer ones last (bottom)
# when listing by name.
BASE_WORK_DIR="work_dirs/${EXP_NAME}"
RUN_ID="$(date +"%Y%m%d_%H%M%S")"
WORK_DIR="${BASE_WORK_DIR}/${RUN_ID}"
mkdir -p "${WORK_DIR}"

# Make the run id available to python processes if needed.
export LAW_RUN_ID="${RUN_ID}"

PYTHONPATH="$(dirname "$0")/..${PYTHONPATH:+:$PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node="${GPUS}" --master_port="${PORT}" \
    "$(dirname "$0")/train.py" "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch "${@:3}" \
    --deterministic \
    --cfg-options evaluation.jsonfile_prefix="${WORK_DIR}/eval/results"
