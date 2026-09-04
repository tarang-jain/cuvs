#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

WORKTREE=${WORKTREE:-$(git rev-parse --show-toplevel)}
DATASET=${DATASET:-/datasets/tarangj/falcon_1024_10M/base_falcon_1024_10M.fbin}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
TMUX_SESSION=${TMUX_SESSION:-falcon-pq-${RUN_ID}}
ARTIFACT_DIR=${ARTIFACT_DIR:-${WORKTREE}/artifacts/falcon-pq/${RUN_ID}}

if [[ -z ${GPU_INDEX:-} ]]; then
  GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (!found && ($2 + 0) < 1024 && ($3 + 0) <= 5) {print $1; found=1}}')
fi
if [[ -z ${GPU_INDEX} ]]; then
  echo "No GPU with less than 1 GiB used memory and at most 5% utilization was found." >&2
  exit 1
fi
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TMUX_SESSION}" >&2
  exit 1
fi
mkdir -p "${ARTIFACT_DIR}"
printf '%s\n' "${TMUX_SESSION}" > "${ARTIFACT_DIR}/tmux_session.txt"
printf '%s\n' "${GPU_INDEX}" > "${ARTIFACT_DIR}/physical_gpu_index.txt"
printf '%s\n' "${ARTIFACT_DIR}" > "${WORKTREE}/artifacts/falcon-pq/latest_run.txt"

printf -v command \
  'cd %q && WORKTREE=%q DATASET=%q ARTIFACT_DIR=%q TMUX_SESSION=%q GPU_INDEX=%q bash %q 2>&1 | tee -a %q' \
  "${WORKTREE}" "${WORKTREE}" "${DATASET}" "${ARTIFACT_DIR}" "${TMUX_SESSION}" "${GPU_INDEX}" \
  "${WORKTREE}/cpp/bench/pq/scripts/run_falcon_pq_workflow.sh" "${ARTIFACT_DIR}/workflow.log"
tmux new-session -d -s "${TMUX_SESSION}" "${command}"

echo "session=${TMUX_SESSION}"
echo "gpu=${GPU_INDEX}"
echo "artifacts=${ARTIFACT_DIR}"
echo "attach: tmux attach -t ${TMUX_SESSION}"
echo "status: tmux has-session -t ${TMUX_SESSION} && tail -n 40 ${ARTIFACT_DIR}/workflow.log"
echo "logs: tail -F ${ARTIFACT_DIR}/workflow.log"
