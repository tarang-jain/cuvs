#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Profile k-means++ init only (max_iter=0) on the 5M x 1536 blobs dataset.
# Requires: cuvs built/installed in the active env; nsys + ncu on PATH.
set -euo pipefail

DATASET="${DATASET:-/datasets/blobs-5M-1536-k1024/base.fbin}"
OUT_DIR="${OUT_DIR:-/tmp/kmeanspp_flash_profile}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/profile_kmeanspp_init.py"
NCU_BIN="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
NSYS_BIN="${NSYS_BIN:-nsys}"

mkdir -p "${OUT_DIR}"

echo "== Dataset: ${DATASET}"
echo "== Output:  ${OUT_DIR}"

# 1) Timeline: wall time + CUDA + NVTX (see kmeans_initialization / initScalableKMeansPlusPlus)
echo "== nsys profile"
"${NSYS_BIN}" profile \
  -t cuda,nvtx,osrt \
  -o "${OUT_DIR}/kmeanspp_init" \
  --force-overwrite true \
  python "${PY_SCRIPT}" \
    --dataset "${DATASET}" \
    --n-clusters 1024 \
    --oversampling-factor 2.0 \
    --max-iter 0 \
    --warmup 0

# 2) Kernel metrics for fused 1NN only (flash path). Full NVTX-range + kernel
#    replay often LaunchTimeout / fails on cuLaunchKernelEx; app replay + single
#    kernel + few sections is the reliable path.
echo "== ncu profile (fused_1nn_f_i32_relaxed only)"
"${NCU_BIN}" \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --replay-mode application \
  --clock-control none \
  -k regex:fused_1nn_f_i32_relaxed \
  -c 1 \
  -o "${OUT_DIR}/fused_1nn_ncu" \
  --force-overwrite \
  python "${PY_SCRIPT}" \
    --dataset "${DATASET}" \
    --n-clusters 1024 \
    --oversampling-factor 2.0 \
    --max-iter 0 \
    --warmup 0

echo "Done."
echo "  nsys: ${OUT_DIR}/kmeanspp_init.nsys-rep"
echo "  ncu:  ${OUT_DIR}/fused_1nn_ncu.ncu-rep"
echo "Open with Nsight Systems / Nsight Compute GUIs, or:"
echo "  nsys stats ${OUT_DIR}/kmeanspp_init.nsys-rep"
echo "  ${NCU_BIN} --import ${OUT_DIR}/fused_1nn_ncu.ncu-rep --page details"
