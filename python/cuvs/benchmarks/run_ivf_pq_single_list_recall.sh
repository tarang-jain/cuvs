#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Profile SIFT-1M IVF-PQ with n_lists=1. PQ codebooks come from the
# preprocessing PQ API (classic k-means); Step-8 init is selected with
# CUVS_KMEANS_PP_STEP8.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${BIN:-${ROOT}/examples/cpp/build/IVF_PQ_SINGLE_LIST_RECALL_EXAMPLE}"
OUT_DIR="${OUT_DIR:-/tmp/ivf_pq_single_list_recall}"
NSYS_BIN="${NSYS_BIN:-nsys}"
BASE="${BASE:-/datasets/sift-128-euclidean/base.fbin}"
QUERIES="${QUERIES:-/datasets/sift-128-euclidean/query.fbin}"
GT="${GT:-/datasets/sift-128-euclidean/groundtruth.neighbors.ibin}"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing ${BIN}. Build with: cmake --build examples/cpp/build --target IVF_PQ_SINGLE_LIST_RECALL_EXAMPLE" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
echo "binary=${BIN}"
echo "out=${OUT_DIR}"

run_one() {
  local mode="$1"
  local tag="$2"
  echo
  echo "== Step-8 ${mode}"
  CUVS_KMEANS_PP_STEP8="${mode}" "${NSYS_BIN}" profile \
    -t cuda,nvtx,osrt \
    -o "${OUT_DIR}/ivf_pq_${tag}" \
    --force-overwrite true \
    "${BIN}" \
      --base "${BASE}" \
      --queries "${QUERIES}" \
      --groundtruth "${GT}" \
    | tee "${OUT_DIR}/ivf_pq_${tag}.log"
}

run_one random random
run_one kmeans++ kmeanspp

echo
echo "Done."
echo "  nsys: ${OUT_DIR}/ivf_pq_random.nsys-rep  ${OUT_DIR}/ivf_pq_kmeanspp.nsys-rep"
echo "  logs: ${OUT_DIR}/ivf_pq_random.log       ${OUT_DIR}/ivf_pq_kmeanspp.log"
