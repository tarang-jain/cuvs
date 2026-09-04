#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -Eeo pipefail

WORKTREE=${WORKTREE:-$(git rev-parse --show-toplevel)}
DATASET=${DATASET:-/datasets/tarangj/falcon_1024_10M/base_falcon_1024_10M.fbin}
ARTIFACT_DIR=${ARTIFACT_DIR:?ARTIFACT_DIR must name the durable run directory}
TMUX_SESSION=${TMUX_SESSION:-falcon-pq}
CONDA_ROOT=${CONDA_ROOT:-/raid/tarangj/miniconda3}
PROFILE_ROWS=${PROFILE_ROWS:-1000000}
SMOKE_ROWS=${SMOKE_ROWS:-4096}
BUILD_JOBS=${BUILD_JOBS:-8}
GPU_INDEX=${GPU_INDEX:?GPU_INDEX must be the selected physical GPU}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate cuvs
export CUDA_VISIBLE_DEVICES=${GPU_INDEX}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
set -u

SCRIPT_DIR=${WORKTREE}/cpp/bench/pq/scripts
BUILD_DIR=${WORKTREE}/cpp/build-falcon-pq
BENCH=${BUILD_DIR}/bench/pq/FALCON_PQ_BENCH
MANIFEST=${ARTIFACT_DIR}/manifest.json
MARKER_DIR=${ARTIFACT_DIR}/stages
LOG_DIR=${ARTIFACT_DIR}/logs
RESULT_DIR=${ARTIFACT_DIR}/results
PROFILE_DIR=${ARTIFACT_DIR}/profiles
REPORT_DIR=${ARTIFACT_DIR}/reports
mkdir -p "${MARKER_DIR}" "${LOG_DIR}" "${RESULT_DIR}" "${PROFILE_DIR}" "${REPORT_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  python "${SCRIPT_DIR}/workflow_manifest.py" init \
    --manifest "${MANIFEST}" \
    --worktree "${WORKTREE}" \
    --dataset "${DATASET}" \
    --gpu-index "${GPU_INDEX}" \
    --artifact-dir "${ARTIFACT_DIR}" \
    --tmux-session "${TMUX_SESSION}"
fi

record_and_run() {
  local stage_name=$1
  shift
  local rendered
  printf -v rendered '%q ' "$@"
  python "${SCRIPT_DIR}/workflow_manifest.py" command \
    --manifest "${MANIFEST}" --stage "${stage_name}" --command "${rendered% }"
  "$@"
}

run_stage() {
  local stage_name=$1
  local required=$2
  local function_name=$3
  local marker=${MARKER_DIR}/${stage_name}.done
  if [[ -f "${marker}" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP ${stage_name}: ${marker} exists"
    return 0
  fi
  local timestamp
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  local log=${LOG_DIR}/${timestamp}_${stage_name}.log
  python "${SCRIPT_DIR}/workflow_manifest.py" stage \
    --manifest "${MANIFEST}" --stage "${stage_name}" --status running \
    --log "${log}" --marker "${marker}"
  echo "[$(date -u +%FT%TZ)] START ${stage_name}; log=${log}"
  set +e
  (set -e; "${function_name}") > >(tee -a "${log}") 2>&1
  local rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    touch "${marker}"
    python "${SCRIPT_DIR}/workflow_manifest.py" stage \
      --manifest "${MANIFEST}" --stage "${stage_name}" --status complete \
      --log "${log}" --marker "${marker}"
    echo "[$(date -u +%FT%TZ)] COMPLETE ${stage_name}"
    return 0
  fi
  python "${SCRIPT_DIR}/workflow_manifest.py" stage \
    --manifest "${MANIFEST}" --stage "${stage_name}" --status failed \
    --log "${log}" --marker "${marker}"
  echo "[$(date -u +%FT%TZ)] FAILED ${stage_name} rc=${rc}"
  if [[ ${required} == required ]]; then
    exit "${rc}"
  fi
  return "${rc}"
}

stage_integration() {
  cd "${WORKTREE}"
  record_and_run integration git merge-base --is-ancestor upstream/pr/2548 HEAD
  record_and_run integration git merge-base --is-ancestor upstream/pr/2552 HEAD
  record_and_run integration git merge-base --is-ancestor upstream/pr/2249 HEAD
  record_and_run integration python -c \
    'import os,struct; p=os.environ["DATASET"]; h=open(p,"rb").read(8); assert struct.unpack("<II",h)==(10000000,1024); assert os.path.getsize(p)==40960000008'
  record_and_run integration git status --short --branch
}

stage_build_libcuvs() {
  cd "${WORKTREE}"
  record_and_run build_libcuvs cmake -S cpp -B "${BUILD_DIR}" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120-real \
    -DCUVS_CUTILE_ARCHITECTURES=120 \
    -DBUILD_PQ_PROFILING_BENCH=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_C_TESTS=OFF \
    -DBUILD_C_LIBRARY=OFF \
    -DBUILD_CAGRA_HNSWLIB=OFF \
    -DBUILD_MG_ALGOS=OFF \
    -DCUVS_NVTX=ON || return $?
  record_and_run build_libcuvs cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}" --target \
    cuvs || return $?
  record_and_run build_libcuvs cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}" --target \
    FALCON_PQ_BENCH || return $?
  test -x "${BENCH}"
}

stage_smoke_test() {
  local serial=${RESULT_DIR}/smoke_serial_api
  local parallel=${RESULT_DIR}/smoke_parallel_2
  record_and_run smoke_test "${BENCH}" --dataset="${DATASET}" --mode=serial_api \
    --concurrency=1 --row-cap="${SMOKE_ROWS}" --device=0 --output-prefix="${serial}"
  record_and_run smoke_test python "${SCRIPT_DIR}/validate_result.py" "${serial}.json" \
    --expected-rows "${SMOKE_ROWS}"
  record_and_run smoke_test "${BENCH}" --dataset="${DATASET}" --mode=parallel_prototype \
    --concurrency=2 --row-cap="${SMOKE_ROWS}" --device=0 --output-prefix="${parallel}"
  record_and_run smoke_test python "${SCRIPT_DIR}/validate_result.py" "${parallel}.json" \
    --expected-rows "${SMOKE_ROWS}"
}

stage_baseline() {
  local output=${RESULT_DIR}/serial_api_full
  record_and_run baseline "${BENCH}" --dataset="${DATASET}" --mode=serial_api \
    --concurrency=1 --device=0 --output-prefix="${output}"
  record_and_run baseline python "${SCRIPT_DIR}/validate_result.py" "${output}.json" \
    --expected-rows 10000000 --require-complete
  python "${SCRIPT_DIR}/workflow_manifest.py" artifact --manifest "${MANIFEST}" \
    --name serial_api_result --path "${output}.json"
}

stage_nsys() {
  local prefix=${PROFILE_DIR}/serial_api_profile
  local output=${RESULT_DIR}/serial_api_nsys
  record_and_run nsys nsys profile --trace=cuda,nvtx,osrt,cublas \
    --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output="${prefix}" \
    "${BENCH}" --dataset="${DATASET}" --mode=serial_api --concurrency=1 \
    --row-cap="${PROFILE_ROWS}" --memory-poll-ms=0 --skip-inertia --device=0 \
    --output-prefix="${output}"
  test -s "${prefix}.nsys-rep"
  record_and_run nsys bash -lc \
    "nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum '${prefix}.nsys-rep' > '${REPORT_DIR}/nsys_summary.txt'"
  record_and_run nsys bash -lc \
    "nsys stats --report cuda_gpu_kern_sum --format csv '${prefix}.nsys-rep' > '${REPORT_DIR}/nsys_kernels.csv'"
  test -s "${REPORT_DIR}/nsys_summary.txt"
  test -s "${REPORT_DIR}/nsys_kernels.csv"
  python "${SCRIPT_DIR}/workflow_manifest.py" artifact --manifest "${MANIFEST}" \
    --name nsys_report --path "${prefix}.nsys-rep"
}

stage_ncu() {
  local prefix=${PROFILE_DIR}/serial_api_steady_state
  local output=${RESULT_DIR}/serial_api_ncu
  record_and_run ncu ncu --set full --target-processes all --profile-from-start off \
    --launch-skip 100 --launch-count 10 --force-overwrite --export "${prefix}" \
    "${BENCH}" --dataset="${DATASET}" --mode=serial_api --concurrency=1 \
    --row-cap="${PROFILE_ROWS}" --memory-poll-ms=0 --skip-inertia --device=0 \
    --output-prefix="${output}"
  test -s "${prefix}.ncu-rep"
  record_and_run ncu bash -lc \
    "ncu --import '${prefix}.ncu-rep' --page details --csv > '${REPORT_DIR}/ncu_details.csv'"
  record_and_run ncu bash -lc \
    "ncu --import '${prefix}.ncu-rep' --page raw > '${REPORT_DIR}/ncu_raw.txt'"
  test -s "${REPORT_DIR}/ncu_details.csv"
  test -s "${REPORT_DIR}/ncu_raw.txt"
  python "${SCRIPT_DIR}/workflow_manifest.py" artifact --manifest "${MANIFEST}" \
    --name ncu_report --path "${prefix}.ncu-rep"
}

stage_prototype_sweep() {
  local concurrency
  for concurrency in 1 2 4 8 16; do
    local output=${RESULT_DIR}/parallel_prototype_${concurrency}_full
    record_and_run prototype_sweep "${BENCH}" --dataset="${DATASET}" \
      --mode=parallel_prototype --concurrency="${concurrency}" --device=0 \
      --output-prefix="${output}"
    record_and_run prototype_sweep python "${SCRIPT_DIR}/validate_result.py" "${output}.json" \
      --expected-rows 10000000 --require-complete
  done
}

stage_final_comparison() {
  record_and_run final_comparison python "${SCRIPT_DIR}/summarize_results.py" \
    --results-dir "${RESULT_DIR}" --output-prefix "${REPORT_DIR}/comparison"
  cp "${WORKTREE}/cpp/bench/pq/README.md" "${REPORT_DIR}/faiss_comparison_and_method.md"
  test -s "${REPORT_DIR}/comparison.csv"
  test -s "${REPORT_DIR}/comparison.md"
  python "${SCRIPT_DIR}/workflow_manifest.py" artifact --manifest "${MANIFEST}" \
    --name comparison_csv --path "${REPORT_DIR}/comparison.csv"
  python "${SCRIPT_DIR}/workflow_manifest.py" artifact --manifest "${MANIFEST}" \
    --name comparison_markdown --path "${REPORT_DIR}/comparison.md"
}

export DATASET
run_stage integration required stage_integration
run_stage build_libcuvs required stage_build_libcuvs
run_stage smoke_test required stage_smoke_test
run_stage baseline required stage_baseline
profile_failures=0
run_stage nsys optional stage_nsys || profile_failures=1
run_stage ncu optional stage_ncu || profile_failures=1
run_stage prototype_sweep required stage_prototype_sweep
run_stage final_comparison required stage_final_comparison
if [[ ${profile_failures} -ne 0 ]]; then
  touch "${ARTIFACT_DIR}/workflow.partial"
  echo "Workflow benchmarks completed, but one or more profiling stages remain incomplete."
  exit 2
fi
touch "${ARTIFACT_DIR}/workflow.complete"
echo "Workflow complete: ${ARTIFACT_DIR}"
