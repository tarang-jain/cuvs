# Durable Falcon PQ profiling prototype

This opt-in benchmark compares the public serial product-quantizer build path with a benchmark-only prototype that trains independent PQ subspaces concurrently. It does not change the public PQ API or production behavior.

## Fixed experiment

The benchmark accepts only the Falcon `10,000,000 x 1024` float32 `.fbin` layout and validates its header, exact byte count, loaded values, and output centroids. The experiment uses 512 two-dimensional codebooks, 8-bit codes, 256 centroids, regular KMeans, scalable KMeans|| initialization with oversampling factor 2, seed 42, one initialization, at most 300 iterations, and tolerance `1e-4`. `max_train_points_per_pq_code=39,063` gives a cap of 10,000,128 rows, so the full run uses all 10 million rows.

`serial_api` calls `cuvs::preprocessing::quantize::pq::build`. The API returns centroids but not per-subspace iteration counts, so its convergence status is explicitly reported as `not_exposed_by_pq_build`; per-subspace and total inertia are evaluated from the returned centroids after the timed region.

`parallel_prototype` preserves the same KMeans parameters but assigns independent subspaces to separate CUDA streams and RAFT resources in waves bounded by `--concurrency`. It reports each fit's iteration count, a conservative convergence flag (`iterations < max_iter`), inertia, wall time, and sampled GPU/host memory. Comparisons use total and per-subspace inertia rather than exact centroid identity.

## Build and direct use

```bash
cmake -S cpp -B cpp/build-falcon-pq -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120-real \
  -DCUVS_CUTILE_ARCHITECTURES=120 \
  -DBUILD_PQ_PROFILING_BENCH=ON \
  -DBUILD_TESTS=OFF
cmake --build cpp/build-falcon-pq --target cuvs
cmake --build cpp/build-falcon-pq --target FALCON_PQ_BENCH

cpp/build-falcon-pq/bench/pq/FALCON_PQ_BENCH \
  --dataset=/datasets/tarangj/falcon_1024_10M/base_falcon_1024_10M.fbin \
  --mode=parallel_prototype --concurrency=8 \
  --output-prefix=/tmp/falcon-pq-8
```

Use `--row-cap=4096` for a smoke test. A row-capped result deliberately reports `complete_row_usage=false`.

## Durable workflow

Launch from the integration worktree:

```bash
cpp/bench/pq/scripts/launch_falcon_pq_workflow.sh
```

The launcher chooses GPU 0 if it meets the idle threshold; otherwise it takes the first idle GPU and records its physical index and UUID. It creates a named detached tmux session and prints attach, status, and log-tail commands. Artifacts live under `artifacts/falcon-pq/<UTC run id>/`.

Stages are integration, libcuvs build, smoke test, baseline, Nsight Systems, Nsight Compute, prototype sweep, and final comparison. Each command and stage status is written atomically to `manifest.json`. A `.done` marker is created only after the command succeeds and its expected output passes validation. Rerunning `run_falcon_pq_workflow.sh` with the same `ARTIFACT_DIR`, `TMUX_SESSION`, and `GPU_INDEX` skips completed stages. Profiling-stage failure is recorded but does not prevent the full benchmark sweep; the overall run remains `workflow.partial` until those stages succeed.

The build stage compiles the shared `cuvs` target and the standalone benchmark; it does not compile test targets. `CUVS_CUTILE_ARCHITECTURES=120` limits embedded cuTile cubins to the GPU architecture used by this experiment. Profiles retain `.nsys-rep` and `.ncu-rep` files plus CSV/text summaries. Nsight capture begins after dataset loading at `cudaProfilerStart`, keeping the reports focused on training. The profile uses a configurable representative row cap (`PROFILE_ROWS`, default one million), while baseline and sweep runs always use all rows.

Tmux protects this work from SSH or client disconnection. It does not protect against host reboot, administrator termination, GPU reset, or machine failure.

## Faiss comparison and kernel model

Faiss's PQ training loop copies and trains one subspace at a time. Its clustering defaults cap training at 256 samples per centroid, or 65,536 rows for 256 centroids, unless that cap is raised. Faiss offers random initialization, exact KMeans++, and AFK-MC², but not scalable KMeans||.

For the two-dimensional Lloyd assignment that matters here, Faiss's CPU implementation uses a specialized AVX-512 fused L2-plus-top-1 organization. It processes multiple samples and centroid lanes together and avoids materializing the full distance matrix. That organization is the model for a possible second GPU prototype: if Nsight Compute shows the current fused 1-NN path is occupancy- or scheduler-limited, batch the launch grid over `[codebook, sample tile]` while retaining codebook-local reductions and leaving the public PQ API unchanged.

The first prototype deliberately tests the lower-risk stream/resource approach. The retained Nsight reports determine whether a codebook-batched fused kernel is justified; lack of speedup is still a valid, reported outcome.
