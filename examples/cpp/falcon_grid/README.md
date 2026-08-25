# Falcon 10M Device PQ Grid

This benchmark reproduces the Falcon 10M product-quantization grid used to compare
classic k-means++, scalable k-means|| with random Step 8, scalable k-means|| with
k-means++ Step 8, and overall random initialization.

The harness keeps the Falcon base vectors device-resident while each cell trains
PQ codebooks, builds a single-list IVF-PQ index, and reports recall@100 against
the Falcon ground truth.

## Inputs

The default paths are:

```bash
/datasets/falcon_1024_10M/base_falcon_1024_10M.fbin
/datasets/falcon_1024_10M/query_falcon_1024_10K.fbin
/datasets/falcon_1024_10M/falcon_1024_10M_gt100_10K.fbin
```

`base` and `query` are `.fbin` files with int32 row/column headers followed by
fp32 row-major data. The ground-truth file is read as the same binary container
with `10000 x 100` neighbor ids.

## Build

From the repository root:

```bash
cmake -S examples/cpp/falcon_grid -B examples/cpp/falcon_grid/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build examples/cpp/falcon_grid/build -j
```

The target is:

```bash
examples/cpp/falcon_grid/build/FALCON_PQ_DEVICE_GRID
```

## Smoke Test

Run one small cell for each `pq_len`:

```bash
python3 python/cuvs/benchmarks/run_falcon_pq_device_grid.py \
  --binary examples/cpp/falcon_grid/build/FALCON_PQ_DEVICE_GRID \
  --output-dir /tmp/falcon_pq_clean_results_smoke \
  --smoke
```

## Full Grid

Run the full 272-cell grid:

```bash
python3 python/cuvs/benchmarks/run_falcon_pq_device_grid.py \
  --binary examples/cpp/falcon_grid/build/FALCON_PQ_DEVICE_GRID \
  --output-dir /tmp/falcon_pq_clean_results
```

Resume after interruption by running the same command again. Completed `cell_id`
records in `results.jsonl` are skipped. To retry only failed cells:

```bash
python3 python/cuvs/benchmarks/run_falcon_pq_device_grid.py \
  --binary examples/cpp/falcon_grid/build/FALCON_PQ_DEVICE_GRID \
  --output-dir /tmp/falcon_pq_clean_results \
  --rerun-errors
```

For a short prefix of the grid:

```bash
python3 python/cuvs/benchmarks/run_falcon_pq_device_grid.py \
  --binary examples/cpp/falcon_grid/build/FALCON_PQ_DEVICE_GRID \
  --output-dir /tmp/falcon_pq_clean_results_subset \
  --max-cells 8
```

## Grid Definition

The runner covers:

- `pq_len={8,4,2,1}`, corresponding to `pq_dim={128,256,512,1024}`.
- Training rows: `256`, `1024`, `2048`, `4096` points/code and full `10M`.
- Initialization rows: `{256,1024,2048,4096}` for every training size.
- Full `10M` initialization is added only for full-training cells.
- Methods: `classic`, `scalable` Step 8 `random`, `scalable` Step 8
  `kmeans++`, and overall `random`.
- Fixed parameters: `pq_bits=8`, `n_clusters=256`, `seed=42`, `max_iter=300`,
  `tol=1e-4`, `topk=100`.

The runner sets `CUVS_KMEANS_PP_SYNC_STEPS=1` so k-means|| NVTX phase ranges are
synchronized with their CUDA work. The benchmark binary also sets
`CUVS_KMEANS_PP_STEP8` from the selected cell.

## Outputs

The output directory contains:

- `results.jsonl`: incremental raw records, one per completed cell.
- `results.csv`: flattened per-cell table.
- `report.md`: completed count and status summary.
- `logs/*.log`: raw stdout/stderr for each cell.

Each result records training, index build, encoding, search, recall@100, inertia,
observed peak device memory, per-subspace Lloyd iteration counts, and wall time.
