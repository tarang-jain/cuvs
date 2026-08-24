#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Run classic k-means with configurable initialization and Lloyd iterations.

Uses max_iter=0 so Lloyd iterations are skipped; NVTX ranges
``kmeans_initialization`` / ``initScalableKMeansPlusPlus`` / ``kmeanspp_*``
mark the seeding path for nsys/ncu.

Default: first 1M rows on device (1M x 1536 fits int32 flat indexing;
5M x 1536 overflows int32 on the device path).

Example (timeline + NVTX):
  nsys profile -t cuda,nvtx,osrt -o /tmp/kmeanspp_init \\
    --force-overwrite true \\
    python profile_kmeanspp_init.py \\
      --dataset /datasets/blobs-5M-1536-k1024/base.fbin

Example (kernel occupancy / metrics, kernels under NVTX init range):
  ncu --set full --nvtx --nvtx-include "initScalableKMeansPlusPlus/" \\
    -o /tmp/kmeanspp_init_ncu --force-overwrite \\
    python profile_kmeanspp_init.py \\
      --dataset /datasets/blobs-5M-1536-k1024/base.fbin --warmup 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cuvs.cluster.kmeans import KMeansParams, fit
from cuvs.common import Resources


def load_fbin_to_cupy(
    path: Path, *, num_rows: int | None = None, chunk_rows: int = 250_000
) -> cp.ndarray:
    """Stream .fbin (optionally truncated) to a device float32 matrix."""
    with open(path, "rb") as f:
        n, dim = np.fromfile(f, dtype=np.int32, count=2)
        n, dim = int(n), int(dim)
        if num_rows is not None:
            if num_rows <= 0:
                raise ValueError(f"--num-rows must be > 0, got {num_rows}")
            if num_rows > n:
                raise ValueError(f"--num-rows={num_rows} exceeds file n={n}")
            n = num_rows
        print(f"  loading {n} x {dim} ({n * dim * 4 / 1e9:.2f} GB) -> device ...", flush=True)
        X = cp.empty((n, dim), dtype=cp.float32)
        offset = 0
        while offset < n:
            m = min(chunk_rows, n - offset)
            host = np.fromfile(f, dtype=np.float32, count=m * dim).reshape(m, dim)
            X[offset : offset + m] = cp.asarray(host)
            offset += m
            if offset % (chunk_rows * 4) == 0 or offset == n:
                print(f"  H2D {offset}/{n}", flush=True)
    cp.cuda.Stream.null.synchronize()
    return X


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=str,
        default="/datasets/blobs-5M-1536-k1024/base.fbin",
        help="Path to base.fbin",
    )
    p.add_argument(
        "--num-rows",
        type=int,
        default=1_000_000,
        help="Rows to load from the start of the file (default 1M; use <=0 for all)",
    )
    p.add_argument("--n-clusters", type=int, default=1024)
    p.add_argument(
        "--init-method",
        choices=("KMeansPlusPlus", "Random"),
        default="KMeansPlusPlus",
    )
    p.add_argument(
        "--init-size",
        type=int,
        default=0,
        help="Rows used for initialization (0 = implementation default/full data)",
    )
    p.add_argument("--oversampling-factor", type=float, default=2.0)
    p.add_argument(
        "--max-iter",
        type=int,
        default=0,
        help="Lloyd iterations after init (0 = init only)",
    )
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0, help="Warmup fits before timed run")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(path)

    num_rows = None if args.num_rows is not None and args.num_rows <= 0 else args.num_rows

    print(f"Loading {path} ...")
    t0 = time.perf_counter()
    X = load_fbin_to_cupy(path, num_rows=num_rows)
    print(f"  device shape={X.shape} dtype={X.dtype} in {time.perf_counter()-t0:.1f}s")

    resources = Resources()
    params = KMeansParams(
        n_clusters=args.n_clusters,
        init_method=args.init_method,
        max_iter=args.max_iter,
        n_init=args.n_init,
        oversampling_factor=args.oversampling_factor,
        init_size=args.init_size,
    )
    print(
        f"KMeansParams(n_clusters={params.n_clusters}, init={params.init_method}, "
        f"max_iter={params.max_iter}, oversampling_factor={params.oversampling_factor}, "
        f"n_init={params.n_init}, init_size={params.init_size})"
    )

    def once(label: str) -> None:
        resources.sync()
        t1 = time.perf_counter()
        centroids, inertia, n_iter = fit(params, X, resources=resources)
        resources.sync()
        elapsed = time.perf_counter() - t1
        print(
            f"  {label}: {elapsed:.3f}s  "
            f"centroids={centroids.shape} inertia={float(inertia):.6g} n_iter={int(n_iter)}"
        )

    for i in range(args.warmup):
        once(f"warmup[{i}]")

    once("timed")


if __name__ == "__main__":
    main()
