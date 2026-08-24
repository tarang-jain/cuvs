#!/usr/bin/env python3
"""Compare initialization cost and convergence for high-dimensional K-means."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cuvs.cluster.kmeans import KMeansParams, fit
from cuvs.common import Resources

from profile_kmeanspp_init import load_fbin_to_cupy


@dataclass(frozen=True)
class Case:
    name: str
    init_method: str
    oversampling: float
    init_size: int
    step8: str = "kmeans++"
    n_init: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="/datasets/blobs-5M-1536-k1024/base.fbin", type=Path
    )
    parser.add_argument("--num-rows", default=1_000_000, type=int)
    parser.add_argument("--n-clusters", default=1024, type=int)
    parser.add_argument("--max-iter", default=300, type=int)
    parser.add_argument("--tol", default=1e-4, type=float)
    parser.add_argument(
        "--host",
        action="store_true",
        help="Use a memory-mapped host input and the streamed K-means path.",
    )
    parser.add_argument(
        "--device-buffer-samples",
        default=1_000_000,
        type=int,
        help="Rows buffered on the GPU for host input.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Run only the named cases (default: run all cases).",
    )
    parser.add_argument(
        "--trace-iters",
        nargs="+",
        type=int,
        help="Report independent fits at these iteration limits instead of one full fit.",
    )
    return parser.parse_args()


def load_fbin_host(path: Path, num_rows: int) -> np.memmap:
    with path.open("rb") as f:
        file_rows, dim = map(int, np.fromfile(f, dtype=np.int32, count=2))
    if num_rows <= 0 or num_rows > file_rows:
        num_rows = file_rows
    print(f"  mapping {num_rows} x {dim} ({num_rows * dim * 4 / 1e9:.2f} GB) on host")
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(num_rows, dim))


def timed_fit(
    resources: Resources,
    x,
    case: Case,
    n_clusters: int,
    max_iter: int,
    tol: float,
    device_buffer_samples: int,
):
    os.environ["CUVS_KMEANS_PP_STEP8"] = case.step8
    params = KMeansParams(
        n_clusters=n_clusters,
        init_method=case.init_method,
        max_iter=max_iter,
        n_init=case.n_init,
        oversampling_factor=case.oversampling,
        init_size=case.init_size,
        tol=tol,
        device_buffer_samples=device_buffer_samples,
    )
    resources.sync()
    start = time.perf_counter()
    _, inertia, n_iter = fit(params, x, resources=resources)
    resources.sync()
    return time.perf_counter() - start, float(inertia), int(n_iter)


def main() -> None:
    args = parse_args()
    if args.host:
        x = load_fbin_host(args.dataset, args.num_rows)
        device_buffer_samples = args.device_buffer_samples
    else:
        x = load_fbin_to_cupy(args.dataset, num_rows=args.num_rows)
        device_buffer_samples = 0
    resources = Resources()
    cases = [
        Case("random", "Random", 2.0, 0),
        Case("random_n_init_4", "Random", 2.0, 0, n_init=4),
        Case("scalable_default_step8_kmeans++", "KMeansPlusPlus", 2.0, 0, "kmeans++"),
        Case(
            "scalable_full_step8_random",
            "KMeansPlusPlus",
            2.0,
            args.num_rows,
            "random",
        ),
        Case(
            "scalable_full_step8_kmeans++",
            "KMeansPlusPlus",
            2.0,
            args.num_rows,
            "kmeans++",
        ),
        Case("scalable_3k_step8_kmeans++", "KMeansPlusPlus", 2.0, 3 * args.n_clusters),
        Case("scalable_5k_step8_kmeans++", "KMeansPlusPlus", 2.0, 5 * args.n_clusters),
        Case("scalable_7k_step8_kmeans++", "KMeansPlusPlus", 2.0, 7 * args.n_clusters),
        Case("scalable_8k_step8_kmeans++", "KMeansPlusPlus", 2.0, 8 * args.n_clusters),
        Case(
            "scalable_10k_step8_kmeans++",
            "KMeansPlusPlus",
            2.0,
            10 * args.n_clusters,
            "kmeans++",
        ),
        Case("scalable_12k_step8_kmeans++", "KMeansPlusPlus", 2.0, 12 * args.n_clusters),
        Case(
            "scalable_12k_step8_random",
            "KMeansPlusPlus",
            2.0,
            12 * args.n_clusters,
            "random",
        ),
        Case("scalable_16k_step8_kmeans++", "KMeansPlusPlus", 2.0, 16 * args.n_clusters),
        Case("scalable_24k_step8_kmeans++", "KMeansPlusPlus", 2.0, 24 * args.n_clusters),
        Case("scalable_32k_step8_kmeans++", "KMeansPlusPlus", 2.0, 32 * args.n_clusters),
        Case("scalable_64k_step8_kmeans++", "KMeansPlusPlus", 2.0, 64 * args.n_clusters),
        Case(
            "scalable_256k_step8_kmeans++", "KMeansPlusPlus", 2.0, 256 * args.n_clusters
        ),
        Case("classic_10k_kmeans++", "KMeansPlusPlus", 0.0, 10 * args.n_clusters),
        Case("classic_12k_kmeans++", "KMeansPlusPlus", 0.0, 12 * args.n_clusters),
        Case("classic_32k_kmeans++", "KMeansPlusPlus", 0.0, 32 * args.n_clusters),
        Case("classic_64k_kmeans++", "KMeansPlusPlus", 0.0, 64 * args.n_clusters),
    ]

    if args.cases:
        requested = set(args.cases)
        cases = [case for case in cases if case.name in requested]
        missing = requested - {case.name for case in cases}
        if missing:
            raise ValueError(f"unknown cases: {', '.join(sorted(missing))}")

    for case in cases:
        if args.trace_iters:
            for max_iter in args.trace_iters:
                elapsed_s, inertia, n_iter = timed_fit(
                    resources,
                    x,
                    case,
                    args.n_clusters,
                    max_iter,
                    args.tol,
                    device_buffer_samples,
                )
                print(
                    "TRACE "
                    f"case={case.name} max_iter={max_iter} elapsed_s={elapsed_s:.6f} "
                    f"n_iter={n_iter} inertia={inertia:.9g}",
                    flush=True,
                )
            continue

        init_s, init_inertia, _ = timed_fit(
            resources,
            x,
            case,
            args.n_clusters,
            0,
            args.tol,
            device_buffer_samples,
        )
        total_s, inertia, n_iter = timed_fit(
            resources,
            x,
            case,
            args.n_clusters,
            args.max_iter,
            args.tol,
            device_buffer_samples,
        )
        print(
            "RESULT "
            f"case={case.name} init_s={init_s:.6f} total_s={total_s:.6f} "
            f"lloyd_est_s={max(0.0, total_s - init_s):.6f} "
            f"n_iter={n_iter} init_inertia={init_inertia:.9g} inertia={inertia:.9g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
