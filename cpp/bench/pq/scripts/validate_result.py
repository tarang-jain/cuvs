#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--allow-missing-inertia", action="store_true")
    args = parser.parse_args()
    with open(args.result, encoding="utf-8") as stream:
        result = json.load(stream)
    assert result["dataset_header_rows"] == 10_000_000
    assert result["dataset_header_dim"] == 1024
    assert result["rows_used"] == args.expected_rows
    assert result["codebook_shape"] == [512, 256, 2]
    assert result["dataset_finite"] is True
    assert result["centroids_finite"] is True
    assert result["output_validated"] is True
    assert math.isfinite(result["training_seconds"]) and result["training_seconds"] > 0
    assert result["convergence_status"]
    if args.require_complete:
        assert result["complete_row_usage"] is True
    if not args.allow_missing_inertia:
        assert len(result["subspaces"]) == 512
        assert math.isfinite(result["total_inertia"]) and result["total_inertia"] >= 0
        for subspace in result["subspaces"]:
            assert math.isfinite(subspace["inertia"]) and subspace["inertia"] >= 0
            if result["mode"] == "parallel_prototype":
                assert isinstance(subspace["converged"], bool)
                assert 0 <= subspace["iterations"] <= 300
    print(f"validated {args.result}")


if __name__ == "__main__":
    main()
