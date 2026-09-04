#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import glob
import json
import math
import pathlib


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(str(pathlib.Path(args.results_dir) / "*.json")))
    records = [load(path) for path in paths]
    records = [record for record in records if record.get("rows_used") == 10_000_000]
    serials = [record for record in records if record["mode"] == "serial_api"]
    if len(serials) != 1:
        raise SystemExit(f"expected one full serial baseline, found {len(serials)}")
    serial = serials[0]
    prototypes = sorted(
        [record for record in records if record["mode"] == "parallel_prototype"],
        key=lambda record: record["concurrency"],
    )
    if [record["concurrency"] for record in prototypes] != [1, 2, 4, 8, 16]:
        raise SystemExit("prototype sweep is incomplete")
    rows = []
    for record in [serial, *prototypes]:
        inertia_delta = (
            100.0 * (record["total_inertia"] - serial["total_inertia"]) / serial["total_inertia"]
            if serial["total_inertia"]
            else math.nan
        )
        rows.append(
            {
                "mode": record["mode"],
                "concurrency": record["concurrency"],
                "seconds": record["training_seconds"],
                "speedup_vs_serial_api": serial["training_seconds"] / record["training_seconds"],
                "total_inertia": record["total_inertia"],
                "inertia_delta_percent": inertia_delta,
                "convergence_status": record["convergence_status"],
                "gpu_peak_bytes": record["gpu_peak_bytes"],
                "gpu_training_peak_delta_bytes": record["gpu_training_peak_delta_bytes"],
                "host_peak_rss_bytes": record["host_peak_rss_bytes"],
            }
        )
    prefix = pathlib.Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_suffix(".md").open("w", encoding="utf-8") as stream:
        stream.write("# Falcon PQ training comparison\n\n")
        stream.write("| mode | concurrency | seconds | speedup | total inertia | inertia delta | convergence | GPU peak GiB |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---|---:|\n")
        for row in rows:
            stream.write(
                f"| {row['mode']} | {row['concurrency']} | {row['seconds']:.6f} | "
                f"{row['speedup_vs_serial_api']:.3f}x | {row['total_inertia']:.9g} | "
                f"{row['inertia_delta_percent']:.6f}% | {row['convergence_status']} | "
                f"{row['gpu_peak_bytes'] / 2**30:.3f} |\n"
            )
        best = max(rows[1:], key=lambda row: row["speedup_vs_serial_api"])
        stream.write(
            f"\nBest prototype: concurrency {best['concurrency']} at "
            f"{best['speedup_vs_serial_api']:.3f}x versus serial_api. "
            "Results are reported even when speedup is at or below 1.0x.\n"
        )
    print(prefix.with_suffix(".csv"))
    print(prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
