#!/usr/bin/env python3
"""Run and incrementally report the Falcon 10M device PQ/IVF-PQ grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REGULAR_INIT_SIZES = (256, 1024, 2048, 4096)
TRAIN_CASES = (
    (256, 65_536),
    (1024, 262_144),
    (2048, 524_288),
    (4096, 1_048_576),
    (0, 10_000_000),
)
PQ_LENS = (8, 4, 2, 1)


def build_grid() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for pq_len in PQ_LENS:
        for max_train_per_code, train_rows in TRAIN_CASES:
            init_sizes = list(REGULAR_INIT_SIZES)
            if train_rows == 10_000_000:
                init_sizes.append(10_000_000)
            for init_size in init_sizes:
                cells.append(
                    cell(pq_len, max_train_per_code, train_rows, init_size, "classic", "na")
                )
                cells.append(
                    cell(
                        pq_len,
                        max_train_per_code,
                        train_rows,
                        init_size,
                        "scalable",
                        "random",
                    )
                )
                cells.append(
                    cell(
                        pq_len,
                        max_train_per_code,
                        train_rows,
                        init_size,
                        "scalable",
                        "kmeans++",
                    )
                )
            cells.append(cell(pq_len, max_train_per_code, train_rows, 0, "random", "na"))
    if len(cells) != 272:
        raise RuntimeError(f"grid construction error: expected 272 cells, got {len(cells)}")
    return cells


def cell(
    pq_len: int,
    max_train_per_code: int,
    train_rows: int,
    init_size: int,
    method: str,
    step8: str,
) -> dict[str, object]:
    train_name = "full" if train_rows == 10_000_000 else str(max_train_per_code)
    init_name = "na" if method == "random" else str(init_size)
    identifier = f"len{pq_len}_train{train_name}_init{init_name}_{method}_{step8}"
    return {
        "cell_id": identifier,
        "pq_len": pq_len,
        "max_train_per_code": max_train_per_code,
        "train_rows": train_rows,
        "init_size": init_size,
        "method": method,
        "step8": step8,
    }


def read_completed(path: Path) -> dict[str, dict[str, object]]:
    completed: dict[str, dict[str, object]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        completed[str(record["cell_id"])] = record
    return completed


def write_reports(output_dir: Path, records: dict[str, dict[str, object]]) -> None:
    ordered = [records[key] for key in sorted(records)]
    csv_path = output_dir / "results.csv"
    fields = sorted({key for record in ordered for key in record if key != "subspace_iterations"})
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    status_counts: dict[str, int] = {}
    for record in ordered:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = [
        "# Falcon 10M device PQ grid",
        "",
        f"Completed records: {len(ordered)} / 272",
        "",
        "## Status",
        "",
    ]
    summary.extend(f"- {status}: {count}" for status, count in sorted(status_counts.items()))
    summary.extend(
        [
            "",
            "Full per-cell values are in `results.csv`; raw output is in `logs/`.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(summary), encoding="utf-8")


def run_cell(
    binary: Path,
    output_dir: Path,
    spec: dict[str, object],
    smoke: bool,
) -> dict[str, object]:
    command = [
        str(binary),
        "--pq-len",
        str(spec["pq_len"]),
        "--method",
        str(spec["method"]),
        "--step8",
        str(spec["step8"] if spec["step8"] != "na" else "kmeans++"),
        "--train-rows",
        str(spec["train_rows"]),
        "--max-train-per-code",
        str(spec["max_train_per_code"]),
        "--init-size",
        str(spec["init_size"]),
        "--max-iter",
        "300",
        "--tol",
        "1e-4",
        "--topk",
        "100",
    ]
    if smoke:
        command.extend(["--row-cap", "4096", "--train-rows", "4096", "--init-size", "256"])
    environment = os.environ.copy()
    environment["CUVS_KMEANS_PP_SYNC_STEPS"] = "1"
    log_path = output_dir / "logs" / f"{spec['cell_id']}.log"
    started = time.time()
    process = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
    result_line = next(
        (line.removeprefix("RESULT_JSON ") for line in reversed(process.stdout.splitlines())
         if line.startswith("RESULT_JSON ")),
        None,
    )
    if result_line is None:
        result: dict[str, object] = {
            "status": "error",
            "returncode": process.returncode,
            "error": process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "no result",
        }
    else:
        result = json.loads(result_line)
        result["returncode"] = process.returncode
    result.update(spec)
    result["wall_seconds"] = time.time() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("examples/cpp/falcon_grid/build/FALCON_PQ_DEVICE_GRID"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/falcon_pq_clean_results"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--max-cells", type=int, default=0)
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"benchmark binary not found: {binary}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    records = read_completed(jsonl_path)

    grid = build_grid()
    if args.smoke:
        grid = [
            cell(pq_len, 0, 4096, 256, "scalable", "kmeans++") for pq_len in PQ_LENS
        ]
    if args.max_cells > 0:
        grid = grid[: args.max_cells]

    for position, spec in enumerate(grid, start=1):
        identifier = str(spec["cell_id"])
        previous = records.get(identifier)
        if previous and not (args.rerun_errors and previous.get("status") == "error"):
            print(f"[{position}/{len(grid)}] skip {identifier}", flush=True)
            continue
        print(f"[{position}/{len(grid)}] run {identifier}", flush=True)
        record = run_cell(binary, output_dir, spec, args.smoke)
        with jsonl_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        records[identifier] = record
        write_reports(output_dir, records)
        print(
            f"  {record.get('status')} recall@100={record.get('recall_at_100', 'na')} "
            f"train_ms={record.get('train_ms', 'na')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
