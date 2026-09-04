#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime as dt
import json
import os
import pathlib
import socket
import struct
import subprocess
import tempfile


def output(command, cwd=None):
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def atomic_write(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    temporary.replace(path)


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def init(args):
    dataset = pathlib.Path(args.dataset).resolve()
    with dataset.open("rb") as stream:
        rows, dim = struct.unpack("<II", stream.read(8))
    expected_bytes = 8 + rows * dim * 4
    actual_bytes = dataset.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(f"invalid fbin size: expected {expected_bytes}, got {actual_bytes}")
    refs = {
        name: output(["git", "rev-parse", f"upstream/pr/{name}"], args.worktree)
        for name in ("2249", "2548", "2552")
    }
    gpu_line = output(
        [
            "nvidia-smi",
            f"--id={args.gpu_index}",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    manifest = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "worktree": str(pathlib.Path(args.worktree).resolve()),
        "branch": output(["git", "branch", "--show-current"], args.worktree),
        "branch_commit": output(["git", "rev-parse", "HEAD"], args.worktree),
        "upstream_main_commit": output(["git", "rev-parse", "upstream/main"], args.worktree),
        "pr_head_commits": refs,
        "dataset": {
            "path": str(dataset),
            "header_rows": rows,
            "header_dim": dim,
            "actual_bytes": actual_bytes,
            "expected_bytes": expected_bytes,
        },
        "host": {
            "hostname": socket.gethostname(),
            "kernel": output(["uname", "-srvmo"]),
        },
        "gpu": {
            "physical_index": int(args.gpu_index),
            "nvidia_smi_record": gpu_line,
        },
        "tools": {
            "cmake": output(["cmake", "--version"]),
            "nvcc": output(["nvcc", "--version"]),
            "nsys": output(["nsys", "--version"]),
            "ncu": output(["ncu", "--version"]),
            "tmux": output(["tmux", "-V"]),
            "compiler": output(["c++", "--version"]),
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        },
        "parameters": {
            "pq_dim": 512,
            "pq_len": 2,
            "pq_bits": 8,
            "centroids": 256,
            "kmeans": "regular",
            "initialization": "scalable KMeans||",
            "oversampling_factor": 2.0,
            "seed": 42,
            "n_init": 1,
            "max_iter": 300,
            "tolerance": 1e-4,
            "max_train_points_per_pq_code": 39063,
            "concurrency_sweep": [1, 2, 4, 8, 16],
        },
        "artifact_directory": str(pathlib.Path(args.artifact_dir).resolve()),
        "tmux_session": args.tmux_session,
        "limitations": [
            "tmux protects against SSH or client disconnection",
            "tmux does not protect against host reboot, administrator termination, GPU reset, or machine failure",
        ],
        "commands": [],
        "stages": {},
        "artifacts": {},
    }
    atomic_write(args.manifest, manifest)


def record_command(args):
    manifest = load(args.manifest)
    manifest["commands"].append(
        {
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stage": args.stage,
            "command": args.command,
        }
    )
    atomic_write(args.manifest, manifest)


def stage(args):
    manifest = load(args.manifest)
    manifest["stages"][args.stage] = {
        "status": args.status,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "log": args.log,
        "marker": args.marker,
    }
    atomic_write(args.manifest, manifest)


def artifact(args):
    manifest = load(args.manifest)
    manifest["artifacts"][args.name] = args.path
    atomic_write(args.manifest, manifest)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--manifest", required=True)
    init_parser.add_argument("--worktree", required=True)
    init_parser.add_argument("--dataset", required=True)
    init_parser.add_argument("--gpu-index", required=True)
    init_parser.add_argument("--artifact-dir", required=True)
    init_parser.add_argument("--tmux-session", required=True)
    init_parser.set_defaults(function=init)
    command_parser = subparsers.add_parser("command")
    command_parser.add_argument("--manifest", required=True)
    command_parser.add_argument("--stage", required=True)
    command_parser.add_argument("--command", required=True)
    command_parser.set_defaults(function=record_command)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--manifest", required=True)
    stage_parser.add_argument("--stage", required=True)
    stage_parser.add_argument("--status", required=True)
    stage_parser.add_argument("--log", required=True)
    stage_parser.add_argument("--marker", required=True)
    stage_parser.set_defaults(function=stage)
    artifact_parser = subparsers.add_parser("artifact")
    artifact_parser.add_argument("--manifest", required=True)
    artifact_parser.add_argument("--name", required=True)
    artifact_parser.add_argument("--path", required=True)
    artifact_parser.set_defaults(function=artifact)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
