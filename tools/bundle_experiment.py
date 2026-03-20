#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from contest_preflight import validate_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle an experiment or record folder.")
    parser.add_argument("path", help="Path to the experiment or record folder.")
    parser.add_argument(
        "--output-dir",
        default="dist",
        help="Directory where the tar.gz artifact should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.path).resolve()
    if not path.is_dir():
        raise SystemExit(f"{path} is not a directory")

    record_mode = "records" in path.parts
    errors, warnings = validate_dir(path, record_mode=record_mode)
    if errors:
        for error in errors:
            print(f"error: {error}")
        raise SystemExit("bundle aborted due to preflight failure")
    for warning in warnings:
        print(f"warning: {warning}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / f"{path.name}.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(path, arcname=path.name)

    print(f"bundle created: {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
