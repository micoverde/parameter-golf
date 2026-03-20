#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path

RECORD_REQUIRED = ("README.md", "submission.json", "train_gpt.py")
EXPERIMENT_REQUIRED = ("README.md", "train_gpt.py")
LOG_PATTERNS = ("train*.log", "train*.txt")
MIN_SUBMISSION_KEYS = ("author", "name")
METRIC_KEYS = ("val_bpb", "mean_val_bpb", "val_loss", "mean_val_loss")
BYTE_KEYS = ("bytes_total", "artifact_bytes", "code_bytes", "bytes_code")


def find_candidate_dirs(root: Path) -> list[Path]:
    if root.name == "records":
        candidates: list[Path] = []
        for track_dir in sorted(root.glob("track_*")):
            candidates.extend(sorted(path for path in track_dir.iterdir() if path.is_dir()))
        return candidates
    return sorted(path for path in root.iterdir() if path.is_dir())


def gather_log_files(path: Path) -> list[Path]:
    logs: list[Path] = []
    for pattern in LOG_PATTERNS:
        logs.extend(sorted(path.glob(pattern)))
    return logs


def compile_file(path: Path) -> str | None:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        return str(exc)
    return None


def validate_submission_json(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON in {path}: {exc}"]

    missing = [key for key in MIN_SUBMISSION_KEYS if key not in data]
    if missing:
        errors.append(f"{path} missing keys: {', '.join(missing)}")
    if not any(key in data for key in METRIC_KEYS):
        errors.append(f"{path} missing any metric key from: {', '.join(METRIC_KEYS)}")
    if not any(key in data for key in BYTE_KEYS):
        errors.append(f"{path} missing any byte key from: {', '.join(BYTE_KEYS)}")
    return errors


def validate_dir(path: Path, record_mode: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = RECORD_REQUIRED if record_mode else EXPERIMENT_REQUIRED

    for name in required:
        file_path = path / name
        if not file_path.exists():
            errors.append(f"{path}: missing required file {name}")

    train_script = path / "train_gpt.py"
    if train_script.exists():
        compile_error = compile_file(train_script)
        if compile_error:
            errors.append(f"{train_script}: {compile_error}")

    readme = path / "README.md"
    if readme.exists() and not readme.read_text(encoding="utf-8").strip():
        errors.append(f"{readme}: file is empty")

    submission_json = path / "submission.json"
    if submission_json.exists():
        errors.extend(validate_submission_json(submission_json))

    logs = gather_log_files(path)
    if record_mode and not logs:
        errors.append(
            f"{path}: missing canonical train log (expected one of {', '.join(LOG_PATTERNS)})"
        )
    if not record_mode and not logs:
        warnings.append(f"{path}: no train log present yet")

    return errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate contest record and experiment folders.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=("records", "experiments"),
        help="Directories to validate. Defaults to 'records experiments'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    checked_dirs = 0

    for raw_path in args.paths:
        path = Path(raw_path).resolve()
        if not path.exists():
            continue
        if path.is_file():
            errors.append(f"{path}: expected a directory")
            continue

        candidate_dirs: list[Path]
        if path.name in {"records", "experiments"}:
            candidate_dirs = find_candidate_dirs(path)
        else:
            candidate_dirs = [path]

        record_mode = "records" in path.parts
        for candidate in candidate_dirs:
            checked_dirs += 1
            dir_errors, dir_warnings = validate_dir(candidate, record_mode=record_mode or "track_" in candidate.parent.name)
            errors.extend(dir_errors)
            warnings.extend(dir_warnings)

    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")

    if errors:
        print(f"preflight failed: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1

    print(f"preflight passed: checked {checked_dirs} folder(s) with {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
