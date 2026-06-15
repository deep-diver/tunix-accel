#!/usr/bin/env python3
"""Merge sharded TPU v5e CCE result JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "v5e-cce-matrix" / "results"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  if not path.exists():
    return rows
  with path.open() as f:
    for line_number, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        rows.append({
            "status": "merge_warning",
            "source_file": str(path),
            "source_line": line_number,
            "error_message": f"invalid JSONL row: {exc}",
        })
        continue
      if isinstance(row, dict):
        row = dict(row)
        row["source_file"] = str(path)
        rows.append(row)
  return rows


def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
  try:
    experiment_id = int(row.get("experiment_id", 10**12))
  except (TypeError, ValueError):
    experiment_id = 10**12
  try:
    repeat_index = int(row.get("repeat_index", 10**12))
  except (TypeError, ValueError):
    repeat_index = 10**12
  return (experiment_id, repeat_index, str(row.get("source_file", "")))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames: list[str] = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="") as f:
    if not fieldnames:
      return
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
  parser.add_argument("--pattern", default="shard_*.jsonl")
  parser.add_argument("--out-jsonl", type=Path, default=None)
  parser.add_argument("--out-csv", type=Path, default=None)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  results_dir = args.results_dir.expanduser().resolve()
  out_jsonl = (
      args.out_jsonl.expanduser().resolve()
      if args.out_jsonl is not None
      else results_dir / "all_results.jsonl"
  )
  out_csv = (
      args.out_csv.expanduser().resolve()
      if args.out_csv is not None
      else results_dir / "all_results.csv"
  )

  rows: list[dict[str, Any]] = []
  for path in sorted(results_dir.glob(args.pattern)):
    if path.resolve() in {out_jsonl, out_csv}:
      continue
    rows.extend(read_jsonl(path))
  rows.sort(key=sort_key)

  write_jsonl(out_jsonl, rows)
  write_csv(out_csv, rows)
  print(f"input_files={len(list(results_dir.glob(args.pattern)))}")
  print(f"rows={len(rows)}")
  print(f"jsonl={out_jsonl}")
  print(f"csv={out_csv}")


if __name__ == "__main__":
  main()
