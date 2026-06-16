#!/usr/bin/env python3
"""Analyze dense mesh/chunk pathology sweep outputs.

The script is intentionally tolerant of partial shards and failures.  It reads
whatever JSONL rows are available, writes coverage/failure tables, and only uses
successful timing rows for plots and the lightweight log-time model.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OPERATION_ORDER = [
    "chunked_matmul_loop",
    "collective_loop",
    "projection_collective_loop",
]
MESH_ORDER = ["fsdp=4,tp=1", "fsdp=2,tp=2", "fsdp=1,tp=4"]
PLOT_COLORS = {
    "chunked_matmul_loop": "#4C78A8",
    "collective_loop": "#F58518",
    "projection_collective_loop": "#54A24B",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  if not path.exists():
    return rows
  with path.open() as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  keys: list[str] = []
  for row in rows:
    for key in row:
      if key not in keys:
        keys.append(key)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def expand_input_patterns(patterns: list[str]) -> list[Path]:
  paths: list[Path] = []
  for pattern in patterns:
    matches = [Path(item) for item in glob.glob(pattern)]
    if matches:
      paths.extend(matches)
    else:
      paths.append(Path(pattern))
  seen = set()
  unique = []
  for path in paths:
    resolved = path.expanduser().resolve()
    if resolved not in seen:
      seen.add(resolved)
      unique.append(resolved)
  return unique


def infer_stack(row: pd.Series) -> str:
  runner = str(row.get("workload_runner", ""))
  mode = str(row.get("execution_mode", ""))
  framework = str(row.get("framework", ""))
  if framework == "pytorch" or runner == "torch_microbench":
    return "torch.compile" if mode == "compile" else "PyTorch eager"
  if runner == "jax_microbench" or str(row.get("jax_backend", "")):
    return "JAX/XLA"
  if runner == "tunix_cce":
    return "Tunix CCE"
  return "unknown"


def normalize(rows: list[dict[str, Any]]) -> pd.DataFrame:
  data = pd.DataFrame(rows)
  if data.empty:
    return data
  for col in [
      "steady_state_mean_step_time_sec",
      "median_step_time_sec",
      "p95_step_time_sec",
      "compile_time_sec",
      "first_step_time_sec",
      "tokens_per_sec",
      "runtime_hbm_peak_gb",
      "xla_planned_hbm_gib_per_chip",
      "token_loop_count",
      "vocab_loop_count",
      "chunk_loop_count",
      "token_chunk",
      "vocab_chunk",
      "tp_degree",
      "fsdp_degree",
      "global_batch_size",
      "sequence_length",
  ]:
    if col not in data:
      data[col] = np.nan
    data[col] = pd.to_numeric(data[col], errors="coerce")
  if "operation_family" not in data:
    data["operation_family"] = data.get("workload_family", "")
  data["operation_family"] = data["operation_family"].fillna(data.get("workload_family", ""))
  data["execution_stack"] = data.apply(infer_stack, axis=1)
  data["has_collective"] = data["operation_family"].astype(str).str.contains("collective")
  data["has_matmul"] = data["operation_family"].astype(str).isin(
      ["chunked_matmul_loop", "projection_collective_loop"]
  )
  data["log_step_time"] = np.log(data["steady_state_mean_step_time_sec"])
  data["log_chunk_loop_count"] = np.log(data["chunk_loop_count"])
  data["log_token_loop_count"] = np.log(data["token_loop_count"])
  data["log_vocab_loop_count"] = np.log(data["vocab_loop_count"])
  data["hbm_gib"] = data["xla_planned_hbm_gib_per_chip"].where(
      data["xla_planned_hbm_gib_per_chip"].notna(),
      data["runtime_hbm_peak_gb"],
  )
  group_cols = [
      "hardware_target",
      "execution_stack",
      "operation_family",
      "shape",
      "mesh_configuration",
  ]
  success = data["status"].eq("success") & data["steady_state_mean_step_time_sec"].notna()
  best = (
      data[success]
      .groupby(group_cols, dropna=False)["steady_state_mean_step_time_sec"]
      .transform("min")
  )
  data.loc[success, "relative_to_best_same_group"] = (
      data.loc[success, "steady_state_mean_step_time_sec"] / best
  )
  return data


def success_rows(data: pd.DataFrame) -> pd.DataFrame:
  if data.empty:
    return data
  return data[
      data["status"].eq("success")
      & data["steady_state_mean_step_time_sec"].notna()
      & np.isfinite(data["steady_state_mean_step_time_sec"])
  ].copy()


def coverage_tables(data: pd.DataFrame, manifest: pd.DataFrame | None, outdir: Path) -> dict[str, Path]:
  outputs: dict[str, Path] = {}
  if not data.empty:
    coverage = (
        data.groupby(["hardware_target", "execution_stack", "operation_family", "status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["hardware_target", "execution_stack", "operation_family", "status"])
    )
    path = outdir / "coverage_by_stack_operation_status.csv"
    coverage.to_csv(path, index=False)
    outputs["coverage"] = path

    failures = data[~data["status"].eq("success")].copy()
    if not failures.empty:
      cols = [
          "experiment_id",
          "hardware_target",
          "execution_stack",
          "operation_family",
          "shape",
          "mesh_configuration",
          "chunk_configuration",
          "status",
          "failure_type",
          "error_message",
      ]
      for col in cols:
        if col not in failures:
          failures[col] = ""
      path = outdir / "failures.csv"
      failures[cols].to_csv(path, index=False)
      outputs["failures"] = path

  if manifest is not None and not manifest.empty:
    observed = set(data.get("experiment_id", pd.Series(dtype=int)).dropna().astype(int).tolist())
    manifest_copy = manifest.copy()
    manifest_copy["observed"] = manifest_copy["experiment_id"].astype(int).isin(observed)
    missing = manifest_copy[~manifest_copy["observed"]]
    path = outdir / "missing_experiments.csv"
    missing.to_csv(path, index=False)
    outputs["missing"] = path
  return outputs


def fastest_table(success: pd.DataFrame, outdir: Path) -> Path | None:
  if success.empty:
    return None
  group_cols = [
      "hardware_target",
      "execution_stack",
      "operation_family",
      "shape",
      "mesh_configuration",
  ]
  idx = success.groupby(group_cols, dropna=False)["steady_state_mean_step_time_sec"].idxmin()
  cols = group_cols + [
      "chunk_configuration",
      "token_loop_count",
      "vocab_loop_count",
      "chunk_loop_count",
      "steady_state_mean_step_time_sec",
      "tokens_per_sec",
      "hbm_gib",
  ]
  path = outdir / "fastest_per_stack_operation_mesh_shape.csv"
  success.loc[idx, cols].sort_values(group_cols).to_csv(path, index=False)
  return path


def outlier_table(success: pd.DataFrame, outdir: Path) -> Path | None:
  if success.empty or "relative_to_best_same_group" not in success:
    return None
  outliers = success[success["relative_to_best_same_group"].ge(2.0)].copy()
  if outliers.empty:
    outliers = success.iloc[0:0].copy()
  outliers["outlier_band"] = pd.cut(
      outliers["relative_to_best_same_group"],
      bins=[2.0, 5.0, 10.0, float("inf")],
      labels=[">=2x", ">=5x", ">=10x"],
      right=False,
  )
  cols = [
      "hardware_target",
      "execution_stack",
      "operation_family",
      "shape",
      "mesh_configuration",
      "chunk_configuration",
      "token_loop_count",
      "vocab_loop_count",
      "chunk_loop_count",
      "steady_state_mean_step_time_sec",
      "relative_to_best_same_group",
      "outlier_band",
      "hbm_gib",
  ]
  path = outdir / "outlier_report.csv"
  outliers[cols].sort_values(
      ["relative_to_best_same_group"], ascending=False
  ).to_csv(path, index=False)
  return path


def recovery_table(success: pd.DataFrame, outdir: Path) -> Path | None:
  if success.empty:
    return None
  rows: list[dict[str, Any]] = []
  group_cols = [
      "hardware_target",
      "execution_stack",
      "operation_family",
      "shape",
      "mesh_configuration",
  ]
  for group_key, part in success.groupby(group_cols, dropna=False):
    if len(part) < 2:
      continue
    slow = part.sort_values("steady_state_mean_step_time_sec", ascending=False).iloc[0]
    fast = part.sort_values("steady_state_mean_step_time_sec", ascending=True).iloc[0]
    if float(fast["steady_state_mean_step_time_sec"]) <= 0:
      continue
    speedup = float(slow["steady_state_mean_step_time_sec"]) / float(
        fast["steady_state_mean_step_time_sec"]
    )
    if speedup < 2.0 or float(fast["chunk_loop_count"]) >= float(slow["chunk_loop_count"]):
      continue
    slow_hbm = slow.get("hbm_gib")
    fast_hbm = fast.get("hbm_gib")
    if pd.notna(slow_hbm) and pd.notna(fast_hbm) and float(slow_hbm) > 0:
      hbm_increase = (float(fast_hbm) - float(slow_hbm)) / float(slow_hbm)
      if hbm_increase > 0.10:
        continue
    else:
      hbm_increase = np.nan
    rows.append({
        "hardware_target": group_key[0],
        "execution_stack": group_key[1],
        "operation_family": group_key[2],
        "shape": group_key[3],
        "mesh_configuration": group_key[4],
        "slow_chunk": slow["chunk_configuration"],
        "fast_chunk": fast["chunk_configuration"],
        "slow_loop_count": slow["chunk_loop_count"],
        "fast_loop_count": fast["chunk_loop_count"],
        "slow_step_time_sec": slow["steady_state_mean_step_time_sec"],
        "fast_step_time_sec": fast["steady_state_mean_step_time_sec"],
        "recovered_speedup": speedup,
        "slow_hbm_gib": slow_hbm,
        "fast_hbm_gib": fast_hbm,
        "hbm_relative_increase": hbm_increase,
    })
  path = outdir / "recovery_report.csv"
  write_csv(path, rows)
  return path


def regression_table(success: pd.DataFrame, outdir: Path) -> Path | None:
  model_data = success[
      success["log_step_time"].notna()
      & success["log_chunk_loop_count"].notna()
      & np.isfinite(success["log_step_time"])
      & np.isfinite(success["log_chunk_loop_count"])
  ].copy()
  if len(model_data) < 12:
    return None
  base_features = pd.DataFrame({
      "intercept": 1.0,
      "log_chunk_loop_count": model_data["log_chunk_loop_count"],
      "log_token_loop_count": model_data["log_token_loop_count"],
      "log_vocab_loop_count": model_data["log_vocab_loop_count"],
      "tp_degree": model_data["tp_degree"],
      "has_collective": model_data["has_collective"].astype(float),
      "has_matmul": model_data["has_matmul"].astype(float),
  })
  base_features["log_loop_x_tp"] = (
      base_features["log_chunk_loop_count"] * base_features["tp_degree"]
  )
  base_features["log_loop_x_collective"] = (
      base_features["log_chunk_loop_count"] * base_features["has_collective"]
  )
  base_features["tp_x_collective"] = base_features["tp_degree"] * base_features["has_collective"]
  base_features["log_loop_x_tp_x_collective"] = (
      base_features["log_chunk_loop_count"]
      * base_features["tp_degree"]
      * base_features["has_collective"]
  )
  fixed_effects = pd.get_dummies(
      model_data[["hardware_target", "execution_stack", "operation_family", "shape"]],
      prefix=["hw", "stack", "op", "shape"],
      drop_first=True,
      dtype=float,
  )
  features = pd.concat([base_features, fixed_effects], axis=1)
  valid = features.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
  features = features[valid]
  y = model_data.loc[valid, "log_step_time"].to_numpy(dtype=float)
  if len(y) <= features.shape[1]:
    return None
  x = features.to_numpy(dtype=float)
  coef, *_ = np.linalg.lstsq(x, y, rcond=None)
  pred = x @ coef
  ss_res = float(np.sum((y - pred) ** 2))
  ss_tot = float(np.sum((y - np.mean(y)) ** 2))
  r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
  rows = [
      {
          "feature": feature,
          "coefficient_log_seconds": value,
          "multiplicative_effect": math.exp(value)
          if feature != "intercept" and abs(value) < 50
          else "",
          "abs_coefficient": abs(value),
          "n_rows": len(y),
          "r2": r2,
      }
      for feature, value in zip(features.columns, coef, strict=True)
  ]
  rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
  path = outdir / "log_step_time_model_coefficients.csv"
  write_csv(path, rows)
  return path


def plot_factor_scatter(success: pd.DataFrame, outdir: Path) -> list[Path]:
  outputs: list[Path] = []
  if success.empty or "relative_to_best_same_group" not in success:
    return outputs
  for (hardware, stack), part in success.groupby(["hardware_target", "execution_stack"], dropna=False):
    if part.empty:
      continue
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, operation in zip(axes, OPERATION_ORDER, strict=True):
      op_part = part[part["operation_family"].astype(str).eq(operation)]
      for mesh in MESH_ORDER:
        mesh_part = op_part[op_part["mesh_configuration"].astype(str).eq(mesh)]
        if mesh_part.empty:
          continue
        ax.scatter(
            mesh_part["chunk_loop_count"],
            mesh_part["relative_to_best_same_group"],
            s=18 + 9 * np.log2(mesh_part["vocab_loop_count"].clip(lower=1)),
            alpha=0.72,
            label=mesh,
            c=mesh_part["token_chunk"],
            cmap="viridis",
            edgecolors="none",
        )
      ax.axhline(1.0, color="#333333", linewidth=1)
      ax.axhline(2.0, color="#999999", linewidth=0.8, linestyle="--")
      ax.axhline(5.0, color="#BBBBBB", linewidth=0.8, linestyle=":")
      ax.set_xscale("log", base=2)
      ax.set_yscale("log", base=2)
      ax.set_title(operation.replace("_", " "))
      ax.set_xlabel("chunk loop count")
      ax.grid(alpha=0.22)
    axes[0].set_ylabel("relative step time vs best same mesh/shape/op")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{hardware} / {stack}: loop granularity within each operation")
    fig.tight_layout()
    safe = f"{hardware}_{stack}".replace("/", "_").replace(" ", "_")
    path = outdir / f"{safe}_chunk_loop_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
  return outputs


def plot_heatmaps(success: pd.DataFrame, outdir: Path) -> list[Path]:
  outputs: list[Path] = []
  if success.empty or "relative_to_best_same_group" not in success:
    return outputs
  for (hardware, stack, shape), part in success.groupby(
      ["hardware_target", "execution_stack", "shape"], dropna=False
  ):
    token_chunks = sorted(part["token_chunk"].dropna().unique())
    vocab_chunks = sorted(part["vocab_chunk"].dropna().unique())
    if not token_chunks or not vocab_chunks:
      continue
    fig, axes = plt.subplots(
        len(OPERATION_ORDER),
        len(MESH_ORDER),
        figsize=(13, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    values_for_scale = part["relative_to_best_same_group"].replace([np.inf, -np.inf], np.nan)
    vmax = float(np.nanpercentile(values_for_scale, 95)) if values_for_scale.notna().any() else 1.0
    vmax = max(vmax, 1.0)
    for row_index, operation in enumerate(OPERATION_ORDER):
      for col_index, mesh in enumerate(MESH_ORDER):
        ax = axes[row_index][col_index]
        sub = part[
            part["operation_family"].astype(str).eq(operation)
            & part["mesh_configuration"].astype(str).eq(mesh)
        ]
        grid = np.full((len(token_chunks), len(vocab_chunks)), np.nan)
        for i, token_chunk in enumerate(token_chunks):
          for j, vocab_chunk in enumerate(vocab_chunks):
            value = sub[
                sub["token_chunk"].eq(token_chunk) & sub["vocab_chunk"].eq(vocab_chunk)
            ]["relative_to_best_same_group"]
            if not value.empty:
              grid[i, j] = float(value.median())
        image = ax.imshow(grid, aspect="auto", vmin=1.0, vmax=vmax, cmap="magma")
        ax.set_title(f"{operation.replace('_loop', '').replace('_', ' ')}\n{mesh}", fontsize=9)
        ax.set_xticks(range(len(vocab_chunks)))
        ax.set_xticklabels([str(int(item)) for item in vocab_chunks], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(token_chunks)))
        ax.set_yticklabels([str(int(item)) for item in token_chunks], fontsize=8)
        if col_index == 0:
          ax.set_ylabel("token chunk")
        if row_index == len(OPERATION_ORDER) - 1:
          ax.set_xlabel("vocab chunk")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="relative step time")
    fig.suptitle(f"{hardware} / {stack} / {shape}: token x vocab chunk sweep")
    safe = f"{hardware}_{stack}_{shape}".replace("/", "_").replace(" ", "_")
    path = outdir / f"{safe}_token_vocab_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
  return outputs


def write_markdown_report(
    *,
    outdir: Path,
    data: pd.DataFrame,
    success: pd.DataFrame,
    outputs: dict[str, Path | list[Path] | None],
) -> Path:
  report_path = outdir / "dense_sweep_analysis.md"
  lines = [
      "# Dense Mesh/Chunk Pathology Sweep Analysis",
      "",
      "## Data status",
      "",
  ]
  lines.append(f"- Rows loaded: {len(data)}")
  lines.append(f"- Successful timing rows: {len(success)}")
  if not data.empty:
    lines.append(f"- Hardware targets: {', '.join(sorted(map(str, data['hardware_target'].dropna().unique())))}")
    lines.append(f"- Execution stacks: {', '.join(sorted(map(str, data['execution_stack'].dropna().unique())))}")
    lines.append(f"- Operations: {', '.join(sorted(map(str, data['operation_family'].dropna().unique())))}")
  lines.extend([
      "",
      "## Hypothesis Test",
      "",
      "Hypothesis: CCE memory benefit is mostly governed by loss-logits dominance, while throughput failure is governed by the interaction between chunk-loop granularity and FSDP/TP mesh layout.",
      "",
      "This script tests the throughput half with synthetic rows by modeling log step time using chunk-loop count, token/vocab loop counts, TP degree, collective presence, and their interactions. Memory evidence is carried through planned/runtime HBM columns when available; missing memory extraction is non-fatal.",
      "",
  ])
  regression = outputs.get("regression")
  if regression:
    lines.append(f"- Log-time model coefficients: `{Path(regression).name}`")
  else:
    lines.append("- Log-time model coefficients: not generated; not enough successful rows yet.")
  fastest = outputs.get("fastest")
  if fastest:
    lines.append(f"- Fastest configuration table: `{Path(fastest).name}`")
  outliers = outputs.get("outliers")
  if outliers:
    lines.append(f"- Outlier report: `{Path(outliers).name}`")
  recovery = outputs.get("recovery")
  if recovery:
    lines.append(f"- Recovery report: `{Path(recovery).name}`")
  plot_paths = outputs.get("plots") or []
  if plot_paths:
    lines.extend(["", "## Plots", ""])
    for path in plot_paths:
      lines.append(f"- `{Path(path).name}`")
  lines.extend([
      "",
      "## Reading Guide",
      "",
      "- If `log_loop_x_tp_x_collective` is large and positive, the strongest evidence points to loop granularity interacting with TP collectives rather than chunk count alone.",
      "- If PyTorch eager rows show the same slope or outliers as JAX/XLA rows, XLA is not a necessary cause, although XLA may still amplify or hide the effect.",
      "- If matmul-only rows are flat while collective/projection+collective rows explode at high loop counts, the root cause is communication/control overhead rather than loss projection math alone.",
      "- If recovery rows show faster large chunks with similar HBM, chunk granularity is a performance knob independent of the memory-saving direction.",
      "",
  ])
  report_path.write_text("\n".join(lines) + "\n")
  return report_path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--results",
      nargs="+",
      required=True,
      help="JSONL result files or glob patterns, e.g. results/shard_*.jsonl.",
  )
  parser.add_argument("--manifest", type=Path, default=None)
  parser.add_argument(
      "--outdir",
      type=Path,
      default=Path(__file__).resolve().parent / "data" / "mesh_pathology_dense_sweep" / "analysis",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  outdir = args.outdir.expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  paths = expand_input_patterns(args.results)
  rows: list[dict[str, Any]] = []
  for path in paths:
    rows.extend(read_jsonl(path))
  data = normalize(rows)
  manifest_df = None
  if args.manifest:
    manifest_rows = read_jsonl(args.manifest.expanduser().resolve())
    manifest_df = pd.DataFrame(manifest_rows)

  if not data.empty:
    normalized_path = outdir / "normalized_results.csv"
    data.to_csv(normalized_path, index=False)

  success = success_rows(data)
  outputs: dict[str, Path | list[Path] | None] = {}
  outputs.update(coverage_tables(data, manifest_df, outdir))
  outputs["fastest"] = fastest_table(success, outdir)
  outputs["outliers"] = outlier_table(success, outdir)
  outputs["recovery"] = recovery_table(success, outdir)
  outputs["regression"] = regression_table(success, outdir)
  plots = []
  plots.extend(plot_factor_scatter(success, outdir))
  plots.extend(plot_heatmaps(success, outdir))
  outputs["plots"] = plots
  report_path = write_markdown_report(outdir=outdir, data=data, success=success, outputs=outputs)
  print(f"report={report_path}")
  print(f"rows_loaded={len(data)} success_rows={len(success)}")


if __name__ == "__main__":
  main()
