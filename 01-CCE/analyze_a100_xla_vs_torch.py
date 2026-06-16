#!/usr/bin/env python3
"""Compare A100 JAX/XLA mesh-pathology rows against PyTorch rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd


OPERATION_ORDER = [
    "chunked_matmul_loop",
    "collective_loop",
    "projection_collective_loop",
]
SIGNATURE_ORDER = [
    "bad",
    "good",
    "control-fsdp4-tp1",
    "control-fsdp1-tp4",
]
STACK_ORDER = ["JAX/XLA", "PyTorch eager", "torch.compile"]
STACK_COLORS = {
    "JAX/XLA": "#4C78A8",
    "PyTorch eager": "#F58518",
    "torch.compile": "#54A24B",
}
SIGNATURE_MARKERS = {
    "bad": "o",
    "good": "D",
    "control-fsdp4-tp1": "s",
    "control-fsdp1-tp4": "^",
}
SIGNATURE_LABELS = {
    "bad": "bad 2x2",
    "good": "good 2x2",
    "control-fsdp4-tp1": "ctrl 4x1",
    "control-fsdp1-tp4": "ctrl 1x4",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")


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


def normalize_rows(jax_rows: list[dict[str, Any]], torch_rows: list[dict[str, Any]]) -> pd.DataFrame:
  normalized: list[dict[str, Any]] = []
  for row in jax_rows:
    workload = row.get("workload_family", "")
    if row.get("hardware_target") != "gpu-a100-80gb-4" or workload not in OPERATION_ORDER:
      continue
    item = dict(row)
    item["operation_family"] = workload
    item["execution_stack"] = "JAX/XLA"
    item["execution_mode"] = "jit"
    normalized.append(item)
  for row in torch_rows:
    item = dict(row)
    mode = item.get("execution_mode", "")
    item["operation_family"] = item.get("operation_family") or item.get("workload_family", "")
    item["execution_stack"] = "torch.compile" if mode == "compile" else "PyTorch eager"
    normalized.append(item)
  data = pd.DataFrame(normalized)
  if data.empty:
    return data
  if "steady_state_mean_step_time_sec" not in data:
    data["steady_state_mean_step_time_sec"] = None
  data["steady_state_mean_step_time_sec"] = pd.to_numeric(
      data["steady_state_mean_step_time_sec"], errors="coerce"
  )
  data["status"] = data.get("status", "").fillna("")
  data["signature_label"] = pd.Categorical(
      data["signature_label"], categories=SIGNATURE_ORDER, ordered=True
  )
  data["operation_family"] = pd.Categorical(
      data["operation_family"], categories=OPERATION_ORDER, ordered=True
  )
  data["execution_stack"] = pd.Categorical(
      data["execution_stack"], categories=STACK_ORDER, ordered=True
  )
  return data.sort_values(["operation_family", "execution_stack", "signature_label"])


def make_bad_good_ratios(data: pd.DataFrame) -> pd.DataFrame:
  success = data[
      data["status"].eq("success") & data["steady_state_mean_step_time_sec"].notna()
  ].copy()
  if success.empty:
    return pd.DataFrame()
  pivot = success.pivot_table(
      index=["operation_family", "execution_stack"],
      columns="signature_label",
      values="steady_state_mean_step_time_sec",
      aggfunc="mean",
      observed=False,
  ).reset_index()
  if "bad" not in pivot:
    pivot["bad"] = pd.NA
  if "good" not in pivot:
    pivot["good"] = pd.NA
  pivot["bad_good_slowdown"] = pivot["bad"] / pivot["good"]
  return pivot.sort_values(["operation_family", "execution_stack"])


def make_failure_summary(data: pd.DataFrame) -> pd.DataFrame:
  failed = data[~data["status"].eq("success")].copy()
  if failed.empty:
    return pd.DataFrame(columns=["operation_family", "execution_stack", "signature_label", "status"])
  cols = [
      "experiment_id",
      "operation_family",
      "execution_stack",
      "signature_label",
      "status",
      "failure_type",
      "error_message",
  ]
  for col in cols:
    if col not in failed:
      failed[col] = ""
  return failed[cols].sort_values(["operation_family", "execution_stack", "signature_label"])


def plot_step_times(data: pd.DataFrame, outdir: Path) -> Path:
  success = data[
      data["status"].eq("success") & data["steady_state_mean_step_time_sec"].notna()
  ].copy()
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
  for ax, operation in zip(axes, OPERATION_ORDER, strict=True):
    part = success[success["operation_family"].astype(str).eq(operation)]
    x_positions = range(len(SIGNATURE_ORDER))
    width = 0.24
    for stack_index, stack in enumerate(STACK_ORDER):
      stack_part = part[part["execution_stack"].astype(str).eq(stack)]
      values = []
      for signature in SIGNATURE_ORDER:
        value = stack_part[
            stack_part["signature_label"].astype(str).eq(signature)
        ]["steady_state_mean_step_time_sec"]
        values.append(float(value.iloc[0]) if not value.empty else float("nan"))
      offsets = [x + (stack_index - 1) * width for x in x_positions]
      ax.bar(
          offsets,
          values,
          width=width,
          label=stack,
          color=STACK_COLORS[stack],
          alpha=0.9,
      )
    ax.set_title(operation.replace("_", " "))
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(
        ["bad\n2x2", "good\n2x2", "ctrl\n4x1", "ctrl\n1x4"],
        fontsize=9,
    )
    ax.set_ylabel("step time (s)")
    ax.grid(axis="y", alpha=0.25)
  axes[0].legend(loc="upper left", fontsize=9)
  fig.suptitle("A100: step time by operation, mesh/chunk signature, and stack")
  fig.tight_layout()
  path = outdir / "a100_step_time_by_operation_stack.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  return path


def plot_bad_good(ratios: pd.DataFrame, outdir: Path) -> Path:
  fig, ax = plt.subplots(figsize=(10, 4.8))
  x_positions = range(len(OPERATION_ORDER))
  width = 0.24
  for stack_index, stack in enumerate(STACK_ORDER):
    part = ratios[ratios["execution_stack"].astype(str).eq(stack)]
    values = []
    for operation in OPERATION_ORDER:
      value = part[part["operation_family"].astype(str).eq(operation)]["bad_good_slowdown"]
      values.append(float(value.iloc[0]) if not value.empty and pd.notna(value.iloc[0]) else float("nan"))
    offsets = [x + (stack_index - 1) * width for x in x_positions]
    ax.bar(offsets, values, width=width, label=stack, color=STACK_COLORS[stack], alpha=0.9)
  ax.axhline(1.0, color="#333333", linewidth=1)
  ax.set_xticks(list(x_positions))
  ax.set_xticklabels([op.replace("_", "\n") for op in OPERATION_ORDER])
  ax.set_ylabel("bad/good step-time ratio")
  ax.set_title("A100: small-chunk slowdown inside each operation")
  ax.grid(axis="y", alpha=0.25)
  ax.legend(loc="upper left")
  fig.tight_layout()
  path = outdir / "a100_bad_good_slowdown_xla_vs_torch.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  return path


def plot_normalized_heatmap(data: pd.DataFrame, outdir: Path) -> Path:
  success = data[
      data["status"].eq("success") & data["steady_state_mean_step_time_sec"].notna()
  ].copy()
  labels = []
  rows = []
  for operation in OPERATION_ORDER:
    for stack in STACK_ORDER:
      part = success[
          success["operation_family"].astype(str).eq(operation)
          & success["execution_stack"].astype(str).eq(stack)
      ]
      if part.empty:
        continue
      good = part[part["signature_label"].astype(str).eq("good")][
          "steady_state_mean_step_time_sec"
      ]
      if good.empty or float(good.iloc[0]) == 0:
        continue
      labels.append(f"{operation.replace('_loop', '').replace('_', ' ')}\n{stack}")
      rows.append([
          float(
              part[part["signature_label"].astype(str).eq(signature)][
                  "steady_state_mean_step_time_sec"
              ].iloc[0]
          )
          / float(good.iloc[0])
          if not part[part["signature_label"].astype(str).eq(signature)].empty
          else float("nan")
          for signature in SIGNATURE_ORDER
      ])
  fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.55 * len(rows))))
  image = ax.imshow(rows, aspect="auto", cmap="YlOrRd", vmin=0.0)
  ax.set_xticks(range(len(SIGNATURE_ORDER)))
  ax.set_xticklabels(["bad 2x2", "good 2x2", "ctrl 4x1", "ctrl 1x4"])
  ax.set_yticks(range(len(labels)))
  ax.set_yticklabels(labels)
  for y, row in enumerate(rows):
    for x, value in enumerate(row):
      if pd.notna(value):
        ax.text(x, y, f"{value:.1f}x", ha="center", va="center", fontsize=9)
  ax.set_title("A100: normalized step time, each row divided by its good 2x2 case")
  fig.colorbar(image, ax=ax, label="relative step time")
  fig.tight_layout()
  path = outdir / "a100_normalized_signature_heatmap_xla_vs_torch.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  return path


def successful_rows_with_relative_time(data: pd.DataFrame) -> pd.DataFrame:
  success = data[
      data["status"].eq("success") & data["steady_state_mean_step_time_sec"].notna()
  ].copy()
  success["relative_step_time"] = np.nan
  for stack in STACK_ORDER:
    for operation in OPERATION_ORDER:
      mask = (
          success["execution_stack"].astype(str).eq(stack)
          & success["operation_family"].astype(str).eq(operation)
      )
      good = success[mask & success["signature_label"].astype(str).eq("good")][
          "steady_state_mean_step_time_sec"
      ]
      if good.empty or float(good.iloc[0]) == 0:
        continue
      success.loc[mask, "relative_step_time"] = (
          success.loc[mask, "steady_state_mean_step_time_sec"] / float(good.iloc[0])
      )
  return success


def add_scatter_legends(ax: plt.Axes) -> None:
  stack_handles = [
      plt.Line2D(
          [0],
          [0],
          marker="o",
          linestyle="",
          markerfacecolor=STACK_COLORS[stack],
          markeredgecolor="#222222",
          label=stack,
          markersize=8,
      )
      for stack in STACK_ORDER
      if stack != "torch.compile"
  ]
  signature_handles = [
      plt.Line2D(
          [0],
          [0],
          marker=SIGNATURE_MARKERS[signature],
          linestyle="",
          markerfacecolor="white",
          markeredgecolor="#222222",
          label=SIGNATURE_LABELS[signature],
          markersize=8,
      )
      for signature in SIGNATURE_ORDER
  ]
  first = ax.legend(handles=stack_handles, loc="upper left", fontsize=8, title="stack")
  ax.add_artist(first)
  ax.legend(
      handles=signature_handles,
      loc="lower right",
      fontsize=8,
      title="signature",
  )


def plot_scatter(
    data: pd.DataFrame,
    outdir: Path,
    *,
    y_column: str,
    ylabel: str,
    title: str,
    filename: str,
    y_log: bool,
) -> Path:
  success = successful_rows_with_relative_time(data)
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharey=False)
  for ax, operation in zip(axes, OPERATION_ORDER, strict=True):
    part = success[success["operation_family"].astype(str).eq(operation)]
    for stack in STACK_ORDER:
      if stack == "torch.compile":
        continue
      for signature in SIGNATURE_ORDER:
        row = part[
            part["execution_stack"].astype(str).eq(stack)
            & part["signature_label"].astype(str).eq(signature)
        ]
        if row.empty or pd.isna(row[y_column].iloc[0]):
          continue
        ax.scatter(
            float(row["chunk_loop_count"].iloc[0]),
            float(row[y_column].iloc[0]),
            s=110,
            marker=SIGNATURE_MARKERS[signature],
            color=STACK_COLORS[stack],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.92,
        )
    ax.set_title(operation.replace("_", " "))
    ax.set_xscale("log", base=2)
    if y_log:
      ax.set_yscale("log", base=2)
    if y_column == "relative_step_time":
      ax.axhline(1.0, color="#333333", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("chunk_loop_count")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.22)
  add_scatter_legends(axes[0])
  fig.suptitle(title)
  fig.tight_layout()
  path = outdir / filename
  fig.savefig(path, dpi=180)
  plt.close(fig)
  return path


def plot_scatter_views(data: pd.DataFrame, outdir: Path) -> list[Path]:
  return [
      plot_scatter(
          data,
          outdir,
          y_column="steady_state_mean_step_time_sec",
          ylabel="step time (s)",
          title="A100: step time vs chunk-loop count",
          filename="a100_scatter_chunk_loop_vs_step_time.png",
          y_log=False,
      ),
      plot_scatter(
          data,
          outdir,
          y_column="relative_step_time",
          ylabel="relative step time",
          title="A100: normalized slowdown vs chunk-loop count",
          filename="a100_scatter_chunk_loop_vs_relative_time.png",
          y_log=False,
      ),
  ]


def normalized_value_grid(data: pd.DataFrame) -> tuple[np.ndarray, dict[tuple[str, str, str], str]]:
  success = data[
      data["status"].eq("success") & data["steady_state_mean_step_time_sec"].notna()
  ].copy()
  values = np.full((len(STACK_ORDER), len(OPERATION_ORDER), len(SIGNATURE_ORDER)), np.nan)
  failure_labels: dict[tuple[str, str, str], str] = {}
  for _, row in data[~data["status"].eq("success")].iterrows():
    failure_labels[(
        str(row.get("execution_stack", "")),
        str(row.get("operation_family", "")),
        str(row.get("signature_label", "")),
    )] = "fail"
  for stack_index, stack in enumerate(STACK_ORDER):
    for operation_index, operation in enumerate(OPERATION_ORDER):
      part = success[
          success["execution_stack"].astype(str).eq(stack)
          & success["operation_family"].astype(str).eq(operation)
      ]
      good = part[part["signature_label"].astype(str).eq("good")][
          "steady_state_mean_step_time_sec"
      ]
      if good.empty or float(good.iloc[0]) == 0:
        continue
      denominator = float(good.iloc[0])
      for signature_index, signature in enumerate(SIGNATURE_ORDER):
        value = part[part["signature_label"].astype(str).eq(signature)][
            "steady_state_mean_step_time_sec"
        ]
        if not value.empty:
          values[stack_index, operation_index, signature_index] = float(value.iloc[0]) / denominator
  return values, failure_labels


def plot_layered_3d_heatmap(data: pd.DataFrame, outdir: Path) -> Path:
  values, failure_labels = normalized_value_grid(data)
  finite = values[np.isfinite(values)]
  vmax = max(1.0, float(np.nanmax(finite))) if finite.size else 1.0
  norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
  cmap = plt.get_cmap("YlOrRd")

  fig = plt.figure(figsize=(11.5, 7.5))
  ax = fig.add_subplot(111, projection="3d")
  ax.set_box_aspect((1.45, 1.0, 0.75))

  for stack_index, stack in enumerate(STACK_ORDER):
    z = stack_index * 1.15
    for operation_index, operation in enumerate(OPERATION_ORDER):
      for signature_index, signature in enumerate(SIGNATURE_ORDER):
        value = values[stack_index, operation_index, signature_index]
        x0, x1 = signature_index - 0.44, signature_index + 0.44
        y0, y1 = operation_index - 0.38, operation_index + 0.38
        verts = [[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]]
        if np.isfinite(value):
          facecolor = cmap(norm(value))
          label = f"{value:.1f}x"
          edgecolor = "#3A3A3A"
          alpha = 0.88
        else:
          state = failure_labels.get((stack, operation, signature), "n/a")
          facecolor = (0.58, 0.58, 0.58, 0.34)
          label = state
          edgecolor = "#777777"
          alpha = 0.42
        collection = Poly3DCollection(
            verts,
            facecolors=[facecolor],
            edgecolors=edgecolor,
            linewidths=0.75,
            alpha=alpha,
        )
        ax.add_collection3d(collection)
        text_color = "#111111" if np.isfinite(value) and value < vmax * 0.72 else "#F8F8F8"
        if not np.isfinite(value):
          text_color = "#333333"
        ax.text(
            signature_index,
            operation_index,
            z + 0.035,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
            zorder=10,
        )
  ax.set_xlim(-0.6, len(SIGNATURE_ORDER) - 0.4)
  ax.set_ylim(-0.55, len(OPERATION_ORDER) - 0.15)
  ax.set_zlim(-0.2, (len(STACK_ORDER) - 1) * 1.15 + 0.35)
  ax.set_xticks(range(len(SIGNATURE_ORDER)))
  ax.set_xticklabels(["bad\n2x2", "good\n2x2", "ctrl\n4x1", "ctrl\n1x4"], fontsize=9)
  ax.set_yticks(range(len(OPERATION_ORDER)))
  ax.set_yticklabels([
      "chunked\nmatmul",
      "collective",
      "projection\n+ collective",
  ], fontsize=9)
  ax.set_zticks([index * 1.15 for index in range(len(STACK_ORDER))])
  ax.set_zticklabels(STACK_ORDER, fontsize=9)
  ax.set_xlabel("mesh/chunk signature", labelpad=12)
  ax.set_ylabel("operation", labelpad=12)
  ax.set_zlabel("execution stack", labelpad=10)
  ax.set_title("A100 layered 3D heatmap: normalized step time by stack", pad=18)
  ax.view_init(elev=24, azim=-52)
  ax.grid(True, alpha=0.22)

  scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
  scalar.set_array([])
  colorbar = fig.colorbar(scalar, ax=ax, shrink=0.62, pad=0.08)
  colorbar.set_label("relative step time, divided by good 2x2 within each stack/operation")
  fig.subplots_adjust(left=0.02, right=0.86, top=0.92, bottom=0.02)
  path = outdir / "a100_layered_3d_heatmap_xla_vs_torch.png"
  fig.savefig(path, dpi=190)
  plt.close(fig)
  return path


def write_report(
    *,
    outdir: Path,
    data: pd.DataFrame,
    ratios: pd.DataFrame,
    failures: pd.DataFrame,
    plots: list[Path],
) -> Path:
  successful = int(data["status"].eq("success").sum()) if not data.empty else 0
  total = int(len(data))
  lines = [
      "# A100 XLA vs PyTorch Mesh Pathology",
      "",
      "This report compares the same A100 4-GPU synthetic mesh-pathology rows across JAX/XLA, PyTorch eager, and torch.compile/Inductor. The PyTorch eager rows are XLA-free CUDA/NCCL workloads.",
      "",
      f"- Rows loaded: {total}",
      f"- Successful timing rows: {successful}",
      f"- Failed or aborted rows: {total - successful}",
      "",
      "## Bad/Good Slowdown",
      "",
      ratios.to_markdown(index=False) if not ratios.empty else "No successful bad/good pairs.",
      "",
      "## Failure Summary",
      "",
      failures.to_markdown(index=False) if not failures.empty else "No failures.",
      "",
      "## Plots",
      "",
  ]
  for path in plots:
    lines.append(f"- `{path.relative_to(outdir.parent.parent) if outdir.parent.parent in path.parents else path}`")
  lines.extend([
      "",
      "## Most Likely Interpretation",
      "",
      "PyTorch eager reproduces a strong small-chunk slowdown on A100 without XLA. That means XLA is not required for the phenomenon. The likely base mechanism is the granularity of many small loop bodies and, for collective workloads, the interaction with TP communication shape.",
      "",
      "This does not prove XLA is irrelevant. JAX/XLA can still change the magnitude through lowering, fusion, and scheduling. In this run torch.compile did not produce usable timing rows for the compiled variants, so compiler-backed CUDA requires a separate smaller reproduction path.",
  ])
  path = outdir / "A100_XLA_VS_TORCH_MESH_PATHOLOGY_REPORT.md"
  path.write_text("\n".join(lines) + "\n")
  return path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--jax-results", type=Path, required=True)
  parser.add_argument("--torch-results", type=Path, required=True)
  parser.add_argument("--outdir", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  outdir = args.outdir.expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  data = normalize_rows(read_jsonl(args.jax_results), read_jsonl(args.torch_results))
  rows = data.to_dict(orient="records") if not data.empty else []
  write_jsonl(outdir / "combined_results.jsonl", rows)
  write_csv(outdir / "combined_results.csv", rows)

  ratios = make_bad_good_ratios(data)
  ratios.to_csv(outdir / "bad_good_ratios.csv", index=False)
  failures = make_failure_summary(data)
  failures.to_csv(outdir / "failure_summary.csv", index=False)

  plots = [
      plot_step_times(data, outdir),
      plot_bad_good(ratios, outdir),
      plot_normalized_heatmap(data, outdir),
      *plot_scatter_views(data, outdir),
      plot_layered_3d_heatmap(data, outdir),
  ]
  report = write_report(outdir=outdir, data=data, ratios=ratios, failures=failures, plots=plots)
  print(f"combined_results={outdir / 'combined_results.jsonl'}")
  print(f"bad_good_ratios={outdir / 'bad_good_ratios.csv'}")
  print(f"failure_summary={outdir / 'failure_summary.csv'}")
  for plot in plots:
    print(f"plot={plot}")
  print(f"report={report}")


if __name__ == "__main__":
  main()
