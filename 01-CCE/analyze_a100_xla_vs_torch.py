#!/usr/bin/env python3
"""Compare A100 JAX/XLA mesh-pathology rows against PyTorch rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
