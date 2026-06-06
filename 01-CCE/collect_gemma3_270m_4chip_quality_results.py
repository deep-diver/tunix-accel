#!/usr/bin/env python3
"""Collect Gemma3 270M four-chip OPUS100 training parity artifacts."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma3_270m_4chip_quality"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

VARIANT_COLOR = {"default": "#555555", "cce": "#159A78"}
VARIANT_LABEL = {"default": "Default CE", "cce": "CCE"}


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def extract_artifacts() -> None:
  if RAW_DIR.exists():
    shutil.rmtree(RAW_DIR)
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    target = RAW_DIR / tarball.name.removesuffix(".tar.gz")
    if target.exists():
      shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(target)], check=True)


def read_results() -> pd.DataFrame:
  frames = []
  for path in sorted(RAW_DIR.glob("**/*_results.csv")):
    frame = pd.read_csv(path)
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  rows = pd.concat(frames, ignore_index=True, sort=False)
  for col in [
      "batch_size",
      "max_length",
      "chips",
      "mesh_fsdp",
      "mesh_tp",
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "final_loss",
      "eval_loss",
      "wall_time_sec",
      "steps_recorded",
  ]:
    if col in rows:
      rows[col] = pd.to_numeric(rows[col], errors="coerce")
  rows["ok"] = rows["status"].eq("ok")
  rows["variant_label"] = rows["variant"].map(VARIANT_LABEL).fillna(rows["variant"])
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}",
      axis=1,
  )
  return rows.sort_values(["variant", "batch_size", "max_length"])


def case_history_path(case: str) -> Path | None:
  matches = sorted(RAW_DIR.glob(f"**/{case}/history.csv"))
  return matches[0] if matches else None


def read_history(rows: pd.DataFrame) -> pd.DataFrame:
  frames = []
  for row in rows.itertuples(index=False):
    path = case_history_path(row.case)
    if path is None:
      continue
    frame = pd.read_csv(path)
    frame["case"] = row.case
    frame["variant"] = row.variant
    frame["variant_label"] = VARIANT_LABEL.get(row.variant, row.variant)
    frame["batch_size"] = row.batch_size
    frame["max_length"] = row.max_length
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  out = pd.concat(frames, ignore_index=True, sort=False)
  for col in ["step", "loss", "step_time_sec", "cumulative_loss_tokens"]:
    if col in out:
      out[col] = pd.to_numeric(out[col], errors="coerce")
  return out.sort_values(["variant", "step"])


def plot_quality(rows: pd.DataFrame, history: pd.DataFrame) -> None:
  if rows.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(
      2,
      1,
      figsize=(9.2, 7.0),
      gridspec_kw={"height_ratios": [1.15, 1.0]},
      constrained_layout=True,
  )
  fig.suptitle(
      "Gemma3 270M OPUS100 parity on TPU v5litepod-4 (4 chips, fsdp=4/tp=1)",
      fontsize=11.5,
      fontweight="bold",
  )

  ax = axes[0]
  if not history.empty:
    for variant, part in history.groupby("variant"):
      part = part.sort_values("step")
      ax.plot(
          part["step"],
          part["loss"],
          color=VARIANT_COLOR.get(variant, "#555555"),
          alpha=0.28,
          linewidth=0.8,
      )
      smooth = part["loss"].rolling(25, min_periods=1).mean()
      ax.plot(
          part["step"],
          smooth,
          color=VARIANT_COLOR.get(variant, "#555555"),
          linewidth=2.0,
          label=VARIANT_LABEL.get(variant, variant),
      )
  ax.set_xlabel("Training step")
  ax.set_ylabel("Train loss")
  ax.set_title("Loss trajectory")
  ax.set_axisbelow(True)
  ax.grid(True, color="#E6E6E6")
  ax.legend(frameon=False)

  ax = axes[1]
  metrics = [
      ("xla_train_step_gib_per_chip", "XLA HBM\nGiB/chip"),
      ("mean_step_time_sec_excl_first", "Step time\nsec"),
      ("final_loss", "Final\ntrain loss"),
      ("eval_loss", "Eval\nloss"),
  ]
  ok = rows[rows["ok"]].copy()
  x_base = range(len(metrics))
  width = 0.34
  for idx, variant in enumerate(["default", "cce"]):
    part = ok[ok["variant"].eq(variant)]
    if part.empty:
      continue
    row = part.iloc[0]
    values = []
    labels = []
    for col, _ in metrics:
      value = row.get(col)
      values.append(float(value) if pd.notna(value) else 0.0)
      if col == "xla_train_step_gib_per_chip":
        labels.append(f"{value:.2f}GiB")
      elif col == "mean_step_time_sec_excl_first":
        labels.append(f"{value:.3f}s")
      elif col in {"final_loss", "eval_loss"}:
        labels.append(f"{value:.4f}")
      else:
        labels.append(str(value))
    normalized = []
    for metric_idx, (col, _) in enumerate(metrics):
      baseline = ok[ok["variant"].eq("default")][col]
      denom = float(baseline.iloc[0]) if not baseline.empty and pd.notna(baseline.iloc[0]) else values[metric_idx]
      normalized.append(values[metric_idx] / denom if denom else 0.0)
    bars = ax.bar(
        [x + (-0.5 + idx) * width for x in x_base],
        normalized,
        width=width,
        color=VARIANT_COLOR.get(variant, "#555555"),
        label=VARIANT_LABEL.get(variant, variant),
    )
    for bar, label in zip(bars, labels, strict=True):
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 0.035,
          label,
          ha="center",
          va="bottom",
          fontsize=8,
      )
  ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
  ax.set_xticks(list(x_base))
  ax.set_xticklabels([label for _, label in metrics])
  ax.set_ylabel("Normalized to Default CE")
  ax.set_title("Same-shape training metrics")
  ax.set_axisbelow(True)
  ax.grid(True, axis="y", color="#E6E6E6")
  ax.legend(frameon=False)

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_4chip_quality.png", dpi=180)
  plt.close(fig)


def build_manifest() -> pd.DataFrame:
  rows = []
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    rows.append({
        "artifact": str(tarball.relative_to(SCRIPT_DIR)),
        "bytes": tarball.stat().st_size,
        "profile": tarball.name.removesuffix(".tar.gz"),
        "tpu": "v5litepod-4",
        "chips": 4,
        "project": "gcp-ml-172005",
        "zone": "us-west4-a",
        "model": "google/gemma-3-270m-it",
    })
  return pd.DataFrame(rows)


def main() -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  extract_artifacts()
  rows = read_results()
  write_frame(build_manifest(), DATA_DIR / "run_manifest.csv")
  if rows.empty:
    print("No four-chip quality results found")
    return
  history = read_history(rows)
  write_frame(rows, DATA_DIR / "training_summary.csv")
  write_frame(history, DATA_DIR / "training_history.csv")
  plot_quality(rows, history)
  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma3_270m_cce_4chip_quality.png'}")


if __name__ == "__main__":
  main()
