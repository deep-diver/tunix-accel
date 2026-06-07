#!/usr/bin/env python3
"""Collect focused Gemma3 12B / 27B CCE boundary artifacts."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma3_12b_27b_cce_focused"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

MODEL_ORDER = ["Gemma3 12B", "Gemma3 27B"]
VARIANT_ORDER = ["default", "cce"]
VARIANT_LABEL = {"default": "Default CE", "cce": "CCE"}
VARIANT_COLOR = {"default": "#555555", "cce": "#159A78"}
HBM_LIMIT_GIB_PER_CHIP = 16.0


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def extract_artifacts() -> None:
  if RAW_DIR.exists():
    shutil.rmtree(RAW_DIR)
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    target = RAW_DIR / tarball.name.removesuffix(".tar.gz")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(target)], check=True)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
  for column in columns:
    if column in frame:
      frame[column] = pd.to_numeric(frame[column], errors="coerce")
  return frame


def read_results() -> pd.DataFrame:
  frames = []
  for path in sorted(RAW_DIR.glob("**/*_results.csv")):
    frame = pd.read_csv(path)
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()

  rows = pd.concat(frames, ignore_index=True, sort=False)
  rows = numeric(
      rows,
      [
          "batch_size",
          "chips",
          "final_loss",
          "lora_rank",
          "max_length",
          "max_steps",
          "mean_step_time_sec_excl_first",
          "mesh_fsdp",
          "mesh_tp",
          "runtime_hbm_headroom_gb",
          "runtime_hbm_limit_gb",
          "runtime_peak_hbm_gb",
          "steps_recorded",
          "token_chunk",
          "valid_tokens_per_sec_excl_first",
          "vocab_chunk",
          "wall_time_sec",
          "xla_train_step_gib_per_chip",
      ],
  )
  rows["ok"] = rows["status"].eq("ok")
  rows["model_order"] = rows["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  rows["variant_label"] = rows["variant"].map(VARIANT_LABEL).fillna(rows["variant"])
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}",
      axis=1,
  )
  rows = rows.drop_duplicates(
      subset=[
          "model",
          "suite",
          "variant",
          "batch_size",
          "max_length",
          "mesh_fsdp",
          "mesh_tp",
          "token_chunk",
          "vocab_chunk",
      ],
      keep="first",
  )
  return rows.sort_values(["model_order", "batch_size", "max_length", "variant"])


def build_manifest() -> pd.DataFrame:
  rows = []
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    rows.append({
        "artifact": str(tarball.relative_to(SCRIPT_DIR)),
        "bytes": tarball.stat().st_size,
        "profile": tarball.name.removesuffix(".tar.gz"),
        "project": "gcp-ml-172005",
        "zone": "us-west4-a",
        "tpu": "v5litepod-8",
        "chips": 8,
        "mesh": "fsdp8/tp1",
        "note": "single-host v5e boundary rerun",
    })
  return pd.DataFrame(rows)


def build_frontier_summary(rows: pd.DataFrame) -> pd.DataFrame:
  out = []
  for keys, part in rows.groupby(["model", "batch_size", "variant"], dropna=False):
    ok = part[part["ok"]]
    failed = part[~part["ok"]]
    out.append({
        "model": keys[0],
        "batch_size": int(keys[1]),
        "variant": keys[2],
        "max_ok_context": int(ok["max_length"].max()) if not ok.empty else 0,
        "first_failed_context": int(failed["max_length"].min()) if not failed.empty else "",
        "ok_rows": int(ok.shape[0]),
        "failed_rows": int(failed.shape[0]),
        "mesh": "fsdp8/tp1",
        "tpu": "v5litepod-8",
        "chips": 8,
    })
  summary = pd.DataFrame(out)
  if summary.empty:
    return summary
  summary["model_order"] = summary["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return summary.sort_values(["model_order", "batch_size", "variant"]).drop(columns="model_order")


def build_boundary_hbm(rows: pd.DataFrame) -> pd.DataFrame:
  points = []
  for keys, part in rows.groupby(["model", "batch_size"], dropna=False):
    lengths = set()
    for variant in VARIANT_ORDER:
      by_variant = part[part["variant"].eq(variant)]
      ok = by_variant[by_variant["ok"]]
      failed = by_variant[~by_variant["ok"]]
      if not ok.empty:
        lengths.add(int(ok["max_length"].max()))
      if not failed.empty:
        lengths.add(int(failed["max_length"].min()))
    for length in sorted(lengths):
      for variant in VARIANT_ORDER:
        row = part[(part["variant"].eq(variant)) & (part["max_length"].eq(length))]
        if row.empty:
          continue
        item = row.iloc[0]
        points.append({
            "model": keys[0],
            "batch_size": int(keys[1]),
            "max_length": int(length),
            "variant": variant,
            "status": item["status"],
            "ok": bool(item["ok"]),
            "xla_train_step_gib_per_chip": item["xla_train_step_gib_per_chip"],
            "mean_step_time_sec_excl_first": item["mean_step_time_sec_excl_first"],
            "mesh": item["mesh"],
            "tpu": item["tpu"],
            "chips": int(item["chips"]),
        })
  out = pd.DataFrame(points)
  if out.empty:
    return out
  out["model_order"] = out["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return out.sort_values(["model_order", "batch_size", "max_length", "variant"]).drop(
      columns="model_order"
  )


def build_matched_metrics(rows: pd.DataFrame) -> pd.DataFrame:
  metric_cols = [
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "final_loss",
  ]
  matched = rows.pivot_table(
      index=["model", "batch_size", "max_length", "mesh"],
      columns="variant",
      values=[col for col in metric_cols if col in rows],
      aggfunc="first",
  )
  if matched.empty:
    return pd.DataFrame()
  matched.columns = ["_".join(col).strip("_") for col in matched.columns.values]
  matched = matched.reset_index()
  if {
      "xla_train_step_gib_per_chip_default",
      "xla_train_step_gib_per_chip_cce",
  } <= set(matched.columns):
    matched["xla_reduction_gib_per_chip"] = (
        matched["xla_train_step_gib_per_chip_default"]
        - matched["xla_train_step_gib_per_chip_cce"]
    )
    matched["xla_reduction_pct"] = (
        1.0
        - matched["xla_train_step_gib_per_chip_cce"]
        / matched["xla_train_step_gib_per_chip_default"]
    ) * 100.0
  if {
      "mean_step_time_sec_excl_first_default",
      "mean_step_time_sec_excl_first_cce",
  } <= set(matched.columns):
    matched["step_time_multiplier_cce_vs_default"] = (
        matched["mean_step_time_sec_excl_first_cce"]
        / matched["mean_step_time_sec_excl_first_default"]
    )
  matched["model_order"] = matched["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return matched.sort_values(["model_order", "batch_size", "max_length"]).drop(
      columns="model_order"
  )


def context_label(value: float) -> str:
  value = int(value)
  if value <= 0:
    return "none"
  return f"{value // 1024}K" if value >= 1024 else str(value)


def context_height(value: float) -> int:
  value = int(value)
  if value <= 0:
    return 0
  return {512: 1, 1024: 2, 2048: 3, 4096: 4}.get(value, 0)


def plot_frontier(summary: pd.DataFrame) -> None:
  if summary.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True, sharey=True)
  fig.suptitle(
      "CCE boundary check on large Gemma3 LoRA jobs\n"
      "Cloud TPU v5litepod-8, 8 chips, fsdp=8/tp=1, LoRA rank 16, synthetic 2-step probe",
      fontsize=11,
      fontweight="bold",
  )
  width = 0.34
  for ax, model in zip(axes, MODEL_ORDER, strict=True):
    part = summary[summary["model"].eq(model)]
    batches = sorted(part["batch_size"].astype(int).unique())
    x_base = list(range(len(batches)))
    for idx, variant in enumerate(VARIANT_ORDER):
      values = []
      for batch in batches:
        row = part[(part["batch_size"].eq(batch)) & (part["variant"].eq(variant))]
        values.append(int(row.iloc[0]["max_ok_context"]) if not row.empty else 0)
      heights = [context_height(value) for value in values]
      bars = ax.bar(
          [x + (-0.5 + idx) * width for x in x_base],
          heights,
          width=width,
          color=VARIANT_COLOR[variant],
          label=VARIANT_LABEL[variant],
      )
      for bar, value, height in zip(bars, values, heights, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            context_label(value),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_title(model, fontsize=10, fontweight="bold")
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"b{batch}" for batch in batches])
    ax.set_ylim(-0.15, 4.55)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["none", "512", "1K", "2K", "4K"])
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E8E8E8", linewidth=0.8)
  axes[0].set_ylabel("Maximum completed context")
  axes[0].legend(frameon=False, fontsize=8, loc="upper right")
  fig.savefig(ASSET_DIR / "gemma3_12b_27b_cce_focused_frontier.png", dpi=190)
  plt.close(fig)


def plot_boundary_hbm(points: pd.DataFrame) -> None:
  if points.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.8), constrained_layout=True, sharey=True)
  fig.suptitle(
      "The large-model rerun saves HBM, but not enough to move the frontier\n"
      "Bars show only each batch's last passing and first failing context; XLA planned HBM is per TPU chip",
      fontsize=11,
      fontweight="bold",
  )
  width = 0.35
  for ax, model in zip(axes, MODEL_ORDER, strict=True):
    part = points[points["model"].eq(model)].copy()
    labels = []
    for batch, length in part[["batch_size", "max_length"]].drop_duplicates().itertuples(index=False):
      labels.append((int(batch), int(length)))
    x_base = list(range(len(labels)))
    for idx, variant in enumerate(VARIANT_ORDER):
      heights = []
      statuses = []
      for batch, length in labels:
        row = part[
            part["batch_size"].eq(batch)
            & part["max_length"].eq(length)
            & part["variant"].eq(variant)
        ]
        if row.empty:
          heights.append(float("nan"))
          statuses.append("")
        else:
          heights.append(float(row.iloc[0]["xla_train_step_gib_per_chip"]))
          statuses.append("OK" if bool(row.iloc[0]["ok"]) else "OOM")
      bars = ax.bar(
          [x + (-0.5 + idx) * width for x in x_base],
          heights,
          width=width,
          color=VARIANT_COLOR[variant],
          label=VARIANT_LABEL[variant],
      )
      for bar, status in zip(bars, statuses, strict=True):
        if not status or pd.isna(bar.get_height()):
          continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{bar.get_height():.1f}G\n{status}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            linespacing=0.95,
        )
    ax.axhline(HBM_LIMIT_GIB_PER_CHIP, color="#C84C4C", linewidth=1.0, linestyle="--")
    ax.text(
        0.012,
        HBM_LIMIT_GIB_PER_CHIP + 0.45,
        "16 GiB/chip fit line",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="#9F3333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
    )
    ax.set_title(model, fontsize=10, fontweight="bold")
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"b{batch}/L{length}" for batch, length in labels], fontsize=7.5)
    ax.set_ylabel("XLA planned HBM\n(GiB/chip)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E8E8E8", linewidth=0.8)
  axes[0].legend(frameon=False, fontsize=8, loc="upper left")
  max_value = points["xla_train_step_gib_per_chip"].dropna().max()
  axes[0].set_ylim(0, max(28, min(max_value + 4, 58)))
  fig.savefig(ASSET_DIR / "gemma3_12b_27b_cce_focused_hbm.png", dpi=190)
  plt.close(fig)


def main() -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  extract_artifacts()
  rows = read_results()
  frontier_summary = build_frontier_summary(rows)
  boundary_hbm = build_boundary_hbm(rows)
  matched = build_matched_metrics(rows)

  write_frame(build_manifest(), DATA_DIR / "run_manifest.csv")
  write_frame(rows, DATA_DIR / "frontier_runs.csv")
  write_frame(frontier_summary, DATA_DIR / "frontier_summary.csv")
  write_frame(boundary_hbm, DATA_DIR / "boundary_hbm_points.csv")
  write_frame(matched, DATA_DIR / "matched_metrics.csv")

  plot_frontier(frontier_summary)
  plot_boundary_hbm(boundary_hbm)

  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma3_12b_27b_cce_focused_frontier.png'}")
  print(f"figure={ASSET_DIR / 'gemma3_12b_27b_cce_focused_hbm.png'}")


if __name__ == "__main__":
  main()
