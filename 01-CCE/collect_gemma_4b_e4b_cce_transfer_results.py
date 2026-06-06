#!/usr/bin/env python3
"""Collect focused Gemma3 4B / Gemma4 E4B CCE transfer artifacts."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma_4b_e4b_cce_transfer"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

MODEL_ORDER = ["Gemma3 4B", "Gemma4 E4B"]
MODEL_LABEL = {"Gemma3 4B": "Gemma3 4B", "Gemma4 E4B": "Gemma4 E4B"}
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
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(target)], check=True)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
  for col in columns:
    if col in frame:
      frame[col] = pd.to_numeric(frame[col], errors="coerce")
  return frame


def add_common_columns(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return rows
  numeric(
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
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}"
      if pd.notna(row.get("mesh_fsdp")) and pd.notna(row.get("mesh_tp"))
      else "",
      axis=1,
  )
  rows["variant_label"] = rows["variant"].map(VARIANT_LABEL).fillna(rows["variant"])
  rows["model_order"] = rows["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return rows


def read_results() -> pd.DataFrame:
  frames = []
  for path in sorted(RAW_DIR.glob("**/*_results.csv")):
    frame = pd.read_csv(path)
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  rows = pd.concat(frames, ignore_index=True, sort=False)
  rows = add_common_columns(rows)
  dedupe = [
      "model",
      "suite",
      "case",
      "variant",
      "batch_size",
      "max_length",
      "mesh_fsdp",
      "mesh_tp",
      "token_chunk",
      "vocab_chunk",
  ]
  rows = rows.drop_duplicates(subset=[col for col in dedupe if col in rows], keep="first")
  return rows.sort_values([
      "model_order",
      "suite",
      "batch_size",
      "max_length",
      "variant",
      "token_chunk",
      "vocab_chunk",
  ])


def build_manifest() -> pd.DataFrame:
  rows = []
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    rows.append({
        "artifact": str(tarball.relative_to(SCRIPT_DIR)),
        "bytes": tarball.stat().st_size,
        "profile": tarball.name.removesuffix(".tar.gz"),
        "tpu": "v5litepod-8",
        "chips": 8,
        "project": "gcp-ml-172005",
        "zone": "us-west4-a",
    })
  return pd.DataFrame(rows)


def build_frontier_summary(rows: pd.DataFrame) -> pd.DataFrame:
  frontier = rows[rows["suite"].fillna("").str.contains("frontier")].copy()
  out = []
  for keys, part in frontier.groupby(["model", "batch_size", "variant"], dropna=False):
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


def build_matched_metrics(rows: pd.DataFrame) -> pd.DataFrame:
  metric_cols = [
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "final_loss",
  ]
  ok = rows[rows["ok"]].copy()
  matched = ok.pivot_table(
      index=["model", "suite", "mesh", "batch_size", "max_length"],
      columns="variant",
      values=[col for col in metric_cols if col in ok],
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
  return matched.sort_values(["model_order", "suite", "batch_size", "max_length"]).drop(
      columns="model_order"
  )


def build_pressure_points(rows: pd.DataFrame) -> pd.DataFrame:
  frontier = rows[rows["suite"].fillna("").str.contains("frontier")].copy()
  pairs = []
  keys = ["model", "batch_size", "max_length", "mesh"]
  for key, part in frontier.groupby(keys, dropna=False):
    by_variant = {row.variant: row for row in part.itertuples(index=False)}
    if "default" not in by_variant or "cce" not in by_variant:
      continue
    default = by_variant["default"]
    cce = by_variant["cce"]
    if bool(default.ok) or not bool(cce.ok):
      continue
    pairs.append({
        "model": key[0],
        "batch_size": int(key[1]),
        "max_length": int(key[2]),
        "mesh": key[3],
        "default_status": default.status,
        "cce_status": cce.status,
        "default_xla_gib_per_chip": default.xla_train_step_gib_per_chip,
        "cce_xla_gib_per_chip": cce.xla_train_step_gib_per_chip,
        "cce_step_time_sec": cce.mean_step_time_sec_excl_first,
        "default_final_loss": getattr(default, "final_loss", math.nan),
        "cce_final_loss": getattr(cce, "final_loss", math.nan),
        "tpu": "v5litepod-8",
        "chips": 8,
    })
  pressure = pd.DataFrame(pairs)
  if pressure.empty:
    return pressure
  pressure["model_order"] = pressure["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return pressure.sort_values(["model_order", "batch_size", "max_length"]).drop(columns="model_order")


def build_chunk_summary(rows: pd.DataFrame) -> pd.DataFrame:
  chunk = rows[rows["suite"].fillna("").str.contains("chunk") & rows["ok"]].copy()
  if chunk.empty:
    return chunk
  out = []
  keys = ["model", "mesh", "batch_size", "max_length"]
  for key, part in chunk.groupby(keys, dropna=False):
    conservative = part[(part["token_chunk"].eq(128)) & (part["vocab_chunk"].eq(8192))]
    best = part.sort_values("mean_step_time_sec_excl_first").head(1)
    if conservative.empty:
      conservative = best
    for label, frame in [("128/8192", conservative.head(1)), ("fastest tested", best)]:
      row = frame.iloc[0]
      out.append({
          "model": key[0],
          "mesh": key[1],
          "batch_size": int(key[2]),
          "max_length": int(key[3]),
          "selection": label,
          "token_chunk": int(row["token_chunk"]),
          "vocab_chunk": int(row["vocab_chunk"]),
          "mean_step_time_sec_excl_first": row["mean_step_time_sec_excl_first"],
          "xla_train_step_gib_per_chip": row["xla_train_step_gib_per_chip"],
          "valid_tokens_per_sec_excl_first": row["valid_tokens_per_sec_excl_first"],
      })
  summary = pd.DataFrame(out)
  summary["model_order"] = summary["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return summary.sort_values(["model_order", "batch_size", "max_length", "selection"]).drop(
      columns="model_order"
  )


def context_rank(value: float) -> int:
  value = int(value)
  if value <= 0:
    return 0
  return {256: 1, 512: 2, 1024: 3, 2048: 4, 4096: 5, 8192: 6}.get(
      value, int(math.log2(value // 256)) + 1
  )


def context_label(value: float) -> str:
  value = int(value)
  if value <= 0:
    return "none"
  return f"{value // 1024}K" if value >= 1024 else str(value)


def plot_frontier(summary: pd.DataFrame) -> None:
  if summary.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(
      len(MODEL_ORDER),
      1,
      figsize=(9.0, 5.9),
      constrained_layout=True,
      sharex=False,
  )
  if len(MODEL_ORDER) == 1:
    axes = [axes]
  fig.suptitle(
      "CCE extends the larger-Gemma LoRA fit frontier\n"
      "Cloud TPU v5litepod-8, 8 chips, fsdp=8/tp=1, LoRA rank 16, synthetic 2-step probe",
      fontsize=11.4,
      fontweight="bold",
  )
  width = 0.35
  for ax, model in zip(axes, MODEL_ORDER, strict=True):
    part = summary[summary["model"].eq(model)].copy()
    batches = sorted(part["batch_size"].dropna().astype(int).unique())
    x_base = list(range(len(batches)))
    for idx, variant in enumerate(["default", "cce"]):
      variant_part = part[part["variant"].eq(variant)].set_index("batch_size")
      values = [
          variant_part.loc[batch, "max_ok_context"] if batch in variant_part.index else 0
          for batch in batches
      ]
      heights = [context_rank(value) for value in values]
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
            fontsize=7.2,
        )
    ax.set_ylim(-0.2, 5.45)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["none", "256", "512", "1K", "2K", "4K"])
    ax.set_ylabel(f"{MODEL_LABEL[model]}\nmax context")
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"b{batch}" for batch in batches], fontsize=8.4)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E6E6E6")
    if model == MODEL_ORDER[0]:
      ax.legend(frameon=False, loc="upper right", fontsize=8)
  axes[-1].set_xlabel("Batch size")
  fig.savefig(ASSET_DIR / "gemma_cce_large_transfer_frontier.png", dpi=190)
  plt.close(fig)


def plot_pressure(pressure: pd.DataFrame) -> None:
  if pressure.empty:
    return
  pressure = pressure.copy()
  pressure["label"] = pressure.apply(
      lambda row: (
          f"{MODEL_LABEL.get(row['model'], row['model'])}\n"
          f"b{int(row['batch_size'])}/L{int(row['max_length'])}"
      ),
      axis=1,
  )
  x = list(range(len(pressure)))
  width = 0.34
  fig, ax = plt.subplots(figsize=(8.9, 4.9), constrained_layout=True)
  fig.suptitle(
      "CCE-only passing pressure points on the same 8-chip setup\n"
      "XLA planned HBM per chip; Default CE bar is the failed compile plan",
      fontsize=11.2,
      fontweight="bold",
  )
  values = {
      "default": pressure["default_xla_gib_per_chip"].tolist(),
      "cce": pressure["cce_xla_gib_per_chip"].tolist(),
  }
  for idx, variant in enumerate(["default", "cce"]):
    bars = ax.bar(
        [i + (-0.5 + idx) * width for i in x],
        values[variant],
        width=width,
        color=VARIANT_COLOR[variant],
        label=VARIANT_LABEL[variant],
      )
    for bar in bars:
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 0.35,
          f"{bar.get_height():.1f}G",
          ha="center",
          va="bottom",
          fontsize=7.4,
      )
  ax.axhline(16.0, color="#C84C4C", linewidth=1.1, linestyle="--")
  ax.text(
      len(pressure) - 0.35,
      16.25,
      "16 GiB/chip nominal HBM",
      color="#9F3333",
      fontsize=7.6,
      ha="right",
      va="bottom",
  )
  ax.set_xticks(x)
  ax.set_xticklabels(pressure["label"], fontsize=8.2)
  ax.set_ylabel("XLA planned HBM (GiB/chip)")
  ax.set_axisbelow(True)
  ax.grid(True, axis="y", color="#E6E6E6")
  ax.legend(frameon=False, fontsize=8, loc="upper left")
  fig.savefig(ASSET_DIR / "gemma_cce_large_transfer_pressure.png", dpi=190)
  plt.close(fig)


def plot_chunk_heatmaps(rows: pd.DataFrame) -> None:
  chunk = rows[rows["suite"].fillna("").str.contains("chunk") & rows["ok"]].copy()
  if chunk.empty:
    return
  groups = list(chunk.groupby(["model", "batch_size", "max_length"], dropna=False))
  fig, axes = plt.subplots(
      len(groups),
      1,
      figsize=(8.6, 3.1 * len(groups)),
      constrained_layout=True,
  )
  if len(groups) == 1:
    axes = [axes]
  fig.suptitle(
      "CCE chunk tuning on larger Gemma pressure points\n"
      "Cell text: mean step time / XLA planned HBM per chip",
      fontsize=11.3,
      fontweight="bold",
  )
  for ax, ((model, batch, length), part) in zip(axes, groups, strict=True):
    pivot = part.pivot_table(
        index="token_chunk",
        columns="vocab_chunk",
        values="mean_step_time_sec_excl_first",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)
    hbm = part.pivot_table(
        index="token_chunk",
        columns="vocab_chunk",
        values="xla_train_step_gib_per_chip",
        aggfunc="first",
    ).reindex(index=pivot.index, columns=pivot.columns)
    im = ax.imshow(pivot.values, cmap="YlGnBu_r", aspect="auto")
    ax.set_title(
        f"{MODEL_LABEL.get(model, model)} b{int(batch)}/L{int(length)} on v5litepod-8 fsdp8/tp1",
        fontsize=9.5,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(value)//1024}K" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(value)) for value in pivot.index])
    ax.set_xlabel("Vocab chunk")
    ax.set_ylabel("Token chunk")
    finite_values = pivot.stack().dropna()
    threshold = finite_values.quantile(0.45) if not finite_values.empty else 0.0
    for y in range(pivot.shape[0]):
      for x in range(pivot.shape[1]):
        value = pivot.iat[y, x]
        hbm_value = hbm.iat[y, x]
        if pd.isna(value):
          continue
        text_color = "#FFFFFF" if value <= threshold else "#111111"
        ax.text(
            x,
            y,
            f"{value:.2f}s\n{hbm_value:.1f}G",
            ha="center",
            va="center",
            fontsize=7.2,
            color=text_color,
        )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean step time (sec)")
  fig.savefig(ASSET_DIR / "gemma_cce_large_transfer_chunk_tuning.png", dpi=190)
  plt.close(fig)


def main() -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  extract_artifacts()
  rows = read_results()
  frontier_summary = build_frontier_summary(rows)
  matched = build_matched_metrics(rows)
  pressure = build_pressure_points(rows)
  chunk_rows = rows[rows["suite"].fillna("").str.contains("chunk")].copy()
  chunk_summary = build_chunk_summary(rows)

  write_frame(build_manifest(), DATA_DIR / "run_manifest.csv")
  write_frame(rows, DATA_DIR / "transfer_runs.csv")
  write_frame(frontier_summary, DATA_DIR / "frontier_summary.csv")
  write_frame(matched, DATA_DIR / "matched_metrics.csv")
  write_frame(pressure, DATA_DIR / "pressure_points.csv")
  write_frame(chunk_rows, DATA_DIR / "chunk_runs.csv")
  write_frame(chunk_summary, DATA_DIR / "chunk_summary.csv")

  plot_frontier(frontier_summary)
  plot_pressure(pressure)
  plot_chunk_heatmaps(rows)

  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma_cce_large_transfer_frontier.png'}")
  print(f"figure={ASSET_DIR / 'gemma_cce_large_transfer_pressure.png'}")
  print(f"figure={ASSET_DIR / 'gemma_cce_large_transfer_chunk_tuning.png'}")


if __name__ == "__main__":
  main()
