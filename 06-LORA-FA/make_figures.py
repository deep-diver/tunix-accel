#!/usr/bin/env python3
"""Builds LoRA-FA visualizations from collected benchmark summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"


RUNS = {
    "full-gemma3-270m": "v5litepod-1, b16, L512, 50 steps",
    "full-gemma3-1b-b8": "v5litepod-8, b8, L512, 50 steps",
    "full-gemma3-4b-b4": "v5litepod-8, b4, L512, 50 steps",
    "probe-gemma3-large-b1": "v5litepod-8, b1, L512, 2 steps",
    "probe-gemma3-large-r64-b1": "v5litepod-8, b1, L512, 2 steps",
    "probe-gemma3-27b-r32-b1": "v5litepod-8, b1, L512, 2 steps",
    "lorafa-probe-gemma4-e2b-b4": "v5litepod-8, b4, L512, 2 steps",
    "lorafa-probe-gemma4-e4b-b2": "v5litepod-8, b2, L512, 2 steps",
    "lorafa-cache-probe-large-r64": (
        "v5litepod-8, b1, L512, 2 steps, cached correction"
    ),
}


def load_summary(run: str, model: str, variant: str) -> dict:
  path = RESULTS / run / model / variant / "unpacked" / "summary.json"
  if not path.exists():
    path = RESULTS / run / model / variant / "summary.json"
  return json.loads(path.read_text())


def row(run: str, model: str, variant: str) -> dict:
  summary = load_summary(run, model, variant)
  memory = summary.get("memory_after_train", {}).get("aggregate", {})
  lora_delta = summary.get("lora_fa", {}).get("value_delta", {})
  return {
      "run": run,
      "model": model,
      "variant": variant,
      "rank": int(summary.get("lora_rank")),
      "lorafa": bool(summary.get("accel", {}).get("lora_fa_installed")),
      "peak_gib": memory.get("peak_bytes_in_use", 0) / (1024**3),
      "step_time": summary.get("mean_step_time_sec_excl_first"),
      "a_delta": lora_delta.get("lorafa_a_value_delta_max"),
      "b_delta": lora_delta.get("lorafa_b_value_delta_max"),
      "setup": RUNS.get(run, ""),
  }


def pair(run: str, model: str, rank: int) -> tuple[dict, dict]:
  return (
      row(run, model, f"standard_lora_r{rank}"),
      row(run, model, f"lorafa_r{rank}"),
  )


def annotate_point(ax, x, y, text, dy=8):
  ax.annotate(
      text,
      xy=(x, y),
      xytext=(0, dy),
      textcoords="offset points",
      ha="center",
      fontsize=8,
      color="#2f3437",
  )


def style_axis(ax):
  ax.grid(axis="y", color="#d8dee4", linewidth=0.8, alpha=0.8)
  ax.set_axisbelow(True)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color("#8c959f")
  ax.spines["bottom"].set_color("#8c959f")
  ax.tick_params(colors="#333333")


def make_rank_sweep():
  models = [
      ("gemma3-270m", "full-gemma3-270m", "Gemma3 270M\nv5litepod-1, b16"),
      ("gemma3-1b", "full-gemma3-1b-b8", "Gemma3 1B\nv5litepod-8, b8"),
      ("gemma3-4b", "full-gemma3-4b-b4", "Gemma3 4B\nv5litepod-8, b4"),
  ]
  ranks = [16, 32, 64]
  fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), sharex="col")
  std_color = "#5269a8"
  fa_color = "#0f9d8a"

  for col, (model, run, title) in enumerate(models):
    standard = [pair(run, model, rank)[0] for rank in ranks]
    lorafa = [pair(run, model, rank)[1] for rank in ranks]
    ax = axes[0, col]
    ax.plot(
        ranks,
        [r["peak_gib"] for r in standard],
        marker="o",
        markersize=8,
        linewidth=2.4,
        color=std_color,
        label="Standard LoRA",
    )
    ax.plot(
        ranks,
        [r["peak_gib"] for r in lorafa],
        marker="o",
        markersize=8,
        linewidth=2.4,
        color=fa_color,
        label="LoRA-FA",
    )
    for rank, std, fa in zip(ranks, standard, lorafa):
      saved = std["peak_gib"] - fa["peak_gib"]
      annotate_point(ax, rank, fa["peak_gib"], f"{saved:.2f} GiB saved", dy=9)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel("Aggregate train peak GiB" if col == 0 else "")
    ax.set_xticks(ranks)
    ax.set_xlim(12, 68)
    style_axis(ax)

    ax = axes[1, col]
    ax.axhline(1.0, color="#8c959f", linewidth=1.0, linestyle="--")
    ratios = [fa["step_time"] / std["step_time"] for std, fa in zip(standard, lorafa)]
    ax.plot(
        ranks,
        ratios,
        marker="o",
        markersize=8,
        linewidth=2.4,
        color="#c2542d",
    )
    for rank, ratio in zip(ranks, ratios):
      annotate_point(ax, rank, ratio, f"{ratio:.2f}x")
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Step time ratio\nLoRA-FA / standard" if col == 0 else "")
    ax.set_xticks(ranks)
    ax.set_xlim(12, 68)
    style_axis(ax)

  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
  fig.suptitle(
      "LoRA-FA rank sweep on Gemma3: memory drops, step-time tradeoff depends on rank",
      fontsize=15,
      y=1.035,
  )
  fig.text(
      0.5,
      0.01,
      "Aggregate train peak is the sum of JAX-reported TPU device peaks. Runs use default CE, LoRA alpha 32, max length 512.",
      ha="center",
      fontsize=9,
      color="#57606a",
  )
  fig.tight_layout(rect=(0, 0.035, 1, 0.94))
  out = ASSETS / "lorafa_gemma3_rank_sweep.png"
  fig.savefig(out, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return out


def make_large_probe():
  gemma3_12b_r64_std = row(
      "probe-gemma3-large-r64-b1", "gemma3-12b", "standard_lora_r64"
  )
  gemma3_12b_r64_fa = row(
      "lorafa-cache-probe-large-r64", "gemma3-12b", "lorafa_r64"
  )
  pairs = [
      ("Gemma3 12B r16\nv5litepod-8 b1", *pair("probe-gemma3-large-b1", "gemma3-12b", 16)),
      ("Gemma3 12B r64\nv5litepod-8 b1", gemma3_12b_r64_std, gemma3_12b_r64_fa),
      ("Gemma3 27B r16\nv5litepod-8 b1", *pair("probe-gemma3-large-b1", "gemma3-27b", 16)),
      ("Gemma3 27B r32\nv5litepod-8 b1", *pair("probe-gemma3-27b-r32-b1", "gemma3-27b", 32)),
      ("Gemma4 E2B r16\nv5litepod-8 b4", *pair("lorafa-probe-gemma4-e2b-b4", "gemma4-e2b", 16)),
      ("Gemma4 E4B r16\nv5litepod-8 b2", *pair("lorafa-probe-gemma4-e4b-b2", "gemma4-e4b", 16)),
  ]
  labels = [item[0] for item in pairs]
  standard = [item[1] for item in pairs]
  lorafa = [item[2] for item in pairs]
  y = np.arange(len(labels))
  height = 0.36

  fig, axes = plt.subplots(
      1,
      2,
      figsize=(13.2, 7.0),
      gridspec_kw={"width_ratios": [1.2, 1.0]},
  )
  std_color = "#5269a8"
  fa_color = "#0f9d8a"

  ax = axes[0]
  ax.barh(y + height / 2, [r["peak_gib"] for r in standard], height, color=std_color, label="Standard LoRA")
  ax.barh(y - height / 2, [r["peak_gib"] for r in lorafa], height, color=fa_color, label="LoRA-FA")
  ax.set_yticks(y)
  ax.set_yticklabels(labels, fontsize=9)
  ax.invert_yaxis()
  ax.set_xlabel("Aggregate train peak GiB")
  ax.set_title("Train memory")
  for idx, (std, fa) in enumerate(zip(standard, lorafa)):
    saved = std["peak_gib"] - fa["peak_gib"]
    pct = saved / std["peak_gib"] * 100
    x = max(std["peak_gib"], fa["peak_gib"])
    ax.text(
        x + 1.0,
        idx,
        f"{saved:.2f} GiB saved ({pct:.1f}%)",
        va="center",
        fontsize=8,
    )
  style_axis(ax)

  ax = axes[1]
  ax.barh(y + height / 2, [r["step_time"] for r in standard], height, color=std_color, label="Standard LoRA")
  ax.barh(y - height / 2, [r["step_time"] for r in lorafa], height, color=fa_color, label="LoRA-FA")
  ax.set_yticks(y)
  ax.set_yticklabels([])
  ax.invert_yaxis()
  ax.set_xlabel("Mean step time, excl. first step (sec)")
  ax.set_title("Step time")
  for idx, (std, fa) in enumerate(zip(standard, lorafa)):
    ratio = fa["step_time"] / std["step_time"]
    x = max(std["step_time"], fa["step_time"])
    ax.text(x + 0.035, idx, f"{ratio:.2f}x", va="center", fontsize=8)
  style_axis(ax)

  handles, legend_labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
  fig.suptitle(
      "LoRA-FA probes on larger Gemma3 and Gemma4 models",
      fontsize=15,
      y=1.03,
  )
  fig.text(
      0.5,
      0.01,
      "All bars are measured on Cloud TPU v5litepod-8 with max length 512. Gemma3 12B r64 uses the cached-correction LoRA-FA implementation. Gemma4 E4B b4 failed before this b2 probe due to default CE logits.",
      ha="center",
      fontsize=9,
      color="#57606a",
  )
  fig.tight_layout(rect=(0, 0.04, 1, 0.94))
  out = ASSETS / "lorafa_large_model_probes.png"
  fig.savefig(out, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return out


def make_12b_memory_accounting():
  ranks = [16, 32, 64]
  standard = [pair("full-gemma3-12b-b1", "gemma3-12b", rank)[0] for rank in ranks]
  lorafa = [pair("full-gemma3-12b-b1", "gemma3-12b", rank)[1] for rank in ranks]

  def in_use_gib(record):
    summary = load_summary(record["run"], record["model"], record["variant"])
    memory = summary.get("memory_after_train", {}).get("aggregate", {})
    return memory.get("bytes_in_use", 0) / (1024**3)

  std_in_use = [in_use_gib(record) for record in standard]
  fa_in_use = [in_use_gib(record) for record in lorafa]
  std_peak = [record["peak_gib"] for record in standard]
  fa_peak = [record["peak_gib"] for record in lorafa]
  std_time = [record["step_time"] for record in standard]
  fa_time = [record["step_time"] for record in lorafa]

  x = np.arange(len(ranks))
  width = 0.34
  std_color = "#5269a8"
  fa_color = "#0f9d8a"

  fig, axes = plt.subplots(
      1,
      3,
      figsize=(14.2, 4.8),
      gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]},
  )

  panels = [
      (
          axes[0],
          std_in_use,
          fa_in_use,
          "Resident after train",
          "Aggregate GiB",
          "saved",
      ),
      (
          axes[1],
          std_peak,
          fa_peak,
          "High-water train peak",
          "Aggregate GiB",
          "delta",
      ),
      (
          axes[2],
          std_time,
          fa_time,
          "Step time",
          "Seconds / step",
          "ratio",
      ),
  ]

  for ax, std_values, fa_values, title, ylabel, annotation_kind in panels:
    ax.bar(x - width / 2, std_values, width, color=std_color, label="Standard LoRA")
    ax.bar(x + width / 2, fa_values, width, color=fa_color, label="LoRA-FA")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"r{rank}" for rank in ranks])
    ax.set_ylabel(ylabel)
    ymax = max(max(std_values), max(fa_values))
    ax.set_ylim(0, ymax * 1.20)
    for idx, (std_value, fa_value) in enumerate(zip(std_values, fa_values)):
      if annotation_kind == "saved":
        text = f"{std_value - fa_value:.2f} GiB saved"
      elif annotation_kind == "delta":
        diff = fa_value - std_value
        text = f"FA {diff:+.2f} GiB"
      else:
        text = f"{fa_value / std_value:.2f}x"
      ax.text(
          idx,
          max(std_value, fa_value) + ymax * 0.035,
          text,
          ha="center",
          va="bottom",
          fontsize=8,
          color="#2f3437",
      )
    style_axis(ax)

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
  fig.suptitle(
      "Gemma3 12B LoRA-FA accounting: resident memory improves, peak needs separate reading",
      fontsize=14,
      y=1.10,
  )
  fig.text(
      0.5,
      0.01,
      "Cloud TPU v5litepod-8, 8 chips, batch 1, max length 512, 50 train steps, default CE, LoRA alpha 32. Memory is aggregate across chips.",
      ha="center",
      fontsize=9,
      color="#57606a",
  )
  fig.tight_layout(rect=(0, 0.055, 1, 0.95))
  out = ASSETS / "lorafa_gemma3_12b_memory_accounting.png"
  fig.savefig(out, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return out


def make_all_models_r16_overview():
  entries = [
      {
          "label": "G3 270M\nfull b16",
          "kind": "full",
          "std": row("full-gemma3-270m", "gemma3-270m", "standard_lora_r16"),
          "fa": row("full-gemma3-270m", "gemma3-270m", "lorafa_r16"),
      },
      {
          "label": "G3 1B\nfull b8",
          "kind": "full",
          "std": row("full-gemma3-1b-b8", "gemma3-1b", "standard_lora_r16"),
          "fa": row("full-gemma3-1b-b8", "gemma3-1b", "lorafa_r16"),
      },
      {
          "label": "G4 E2B\nprobe b4",
          "kind": "probe",
          "std": row("lorafa-probe-gemma4-e2b-b4", "gemma4-e2b", "standard_lora_r16"),
          "fa": row("lorafa-probe-gemma4-e2b-b4", "gemma4-e2b", "lorafa_r16"),
      },
      {
          "label": "G3 4B\nfull b4",
          "kind": "full",
          "std": row("full-gemma3-4b-b4", "gemma3-4b", "standard_lora_r16"),
          "fa": row("full-gemma3-4b-b4", "gemma3-4b", "lorafa_r16"),
      },
      {
          "label": "G4 E4B\nprobe b2",
          "kind": "probe",
          "std": row("lorafa-probe-gemma4-e4b-b2", "gemma4-e4b", "standard_lora_r16"),
          "fa": row("lorafa-probe-gemma4-e4b-b2", "gemma4-e4b", "lorafa_r16"),
      },
      {
          "label": "G3 12B\nfull b1",
          "kind": "full",
          "std": row("full-gemma3-12b-b1", "gemma3-12b", "standard_lora_r16"),
          "fa": row("full-gemma3-12b-b1", "gemma3-12b", "lorafa_r16"),
      },
      {
          "label": "G3 27B\nprobe b1",
          "kind": "probe",
          "std": row("probe-gemma3-large-b1", "gemma3-27b", "standard_lora_r16"),
          "fa": row("probe-gemma3-large-b1", "gemma3-27b", "lorafa_r16"),
      },
  ]

  def memory_value(record, field):
    summary = load_summary(record["run"], record["model"], record["variant"])
    memory = summary.get("memory_after_train", {}).get("aggregate", {})
    return memory.get(field, 0) / (1024**3)

  labels = [entry["label"] for entry in entries]
  resident_savings = []
  peak_savings = []
  step_ratios = []
  for entry in entries:
    std = entry["std"]
    fa = entry["fa"]
    std_resident = memory_value(std, "bytes_in_use")
    fa_resident = memory_value(fa, "bytes_in_use")
    resident_savings.append((std_resident - fa_resident) / std_resident * 100)
    peak_savings.append((std["peak_gib"] - fa["peak_gib"]) / std["peak_gib"] * 100)
    step_ratios.append(fa["step_time"] / std["step_time"])

  x = np.arange(len(entries))
  width = 0.34
  resident_color = "#0f9d8a"
  peak_color = "#5269a8"
  ratio_color = "#c2542d"

  fig, axes = plt.subplots(
      1,
      2,
      figsize=(14.6, 5.7),
      gridspec_kw={"width_ratios": [1.45, 1.0]},
  )

  ax = axes[0]
  ax.axhline(0, color="#8c959f", linewidth=1.0)
  ax.bar(x - width / 2, resident_savings, width, color=resident_color, label="Resident memory saved")
  ax.bar(x + width / 2, peak_savings, width, color=peak_color, label="High-water peak saved")
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=8)
  ax.set_ylabel("Memory saved vs Standard LoRA (%)")
  ax.set_title("Common rank r16 memory comparison")
  ymin = min(0, min(resident_savings), min(peak_savings)) - 6
  ymax = max(resident_savings + peak_savings) + 8
  ax.set_ylim(ymin, ymax)
  for idx, value in enumerate(resident_savings):
    va = "bottom" if value >= 0 else "top"
    offset = 1.2 if value >= 0 else -1.2
    ax.text(idx - width / 2, value + offset, f"{value:.1f}%", ha="center", va=va, fontsize=7)
  for idx, value in enumerate(peak_savings):
    va = "bottom" if value >= 0 else "top"
    offset = 1.2 if value >= 0 else -1.2
    ax.text(idx + width / 2, value + offset, f"{value:.1f}%", ha="center", va=va, fontsize=7)
  style_axis(ax)
  ax.legend(frameon=False, loc="upper right", fontsize=8)

  ax = axes[1]
  ax.axhline(1.0, color="#8c959f", linewidth=1.0, linestyle="--")
  for idx, (entry, ratio) in enumerate(zip(entries, step_ratios)):
    face = ratio_color if entry["kind"] == "full" else "white"
    ax.scatter(
        idx,
        ratio,
        s=80,
        marker="o",
        facecolors=face,
        edgecolors=ratio_color,
        linewidths=2.0,
        zorder=3,
    )
    ax.text(idx, ratio + 0.035, f"{ratio:.2f}x", ha="center", fontsize=7)
  ax.plot(x, step_ratios, color=ratio_color, linewidth=1.2, alpha=0.45)
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=8)
  ax.set_ylabel("LoRA-FA / Standard LoRA")
  ax.set_title("Step time ratio")
  ax.set_ylim(0.75, max(step_ratios) + 0.18)
  style_axis(ax)
  ax.scatter([], [], s=80, facecolors=ratio_color, edgecolors=ratio_color, label="50-step full")
  ax.scatter([], [], s=80, facecolors="white", edgecolors=ratio_color, linewidths=2.0, label="2-step probe")
  ax.legend(frameon=False, loc="upper left", fontsize=8)

  fig.suptitle(
      "LoRA-FA overview across collected models: one rank, mixed experiment depth",
      fontsize=14,
      y=1.03,
  )
  fig.text(
      0.5,
      0.01,
      "Rank 16, LoRA alpha 32, max length 512. Full runs use 50 train steps; probe runs use 2 train steps. TPU type is v5litepod; chip count is 1 for 270M and 8 for the other shown runs.",
      ha="center",
      fontsize=9,
      color="#57606a",
  )
  fig.tight_layout(rect=(0, 0.07, 1, 0.96))
  out = ASSETS / "lorafa_all_models_r16_overview.png"
  fig.savefig(out, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return out


def make_correctness_delta():
  checks = [
      ("270M r16", *pair("full-gemma3-270m", "gemma3-270m", 16)),
      ("1B r16", *pair("full-gemma3-1b-b8", "gemma3-1b", 16)),
      ("4B r16", *pair("full-gemma3-4b-b4", "gemma3-4b", 16)),
      ("12B r16", *pair("probe-gemma3-large-b1", "gemma3-12b", 16)),
      ("27B r32", *pair("probe-gemma3-27b-r32-b1", "gemma3-27b", 32)),
      ("G4 E2B r16", *pair("lorafa-probe-gemma4-e2b-b4", "gemma4-e2b", 16)),
      ("G4 E4B r16", *pair("lorafa-probe-gemma4-e4b-b2", "gemma4-e4b", 16)),
  ]
  labels = [item[0] for item in checks]
  standard = [item[1]["a_delta"] for item in checks]
  lorafa = [item[2]["a_delta"] for item in checks]
  x = np.arange(len(labels))
  width = 0.36

  fig, ax = plt.subplots(figsize=(11.6, 4.6))
  ax.bar(x - width / 2, standard, width, color="#5269a8", label="Standard LoRA A delta")
  ax.bar(x + width / 2, lorafa, width, color="#0f9d8a", label="LoRA-FA A delta")
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=0, fontsize=9)
  ax.set_ylabel("Max absolute delta in LoRA A")
  ax.set_title("Freeze check: LoRA-FA leaves A unchanged")
  for idx, value in enumerate(lorafa):
    ax.text(idx + width / 2, value + 0.00025, "0", ha="center", fontsize=8)
  style_axis(ax)
  ax.legend(frameon=False, ncol=2, loc="upper right")
  fig.tight_layout()
  out = ASSETS / "lorafa_a_freeze_check.png"
  fig.savefig(out, dpi=180, bbox_inches="tight")
  plt.close(fig)
  return out


def main():
  ASSETS.mkdir(parents=True, exist_ok=True)
  outputs = [
      make_all_models_r16_overview(),
      make_rank_sweep(),
      make_large_probe(),
      make_12b_memory_accounting(),
      make_correctness_delta(),
  ]
  for output in outputs:
    print(output)


if __name__ == "__main__":
  main()
