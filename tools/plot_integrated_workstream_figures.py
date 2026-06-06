#!/usr/bin/env python3
"""Regenerate the existing report figures with Gemma4 rows folded in."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BLUE = "#4e7fb2"
GREEN = "#2ca24f"
ORANGE = "#ff7f0e"
GRAY = "#6f7f93"
RED = "#d62728"
TEAL = "#2c7f7b"
INK = "#111827"
MUTED = "#4b5563"
GRID = "#d7dde5"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def fmt_time(value: float) -> str:
    if value < 1:
        return f"{value:.2f}s"
    if value < 10:
        return f"{value:.1f}s"
    return f"{value:.0f}s"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 180,
        }
    )


def annotate_bar(ax, bar, text: str, dy: float = 0.02, size: int = 12) -> None:
    y = bar.get_height()
    ylim = ax.get_ylim()
    offset = (ylim[1] - ylim[0]) * dy
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y + offset,
        text,
        ha="center",
        va="bottom",
        fontsize=size,
        fontweight="bold",
        color=INK,
    )


def style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path.relative_to(ROOT))


def cce_memory_plot() -> None:
    g3 = read_csv(ROOT / "01-CCE/data/gemma3_b16_aggregate_hbm.csv")
    g4 = read_csv(ROOT / "01-CCE/data/gemma4_base_cce_tpu_l2048_b1.csv")

    groups: list[dict[str, object]] = []
    for size, label in [("270m", "G3 270M\nb16 L2048\n1 chip"), ("1b", "G3 1B\nb16 L2048\n4 chips"), ("4b", "G3 4B\nb16 L2048\n8 chips")]:
        by = {r["variant"]: r for r in g3 if r["size"] == size}
        groups.append(
            {
                "label": label,
                "default": fnum(by["default-lora"]["xla_peak_gib"]),
                "patched": fnum(by["cce-lora"]["xla_peak_gib"]),
                "default_abs": fnum(by["default-lora"]["aggregate_hbm_gib"]),
                "patched_abs": fnum(by["cce-lora"]["aggregate_hbm_gib"]),
                "default_status": by["default-lora"]["status"],
                "patched_status": by["cce-lora"]["status"],
                "patched_note": f"{(1 - fnum(by['cce-lora']['xla_peak_gib']) / fnum(by['default-lora']['xla_peak_gib'])) * 100:.0f}% lower",
            }
        )

    for model, label in [
        ("Gemma4 E2B", "G4 E2B\nbase b1 L2048\n4 chips"),
        ("Gemma4 E4B", "G4 E4B\nbase b1 L2048\n8 chips"),
    ]:
        by = {r["variant"]: r for r in g4 if r["model"] == model}
        default = fnum(by["default"]["compile_oom_used_gb_per_chip"])
        if by["cce"]["status"] == "ok":
            patched = fnum(by["cce"]["runtime_peak_hbm_gb_per_chip"])
            note = "runtime peak"
            patched_abs = fnum(by["cce"]["runtime_peak_hbm_gb_aggregate"])
        else:
            patched = fnum(by["cce"]["compile_oom_used_gb_per_chip"])
            note = f"{(1 - patched / default) * 100:.0f}% lower"
            patched_abs = patched * int(by["cce"]["chips"])
        groups.append(
            {
                "label": label,
                "default": default,
                "patched": patched,
                "default_abs": default * int(by["default"]["chips"]),
                "patched_abs": patched_abs,
                "default_status": by["default"]["status"],
                "patched_status": by["cce"]["status"],
                "patched_note": note,
            }
        )

    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(17, 8))
    defaults = [g["default"] for g in groups]
    patched = [g["patched"] for g in groups]
    b1 = ax.bar(x - width / 2, defaults, width, color=GRAY, label="Default CE")
    b2 = ax.bar(x + width / 2, patched, width, color=GREEN, label="CCE")

    for bars, key in [(b1, "default_status"), (b2, "patched_status")]:
        for bar, g in zip(bars, groups):
            if g[key] == "oom":
                bar.set_hatch("//")
                bar.set_edgecolor(INK)
                bar.set_linewidth(0.8)

    ax.axhline(15.75, color="#111827", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.text(-0.45, 15.75 + 0.35, "~15.75 GB/chip v5e HBM", ha="left", va="bottom", color=MUTED, fontsize=10)
    ax.axvline(2.5, color="#9ca3af", linewidth=1.2, linestyle="--")
    ax.text(1.0, max(defaults + patched) * 1.06, "Gemma3 train-shape XLA planned HBM", ha="center", color=MUTED, fontsize=12)
    ax.text(3.5, max(defaults + patched) * 1.06, "Gemma4 base boundary check", ha="center", color=MUTED, fontsize=12)

    ax.set_xticks(x, [g["label"] for g in groups])
    ax.set_ylabel("Max per-chip HBM pressure (GB/GiB)")
    ax.set_ylim(0, max(defaults + patched) * 1.18)
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    style_axes(ax)

    for bar, g in zip(b1, groups):
        annotate_bar(ax, bar, f"{bar.get_height():.1f}/chip\n{str(g['default_status']).upper()}", size=10)
    for bar, g in zip(b2, groups):
        annotate_bar(ax, bar, f"{bar.get_height():.1f}/chip\n{g['patched_note']}\n{str(g['patched_status']).upper()}", size=10)

    fig.suptitle("Gemma LoRA SFT Memory Boundary: Default CE vs CCE", fontsize=28, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.915,
        "Gemma4 rows are folded into the same memory-boundary readout; hatching marks compile OOM.",
        ha="center",
        fontsize=15,
        color=MUTED,
    )
    fig.text(
        0.5,
        0.03,
        "Boundary charts use max per-chip HBM because OOM is per-chip. Gemma4 OK CCE uses retained runtime peak; tables retain aggregate accounting.",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.03, 0.07, 0.98, 0.88))
    save(fig, ROOT / "01-CCE/assets/gemma3_gemma4_cce_per_chip_hbm.png")


def packing_plot() -> None:
    g3 = read_csv(ROOT / "02-PACKING/data/gemma3_1b_4b_scale_smoke_summary.csv")
    models = ["Gemma3 1B", "Gemma3 4B"]
    labels = ["Gemma3 1B", "Gemma3 4B"]
    unpacked = []
    packed = []
    dens_u = []
    dens_p = []
    step_u = []
    step_p = []
    for model in models:
        by = {r["variant"]: r for r in g3 if r["model"] == model}
        unpacked.append(fnum(by["unpacked"]["loss_tokens_per_sec_excl_first"]))
        packed.append(fnum(by["packed"]["loss_tokens_per_sec_excl_first"]))
        dens_u.append(fnum(by["unpacked"]["packing_efficiency"]))
        dens_p.append(fnum(by["packed"]["packing_efficiency"]))
        step_u.append(fnum(by["unpacked"]["mean_step_time_sec_excl_first"]))
        step_p.append(fnum(by["packed"]["mean_step_time_sec_excl_first"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    fig.suptitle("Train Throughput Comes From Denser Steps", fontsize=22, fontweight="bold", y=0.995)

    x = np.arange(len(labels))
    width = 0.36
    ax = axes[0]
    b1 = ax.bar(x - width / 2, unpacked, width, color=BLUE, label="unpacked")
    b2 = ax.bar(x + width / 2, packed, width, color=ORANGE, label="packed")
    ax.set_title("Target-token throughput", pad=18)
    ax.set_ylabel("Target tokens/sec")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(packed) * 1.34)
    ax.legend(frameon=False)
    style_axes(ax)
    for i, (u, p) in enumerate(zip(unpacked, packed)):
        annotate_bar(ax, b1[i], f"{u:.0f}", size=10)
        annotate_bar(ax, b2[i], f"{p:.0f}", size=10)
        ax.text(i, max(packed) * 1.18, f"{p/u:.1f}x", ha="center", fontsize=12, fontweight="bold", color=INK)

    ax = axes[1]
    b1 = ax.bar(x - width / 2, dens_u, width, color=BLUE, label="unpacked")
    b2 = ax.bar(x + width / 2, dens_p, width, color=ORANGE, label="packed")
    ax.set_title("Batch density", pad=18)
    ax.set_ylabel("Non-pad token density")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.24)
    style_axes(ax)
    for i, (u, p) in enumerate(zip(dens_u, dens_p)):
        annotate_bar(ax, b1[i], f"{u*100:.1f}%", size=10)
        annotate_bar(ax, b2[i], f"{p*100:.1f}%", size=10)
        ax.text(i, 1.15, f"{p/u:.1f}x", ha="center", fontsize=12, fontweight="bold", color=INK)

    ax = axes[2]
    b1 = ax.bar(x - width / 2, step_u, width, color=BLUE)
    b2 = ax.bar(x + width / 2, step_p, width, color=ORANGE)
    ax.set_title("Step time", pad=18)
    ax.set_ylabel("Seconds/step")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(step_u + step_p) * 1.45)
    style_axes(ax)
    for bars, vals in [(b1, step_u), (b2, step_p)]:
        for bar, val in zip(bars, vals):
            annotate_bar(ax, bar, f"{val:.3f}", size=10)

    fig.text(
        0.5,
        0.02,
        "This figure only includes comparable packing throughput runs. Gemma4 fixed-shape compile checks are reported separately as a negative control table.",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.90), w_pad=2.8)
    save(fig, ROOT / "02-PACKING/assets/gemma3_1b_4b_throughput_and_density.png")


def tiled_mlp_plot() -> None:
    g3 = read_csv(ROOT / "03-TILED-MLP/data/gemma3_4b_context_keypoints.csv")
    g4 = read_csv(ROOT / "03-TILED-MLP/data/gemma4_base_tiled_mlp_tpu_l2048_b1.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.suptitle("Gemma MLP Memory Boundary", fontsize=24, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.94,
        "Gemma3 4B uses the original context boundary; Gemma4 rows are base b1/L2048 boundary checks.",
        ha="center",
        color=MUTED,
        fontsize=13,
    )

    ax = axes[0]
    ax.barh([1, 0], [2048, 4096], color=[BLUE, GREEN], height=0.45)
    ax.set_yticks([1, 0], ["Gemma3 4B\nDefault MLP", "Gemma3 4B\nTiled MLP"])
    ax.set_xlim(0, 4600)
    ax.set_xticks([0, 1024, 2048, 3072, 4096])
    ax.set_title("A. Gemma3 completed context")
    ax.set_xlabel("Longest completed context length")
    ax.text(2048 - 70, 1, "L2,048", ha="right", va="center", color="white", fontweight="bold", fontsize=12)
    ax.text(4096 - 70, 0, "L4,096", ha="right", va="center", color="white", fontweight="bold", fontsize=12)
    ax.text(2230, 0.48, "Default L4096: compile OOM", color=RED, fontweight="bold", fontsize=11)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    bars = []
    labels = []
    values = []
    colors = []
    statuses = []
    for context in ["2048", "4096"]:
        for variant, label, color in [("default", "Default", BLUE), ("tiled", "Tiled", GREEN)]:
            r = next(row for row in g3 if row["max_length"] == context and row["variant"] == variant)
            labels.append(f"G3 4B\nL{context}\n{label}")
            values.append(fnum(r["xla_hbm_gib_per_chip"]))
            colors.append(color if r["status"] == "ok" else RED)
            statuses.append(r["status"])
    for model in ["Gemma4 E2B", "Gemma4 E4B"]:
        by = {r["variant"]: r for r in g4 if r["model"] == model}
        chips = int(by["default"]["chips"])
        labels.append(model.replace("Gemma4 ", "G4\n") + "\nDefault")
        values.append(fnum(by["default"]["compile_oom_used_gb_per_chip"]))
        colors.append(RED)
        statuses.append("oom")
        labels.append(model.replace("Gemma4 ", "G4\n") + "\nTiled")
        values.append(fnum(by["tiled_mlp"]["runtime_peak_hbm_gb_per_chip"]))
        colors.append(GREEN)
        statuses.append("ok")
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors)
    for bar, status in zip(bars, statuses):
        if status == "oom":
            bar.set_hatch("//")
            bar.set_edgecolor(INK)
    ax.set_title("B. Memory pressure at retained keypoints")
    ax.axhline(15.75, color=INK, linewidth=1.0, linestyle="--", alpha=0.65)
    ax.text(-0.45, 15.75 + 0.35, "~15.75 GB/chip v5e HBM", ha="left", va="bottom", color=MUTED, fontsize=10)
    ax.set_ylabel("Max per-chip HBM pressure (GB/GiB)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(values) * 1.22)
    style_axes(ax)
    for bar, value, status in zip(bars, values, statuses):
        annotate_bar(ax, bar, f"{value:.1f}/chip\n{status.upper()}", size=9)
    fig.text(
        0.5,
        0.02,
        "Hatched bars are compile OOM. Gemma4 OK bars use retained runtime peak because successful XLA memory reports were not retained.",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.03, 0.07, 0.99, 0.90))
    save(fig, ROOT / "03-TILED-MLP/assets/gemma3_4b_context_boundary_memory.png")


def activation_plot() -> None:
    g3 = read_csv(ROOT / "04-ACTIVATION-POLICY/data/gemma3_4b_activation_policy_keypoints.csv")
    g4 = read_csv(ROOT / "04-ACTIVATION-POLICY/data/gemma4_base_activation_policy_tpu_l2048_b1.csv")

    rows: list[tuple[str, float, str, str]] = []
    for context in ["2048", "4096"]:
        for policy, label, color in [("none", "before\nno policy", GRAY), ("split_offload", "after\nsplit offload", TEAL)]:
            r = next(row for row in g3 if row["max_length"] == context and row["activation_policy"] == policy)
            limit = fnum(r["runtime_limit_gib_aggregate"])
            if np.isnan(limit):
                limit = 15.748046398162842 * int(r["chips"])
            per_chip_limit = limit / int(r["chips"])
            headroom = per_chip_limit - fnum(r["xla_hbm_gib_per_chip"])
            rows.append((f"G3 4B L{context}\n{label}", headroom, r["status"], color))
    for model in ["Gemma4 E2B", "Gemma4 E4B"]:
        by = {r["variant"]: r for r in g4 if r["model"] == model}
        chips = int(by["default"]["chips"])
        limit = fnum(by["default"]["compile_oom_limit_gb_per_chip"])
        default_pressure = fnum(by["default"]["compile_oom_used_gb_per_chip"])
        rows.append((model.replace("Gemma4 ", "G4 ") + "\nbefore\nno policy", limit - default_pressure, "oom", GRAY))
        runtime_limit = fnum(by["split_offload"]["runtime_hbm_limit_gb_aggregate"]) / chips
        runtime_peak = fnum(by["split_offload"]["runtime_peak_hbm_gb_per_chip"])
        rows.append((model.replace("Gemma4 ", "G4 ") + "\nafter\nsplit offload", runtime_limit - runtime_peak, "ok", TEAL))

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    statuses = [r[2] for r in rows]
    colors = [RED if r[2] == "oom" else r[3] for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(y, values, color=colors)
    for bar, status in zip(bars, statuses):
        if status == "oom":
            bar.set_hatch("//")
            bar.set_edgecolor(INK)
    ax.axvline(0, color=INK, linewidth=1.4)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Per-chip HBM headroom vs limit, GB/GiB  (limit - memory pressure)")
    ax.set_title("Activation Offload HBM Headroom: Gemma3 + Gemma4")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    xmin, xmax = min(values) * 1.25, max(values) * 1.32
    ax.set_xlim(xmin, xmax)
    for bar, val, status in zip(bars, values, statuses):
        if val < -2:
            ax.text(
                val / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}  {status.upper()}",
                va="center",
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
        elif val < 0:
            ax.text(
                val - 0.35,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}  {status.upper()}",
                va="center",
                ha="right",
                fontsize=11,
                fontweight="bold",
                color=INK,
            )
        else:
            dx = (xmax - xmin) * 0.015
            ax.text(val + dx, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}  {status.upper()}", va="center", ha="left", fontsize=11, fontweight="bold")
    fig.text(
        0.5,
        0.03,
        "Boundary charts use per-chip headroom because OOM is per-chip. Gemma4 after bars use retained runtime peak headroom.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.95))
    save(fig, ROOT / "04-ACTIVATION-POLICY/assets/gemma3_4b_activation_hbm_headroom.png")


def main() -> None:
    configure()
    cce_memory_plot()
    packing_plot()
    tiled_mlp_plot()
    activation_plot()


if __name__ == "__main__":
    main()
