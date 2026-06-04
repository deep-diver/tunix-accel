#!/usr/bin/env python3
"""Build corrected Gemma3 12B/27B sweep tables and figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

VARIANT_ORDER = ["default", "cce", "tiled_mlp", "split_offload", "splash", "stacked"]
VARIANT_LABEL = {
    "default": "Default",
    "cce": "CCE",
    "tiled_mlp": "Tiled MLP",
    "split_offload": "Split/offload",
    "splash": "Splash",
    "stacked": "Stacked",
}
VARIANT_COLOR = {
    "default": "#6b7280",
    "cce": "#2563eb",
    "tiled_mlp": "#059669",
    "split_offload": "#ea580c",
    "splash": "#7c3aed",
    "stacked": "#dc2626",
}
MODEL_LABEL = {
    "12b": "Gemma3 12B\nCloud TPU v5litepod-4 · 4 chips",
    "27b": "Gemma3 27B\nCloud TPU v5litepod-8 · 8 chips",
}
SHORT_MODEL_LABEL = {
    "12b": "Gemma3 12B · v5litepod-4 · 4 chips",
    "27b": "Gemma3 27B · v5litepod-8 · 8 chips",
}
SETUP_FOOTER = (
    "Setup: Gemma3 12B on Cloud TPU v5litepod-4 (4 chips, mesh fsdp=4,tp=1); "
    "Gemma3 27B on Cloud TPU v5litepod-8 (8 chips, mesh fsdp=8,tp=1). "
    "Batch=1, LoRA r16/alpha32, 2 train steps. Memory = XLA train-step planned HBM per chip."
)
HBM_LIMIT_GIB = 15.75


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_size, dirname in [("12b", "12b_corrected"), ("27b", "27b_corrected")]:
        path = RAW / dirname / "sweep_results.csv"
        with path.open() as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["source_run"] = dirname
                row["model_size"] = model_size
                row["batch_size"] = int(row["batch_size"])
                row["max_length"] = int(row["max_length"])
                row["chips"] = int(row["chips"])
                row["xla_train_step_gib_per_chip"] = as_float(row["xla_train_step_gib_per_chip"])
                row["mean_step_time_sec_excl_first"] = as_float(row["mean_step_time_sec_excl_first"])
                row["first_step_time_sec"] = as_float(row["first_step_time_sec"])
                row["second_step_time_sec"] = as_float(row["second_step_time_sec"])
                row["autopatch_effective"] = row["autopatch_effective"] == "True"
                row["requested_autopatch"] = row["requested_autopatch"] == "True"
                rows.append(row)
    return rows


def write_tables(rows: list[dict[str, object]]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_size",
        "model_id",
        "tpu",
        "chips",
        "mesh_fsdp",
        "mesh_tp",
        "variant",
        "batch_size",
        "max_length",
        "status",
        "xla_train_step_gib_per_chip",
        "mean_step_time_sec_excl_first",
        "first_step_time_sec",
        "second_step_time_sec",
        "requested_autopatch",
        "autopatch_effective",
        "cce_installed",
        "gemma3_tiled_mlp_installed",
        "gemma3_activation_policy_installed",
        "gemma3_splash_attention_installed",
        "failure_type",
        "oom_used_gib",
        "oom_limit_gib",
        "source_run",
    ]
    for path in [
        DATA / "gemma3_large_patch_sweep_summary.csv",
        DATA / "gemma3_large_patch_sweep_corrected_summary.csv",
    ]:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "status": "corrected",
        "measurement": "XLA train-step planned HBM per chip from buffer assignment",
        "hbm_limit_gib_per_chip": HBM_LIMIT_GIB,
        "full_raw_archives": {
            "12b": "gs://gcp-ml-172005-ddpm-training/tunix-large-sweep/gemma3-large-patch-sweep-corrected-12b.tar.gz",
            "27b": "gs://gcp-ml-172005-ddpm-training/tunix-large-sweep/gemma3-large-patch-sweep-corrected-27b.tar.gz",
        },
    }
    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def row_for(rows: list[dict[str, object]], model: str, variant: str, context: int) -> dict[str, object]:
    matches = [
        r for r in rows
        if r["model_size"] == model and r["variant"] == variant and r["max_length"] == context
    ]
    if not matches:
        raise KeyError((model, variant, context))
    return matches[0]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": "#111827",
        "axes.linewidth": 0.8,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "axes.labelcolor": "#111827",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


def savefig(path: Path, *, footer: str | None = SETUP_FOOTER) -> None:
    fig = plt.gcf()
    rect = (0, 0.07, 1, 0.93) if footer else (0, 0, 1, 0.94)
    if footer:
        fig.text(
            0.5,
            0.018,
            footer,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4b5563",
        )
    fig.tight_layout(rect=rect)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_l512_patch_impact(rows: list[dict[str, object]]) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for ax, model in zip(axes, ["12b", "27b"]):
        values = []
        default = row_for(rows, model, "default", 512)["xla_train_step_gib_per_chip"]
        assert isinstance(default, float)
        for variant in VARIANT_ORDER:
            row = row_for(rows, model, variant, 512)
            value = row["xla_train_step_gib_per_chip"]
            assert isinstance(value, float)
            values.append(value)
        xs = range(len(VARIANT_ORDER))
        ax.bar(
            xs,
            values,
            color=[VARIANT_COLOR[v] for v in VARIANT_ORDER],
            width=0.74,
            edgecolor="white",
            linewidth=1.2,
        )
        for x, value, variant in zip(xs, values, VARIANT_ORDER):
            delta = (default - value) / default * 100
            label = f"{value:.2f}"
            ax.text(x, value + 0.07, label, ha="center", va="bottom", fontsize=8)
            if variant != "default":
                pct_label = f"save {delta:.1f}%" if delta >= 0 else f"+{abs(delta):.1f}%"
                ax.text(
                    x,
                    value - 0.25,
                    pct_label,
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color="white" if delta > 1.5 else "#111827",
                    fontweight="bold" if delta > 1.5 else "normal",
                )
        ax.set_title(MODEL_LABEL[model])
        ax.set_xticks(list(xs))
        ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER], rotation=25, ha="right")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(10.8, 15.4)
    axes[0].set_ylabel("XLA planned HBM / chip (GiB), batch 1, L512")
    fig.suptitle("Corrected patch run: memory moves at the shared L512 baseline", fontweight="bold")
    savefig(ASSETS / "gemma3_large_l512_patch_impact.png")


def plot_context_frontier(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7), sharex=True)
    contexts = [512, 1024, 2048, 4096, 8192]
    for ax, model in zip(axes, ["12b", "27b"]):
        y_positions = list(range(len(VARIANT_ORDER)))
        for y, variant in zip(y_positions, VARIANT_ORDER):
            model_rows = [
                r for r in rows
                if r["model_size"] == model and r["variant"] == variant
            ]
            if not model_rows:
                continue
            ok_contexts = [int(r["max_length"]) for r in model_rows if r["status"] == "ok"]
            max_ok = max(ok_contexts) if ok_contexts else 0
            next_fail = min(
                [
                    int(r["max_length"]) for r in model_rows
                    if r["status"] != "ok" and int(r["max_length"]) > max_ok
                ],
                default=None,
            )
            ax.barh(
                y,
                max_ok,
                left=0,
                height=0.54,
                color=VARIANT_COLOR[variant],
                alpha=0.88,
            )
            ax.text(
                max_ok * 1.05,
                y,
                f"L{max_ok}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#111827",
            )
            if next_fail:
                ax.scatter([next_fail], [y], marker="x", color="#111827", s=46, linewidth=1.4)
                ax.text(next_fail * 1.07, y + 0.22, f"L{next_fail} OOM", fontsize=7.5, color="#4b5563")
        ax.set_title(MODEL_LABEL[model])
        ax.set_yticks(y_positions)
        ax.set_yticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER])
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts)
        ax.set_xticklabels([str(c) for c in contexts])
        ax.set_xlim(384, 11000)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
    axes[0].set_xlabel("Longest completed context length")
    axes[1].set_xlabel("Longest completed context length")
    fig.suptitle("Corrected batch-1 measured context frontier", fontweight="bold")
    savefig(ASSETS / "gemma3_large_context_frontier.png")


def plot_practical_readout(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.8),
        gridspec_kw={"width_ratios": [1.16, 1.0]},
    )

    ax = axes[0]
    labels = []
    colors = []
    statuses = []
    values = []
    ypos = []
    y = 0
    for model in ["12b", "27b"]:
        for variant in VARIANT_ORDER:
            try:
                row = row_for(rows, model, variant, 1024)
            except KeyError:
                continue
            labels.append(f"{MODEL_LABEL[model].splitlines()[0]}  {VARIANT_LABEL[variant]}")
            colors.append(VARIANT_COLOR[variant])
            statuses.append(row["status"])
            value = row["xla_train_step_gib_per_chip"]
            assert isinstance(value, float)
            values.append(value)
            ypos.append(y)
            y += 1
        y += 0.8
    bars = ax.barh(
        ypos,
        values,
        color=[c if s == "ok" else "#d1d5db" for c, s in zip(colors, statuses)],
        edgecolor=colors,
        linewidth=2.0,
        height=0.62,
    )
    ax.axvline(HBM_LIMIT_GIB, color="#111827", linestyle=(0, (5, 4)), linewidth=1.1)
    ax.text(
        HBM_LIMIT_GIB + 0.35,
        min(ypos) - 0.75,
        "v5e fit line\n~15.75 GiB/chip",
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
    )
    for y_i, bar, value, status in zip(ypos, bars, values, statuses):
        ax.text(
            value + 0.45,
            y_i,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=8,
        )
        if status != "ok":
            ax.text(
                min(value - 1.0, HBM_LIMIT_GIB - 0.45),
                y_i,
                "OOM",
                ha="right",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#111827",
            )
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("XLA planned HBM / chip at L1024 (GiB)")
    ax.set_title("At the first pressure point, only some patches cross the fit line")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 27.5)

    ax = axes[1]
    cases = [
        ("12B Default\nL512", "12b", "default", 512),
        ("12B Tiled\nL1024", "12b", "tiled_mlp", 1024),
        ("12B Stacked\nL4096", "12b", "stacked", 4096),
        ("27B Default\nL512", "27b", "default", 512),
        ("27B Split/offload\nL1024", "27b", "split_offload", 1024),
        ("27B Stacked\nL2048", "27b", "stacked", 2048),
    ]
    labels2, values2, colors2 = [], [], []
    for label, model, variant, context in cases:
        row = row_for(rows, model, variant, context)
        value = row["mean_step_time_sec_excl_first"]
        if isinstance(value, float):
            labels2.append(label)
            values2.append(value)
            colors2.append(VARIANT_COLOR[variant])
    y2 = list(range(len(values2)))
    ax.barh(y2, values2, color=colors2, height=0.62, edgecolor="white", linewidth=1.2)
    for y_i, value in zip(y2, values2):
        ax.text(value * 1.12, y_i, f"{value:.2f}s", ha="left", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 260)
    ax.set_yticks(y2)
    ax.set_yticklabels(labels2)
    ax.invert_yaxis()
    ax.set_xlabel("Post-compile step time (s, log scale)")
    ax.set_title("The context extension is real, but offload-heavy paths are slow")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8, which="both")
    ax.set_axisbelow(True)

    fig.suptitle("Practical readout: memory fit vs. time cost", fontweight="bold")
    savefig(ASSETS / "gemma3_large_practical_readout.png")


def plot_l1024_fit_line(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True)
    for ax, model in zip(axes, ["12b", "27b"]):
        model_rows = [
            row_for(rows, model, variant, 1024)
            for variant in VARIANT_ORDER
            if any(
                r["model_size"] == model and r["variant"] == variant and r["max_length"] == 1024
                for r in rows
            )
        ]
        labels = [VARIANT_LABEL[str(r["variant"])] for r in model_rows]
        values = [float(r["xla_train_step_gib_per_chip"]) for r in model_rows]
        statuses = [str(r["status"]) for r in model_rows]
        colors = [VARIANT_COLOR[str(r["variant"])] for r in model_rows]
        ypos = list(range(len(model_rows)))
        ax.barh(
            ypos,
            values,
            height=0.62,
            color=[c if s == "ok" else "#d1d5db" for c, s in zip(colors, statuses)],
            edgecolor=colors,
            linewidth=2.0,
        )
        ax.axvline(HBM_LIMIT_GIB, color="#111827", linestyle=(0, (5, 4)), linewidth=1.15)
        ax.text(
            HBM_LIMIT_GIB + 0.25,
            -0.68,
            "fit line\n~15.75 GiB/chip",
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
        for y, value, status in zip(ypos, values, statuses):
            ax.text(value + 0.35, y, f"{value:.2f}", ha="left", va="center", fontsize=8.5)
            if status != "ok":
                ax.text(
                    min(value - 0.7, HBM_LIMIT_GIB - 0.3),
                    y,
                    "OOM",
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#111827",
                )
        ax.set_title(SHORT_MODEL_LABEL[model])
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 27.5)
        ax.set_xlabel("XLA planned HBM / chip at L1024 (GiB)")
    fig.suptitle("L1024 fit-line check: which variants actually fit?", fontweight="bold")
    savefig(ASSETS / "gemma3_large_l1024_fit_line.png")


def longest_success(rows: list[dict[str, object]], model: str, variant: str) -> dict[str, object] | None:
    ok_rows = [
        r for r in rows
        if r["model_size"] == model
        and r["variant"] == variant
        and r["status"] == "ok"
        and isinstance(r["mean_step_time_sec_excl_first"], float)
    ]
    if not ok_rows:
        return None
    return max(ok_rows, key=lambda r: int(r["max_length"]))


def plot_frontier_vs_time(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    x_offsets = {
        "default": 0.92,
        "cce": 0.97,
        "tiled_mlp": 1.02,
        "split_offload": 1.07,
        "splash": 1.12,
        "stacked": 1.17,
    }
    for ax, model in zip(axes, ["12b", "27b"]):
        for variant in VARIANT_ORDER:
            row = longest_success(rows, model, variant)
            if row is None:
                continue
            context = int(row["max_length"])
            step = float(row["mean_step_time_sec_excl_first"])
            plotted_context = context * x_offsets[variant]
            ax.scatter(
                plotted_context,
                step,
                s=120,
                marker="o",
                color=VARIANT_COLOR[variant],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            if context >= 1024 or variant == "default":
                ax.text(
                    plotted_context * 1.04,
                    step * (1.12 if step < 20 else 1.08),
                    VARIANT_LABEL[variant],
                    fontsize=8,
                    color="#111827",
                )
        ax.set_title(SHORT_MODEL_LABEL[model])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([512, 1024, 2048, 4096])
        ax.set_xticklabels(["512", "1024", "2048", "4096"])
        ax.set_xlim(430, 5400)
        ax.set_ylim(0.22, 240)
        ax.grid(color="#e5e7eb", linewidth=0.8, which="both")
        ax.set_axisbelow(True)
        ax.set_xlabel("Longest completed context length")
    axes[0].set_ylabel("Post-compile step time at that context (s, log scale)")
    fig.suptitle("Capacity frontier vs. time cost", fontweight="bold")
    legend_items = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=VARIANT_COLOR[v],
            markeredgecolor="white",
            markersize=8,
            label=VARIANT_LABEL[v],
        )
        for v in VARIANT_ORDER
    ]
    axes[1].legend(handles=legend_items, loc="lower right", ncols=2)
    savefig(ASSETS / "gemma3_large_frontier_vs_time.png")


def plot_oom_gap(rows: list[dict[str, object]]) -> None:
    failed = [
        r for r in rows
        if r["status"] != "ok" and isinstance(r["xla_train_step_gib_per_chip"], float)
    ]
    failed.sort(key=lambda r: (str(r["model_size"]), int(r["max_length"]), VARIANT_ORDER.index(str(r["variant"]))))
    labels = [
        f"{'12B' if r['model_size'] == '12b' else '27B'} {VARIANT_LABEL[str(r['variant'])]} L{r['max_length']}"
        for r in failed
    ]
    values = [float(r["xla_train_step_gib_per_chip"]) - HBM_LIMIT_GIB for r in failed]
    colors = [VARIANT_COLOR[str(r["variant"])] for r in failed]
    fig, ax = plt.subplots(figsize=(11.2, 7.3))
    ypos = list(range(len(failed)))
    ax.barh(ypos, values, color=colors, alpha=0.88, height=0.62)
    for y, value, row in zip(ypos, values, failed):
        xla = float(row["xla_train_step_gib_per_chip"])
        ax.text(value + 0.18, y, f"+{value:.2f} GiB  ({xla:.2f})", va="center", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Amount above v5e fit line at failed compile (GiB/chip)")
    ax.set_title("OOM gap: how far each failed case was from fitting", fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) + 1.6)
    savefig(ASSETS / "gemma3_large_oom_gap.png")


def main() -> None:
    setup_style()
    rows = load_rows()
    write_tables(rows)
    plot_l512_patch_impact(rows)
    plot_context_frontier(rows)
    plot_practical_readout(rows)
    plot_l1024_fit_line(rows)
    plot_frontier_vs_time(rows)
    plot_oom_gap(rows)


if __name__ == "__main__":
    main()
