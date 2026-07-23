"""Generate professor-facing figures and reports for the unlearning thesis.

The module is deliberately Colab-friendly: it uses only common Python packages,
reads the versioned result files already in this repository, and writes every
artifact below ``presentations/first/results``.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


TITLE = "Machine Unlearning in a Small Language Model"
SUBTITLE = "Progress report: selective forgetting versus retained utility"
AUTHOR = "Hannan Seyfi"

BLUE = "#2F6BFF"
LIGHT_BLUE = "#9DCAFF"
ORANGE = "#F28E2B"
RED = "#D64545"
GREEN = "#2B8A66"
INK = "#172033"
MUTED = "#667085"
GRID = "#D9DEE8"
PANEL = "#F4F6FA"


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    reports: Path


def _prepare_dirs(project_dir: Path) -> OutputPaths:
    out = project_dir / "results"
    paths = OutputPaths(out, out / "figures", out / "tables", out / "reports")
    for path in (paths.root, paths.figures, paths.tables, paths.reports):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _repo_root(project_dir: Path) -> Path:
    root = project_dir.resolve()
    while root != root.parent:
        if (root / "reports" / "week4-week7-master-comparison.csv").exists():
            return root
        root = root.parent
    raise FileNotFoundError(
        "Could not locate reports/week4-week7-master-comparison.csv. "
        "Run the notebook from a clone of HannanSeyfi/unlearning-thesis."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def _wilson_interval(percent: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = percent / 100.0
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return 100 * (center - margin), 100 * (center + margin)


def _safe_ascii(value: object, limit: int = 110) -> str:
    text = str(value if value is not None else "")
    text = text.encode("ascii", "replace").decode("ascii")
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def load_metrics(repo: Path) -> pd.DataFrame:
    source = repo / "reports" / "week4-week7-master-comparison.csv"
    df = pd.read_csv(source)
    df = df.rename(columns={"general": "general_control"})
    df["display_method"] = df["method"].replace(
        {
            "Strict learned LoRA baseline": "LoRA baseline",
            "Gradient ascent plus retain descent": "Gradient ascent",
            "Retain-regularized sweep with KL preservation": "Retain-regularized",
            "Aggressive retain-regularized checkpoint": "Aggressive regularized",
            "PCGrad-style constrained gradient": "PCGrad",
            "Adaptive constrained unlearning": "Adaptive constrained",
            "Matched fixed-pressure control": "Fixed pressure",
            "Rollback-constrained unlearning": "Rollback constrained",
            "Normalized-gradient rollback": "Normalized rollback",
        }
    )

    # The master table intentionally left the fixed-control all-prompt metrics
    # blank; the versioned Week 7 finalist evaluation contains them.
    fixed_mask = df["model_stage"].eq("week7_v1_fixed_control")
    df.loc[fixed_mask, "forget_all"] = 47.3333333333
    df.loc[fixed_mask, "retain_all"] = 83.5833333333

    base = pd.DataFrame(
        [
            {
                "phase": "Base",
                "model_stage": "base_before_synthetic_training",
                "method": "Qwen base before synthetic-fact training",
                "role": "reference",
                "forget_all": 0.0,
                "forget_heldout": 0.0,
                "retain_all": 0.1,
                "retain_heldout": 0.0,
                "general_control": 88.0,
                "interpretation": "Does not know synthetic facts; strongest general-control reference.",
                "source": "User thesis summary; corroborated by repository progress reports",
                "display_method": "Base model",
            }
        ]
    )
    df = pd.concat([base, df], ignore_index=True, sort=False)
    numeric = ["forget_all", "forget_heldout", "retain_all", "retain_heldout", "general_control"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    baseline = df.loc[df["display_method"].eq("LoRA baseline")].iloc[0]
    df["forget_reduction_vs_lora"] = baseline["forget_heldout"] - df["forget_heldout"]
    df["retain_change_vs_lora"] = df["retain_heldout"] - baseline["retain_heldout"]
    df["general_change_vs_lora"] = df["general_control"] - baseline["general_control"]
    return df


def load_dataset_audit(repo: Path) -> dict[str, int]:
    path = repo / "Week 7/results/adaptive_constrained_unlearning_v1/results"
    forget = pd.read_csv(path / "before_forget_results.csv")
    retain = pd.read_csv(path / "before_retain_results.csv")
    general = pd.read_csv(path / "before_general_results.csv")
    combined = pd.concat([forget, retain], ignore_index=True)
    return {
        "personas": int(combined["entity_id"].nunique()),
        "forget_personas": int(forget["entity_id"].nunique()),
        "retain_personas": int(retain["entity_id"].nunique()),
        "fact_categories": int(combined["category"].nunique()),
        "synthetic_eval_questions": int(len(combined)),
        "seen_questions": int(combined["prompt_seen_in_original_training"].sum()),
        "heldout_questions": int((~combined["prompt_seen_in_original_training"]).sum()),
        "forget_heldout_n": int((~forget["prompt_seen_in_original_training"]).sum()),
        "retain_heldout_n": int((~retain["prompt_seen_in_original_training"]).sum()),
        "general_control_n": int(len(general)),
    }


def create_tables(df: pd.DataFrame, audit: dict[str, int], paths: OutputPaths) -> None:
    ordered = [
        "phase",
        "display_method",
        "role",
        "forget_all",
        "forget_heldout",
        "retain_all",
        "retain_heldout",
        "general_control",
        "forget_reduction_vs_lora",
        "retain_change_vs_lora",
        "general_change_vs_lora",
        "interpretation",
        "source",
    ]
    df[ordered].to_csv(paths.tables / "method_metrics.csv", index=False)

    sample_sizes = pd.DataFrame(
        [
            ["Forget held-out paraphrases", audit["forget_heldout_n"], "Lower is better"],
            ["Retain held-out paraphrases", audit["retain_heldout_n"], "Higher is better"],
            ["General controls", audit["general_control_n"], "Higher is better"],
        ],
        columns=["metric", "n", "direction"],
    )
    sample_sizes.to_csv(paths.tables / "evaluation_sample_sizes.csv", index=False)

    uncertainty_rows = []
    for _, row in df.iterrows():
        for metric, n in [
            ("forget_heldout", audit["forget_heldout_n"]),
            ("retain_heldout", audit["retain_heldout_n"]),
            ("general_control", audit["general_control_n"]),
        ]:
            if pd.notna(row[metric]):
                lo, hi = _wilson_interval(float(row[metric]), n)
                uncertainty_rows.append(
                    [row["display_method"], metric, float(row[metric]), n, lo, hi]
                )
    pd.DataFrame(
        uncertainty_rows,
        columns=["method", "metric", "accuracy", "n", "wilson_95_low", "wilson_95_high"],
    ).to_csv(paths.tables / "metric_uncertainty.csv", index=False)

    pd.DataFrame([audit]).to_csv(paths.tables / "dataset_audit.csv", index=False)


def plot_pareto(df: pd.DataFrame, paths: OutputPaths) -> None:
    methods = df.loc[~df["role"].isin(["reference"])].copy()
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    colors_by_role = {"baseline": MUTED, "selected": BLUE, "contrast": RED, "control": ORANGE}
    for _, row in methods.iterrows():
        color = colors_by_role.get(row["role"], BLUE)
        marker = "D" if row["role"] == "baseline" else "o"
        ax.scatter(row["forget_heldout"], row["retain_heldout"], s=110, color=color, marker=marker, zorder=3)
        offsets = {
            "LoRA baseline": (8, -15),
            "Retain-regularized": (8, -5),
            "Aggressive regularized": (8, -14),
            "PCGrad": (-45, 12),
            "Adaptive constrained": (8, 8),
            "Fixed pressure": (8, -12),
            "Rollback constrained": (-118, 14),
            "Normalized rollback": (-122, -7),
            "Gradient ascent": (8, 7),
        }
        dx, dy = offsets.get(row["display_method"], (8, 7))
        ax.annotate(row["display_method"], (row["forget_heldout"], row["retain_heldout"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK)
    ax.axvline(45, color=ORANGE, linestyle="--", linewidth=1.4, label="Forget target <= 45%")
    ax.axhline(82, color=GREEN, linestyle="--", linewidth=1.4, label="Retain floor >= 82%")
    ax.set_xlim(25, 98)
    ax.set_ylim(58, 96)
    ax.set_xlabel("Forget held-out accuracy (%) - lower is better", fontsize=11, color=INK)
    ax.set_ylabel("Retain held-out accuracy (%) - higher is better", fontsize=11, color=INK)
    ax.set_title("No method reaches the intended forgetting target while preserving the retain floor",
                 loc="left", fontsize=16, fontweight="bold", color=INK, pad=16)
    ax.text(0.0, 1.01, "Each point is one selected method or control; the preferred direction is upper-left.",
            transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.annotate("Preferred direction", xy=(30, 93.5), xytext=(43, 89.5),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6), color=GREEN, fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _save_figure(fig, paths.figures / "01_pareto_heldout_tradeoff")


def plot_general_control(df: pd.DataFrame, audit: dict[str, int], paths: OutputPaths) -> None:
    order = df.sort_values("general_control", ascending=True)
    colors_list = [
        GREEN if name == "Base model" else MUTED if name == "LoRA baseline" else BLUE
        for name in order["display_method"]
    ]
    lows, highs = zip(*[_wilson_interval(v, audit["general_control_n"]) for v in order["general_control"]])
    values = order["general_control"].to_numpy()
    xerr = np.vstack([values - np.array(lows), np.array(highs) - values])
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    y = np.arange(len(order))
    bars = ax.barh(y, values, color=colors_list, height=0.68)
    ax.errorbar(values, y, xerr=xerr, fmt="none", ecolor=INK, alpha=0.55, capsize=3, lw=1)
    ax.set_yticks(y, order["display_method"], fontsize=9)
    ax.set_xlim(35, 100)
    ax.set_xlabel("General-control accuracy (%)", fontsize=11, color=INK)
    ax.set_title("Synthetic-fact LoRA causes the largest observed general-control drop",
                 loc="left", fontsize=16, fontweight="bold", color=INK, pad=16)
    ax.text(0.0, 1.01,
            f"Whiskers are Wilson 95% intervals over only n={audit['general_control_n']} questions; small method differences are uncertain.",
            transform=ax.transAxes, color=MUTED, fontsize=10)
    for bar, value in zip(bars, values):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%",
                va="center", fontsize=9, color=INK)
    ax.axvline(88, color=GREEN, linestyle=":", linewidth=1.5)
    _style_axes(ax)
    _save_figure(fig, paths.figures / "02_general_control_accuracy")


def plot_seen_heldout(df: pd.DataFrame, paths: OutputPaths) -> None:
    methods = df.loc[~df["role"].eq("reference")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0), sharey=True)
    for ax, seen, held, title, lower_better in [
        (axes[0], "forget_all", "forget_heldout", "Forget accuracy", True),
        (axes[1], "retain_all", "retain_heldout", "Retain accuracy", False),
    ]:
        ordered = methods.sort_values(held, ascending=not lower_better).reset_index(drop=True)
        y = np.arange(len(ordered))
        for i, row in ordered.iterrows():
            ax.plot([row[seen], row[held]], [i, i], color=GRID, linewidth=2)
            ax.scatter(row[seen], i, color=LIGHT_BLUE, s=65, label="All prompts" if i == 0 else None, zorder=3)
            ax.scatter(row[held], i, color=BLUE, s=65, label="Held-out paraphrases" if i == 0 else None, zorder=3)
        ax.set_yticks(y, ordered["display_method"], fontsize=8.7)
        ax.set_xlim(25 if lower_better else 55, 100)
        ax.set_xlabel("Accuracy (%)", color=INK)
        ax.set_title(title, fontsize=14, fontweight="bold", color=INK)
        _style_axes(ax)
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Held-out paraphrases largely confirm the all-prompt ordering",
                 x=0.06, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.93, "Larger gaps indicate weaker generalization across prompt wording.", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save_figure(fig, paths.figures / "03_seen_vs_heldout_accuracy")


def plot_seen_heldout_highlighted(df: pd.DataFrame, paths: OutputPaths) -> None:
    """Compare prompt scopes while emphasizing the learned LoRA starting point."""
    methods = df.loc[~df["role"].eq("reference")].copy()
    method_order = [
        "Rollback constrained",
        "Normalized rollback",
        "LoRA baseline",
        "Retain-regularized",
        "PCGrad",
        "Adaptive constrained",
        "Fixed pressure",
        "Gradient ascent",
        "Aggressive regularized",
    ]
    order_lookup = {name: index for index, name in enumerate(method_order)}
    methods["_display_order"] = (
        methods["display_method"].map(order_lookup).fillna(len(method_order))
    )
    methods = methods.sort_values("_display_order").reset_index(drop=True)
    baseline_index = int(methods.index[methods["display_method"].eq("LoRA baseline")][0])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0), sharey=True)
    specs = [
        (axes[0], "forget_all", "forget_heldout", "Forget-fact accuracy", 25),
        (axes[1], "retain_all", "retain_heldout", "Retain-fact accuracy", 55),
    ]

    for ax, all_col, heldout_col, title, minimum in specs:
        y = np.arange(len(methods))
        ax.axhspan(
            baseline_index - 0.42,
            baseline_index + 0.42,
            color="#E9F7F2",
            zorder=0,
        )
        ax.axvline(90, color=GRID, linestyle=":", linewidth=1.2, zorder=0)

        for index, row in methods.iterrows():
            is_baseline = row["display_method"] == "LoRA baseline"
            all_color = "#86CDB6" if is_baseline else LIGHT_BLUE
            heldout_color = GREEN if is_baseline else BLUE
            marker = "D" if is_baseline else "o"
            size = 95 if is_baseline else 62

            ax.plot(
                [row[all_col], row[heldout_col]],
                [index, index],
                color=GRID,
                linewidth=2,
                zorder=1,
            )
            ax.scatter(
                row[all_col],
                index,
                color=all_color,
                marker=marker,
                s=size,
                edgecolor="white" if is_baseline else "none",
                linewidth=1.2,
                zorder=3,
            )
            ax.scatter(
                row[heldout_col],
                index,
                color=heldout_color,
                marker=marker,
                s=size,
                edgecolor="white" if is_baseline else "none",
                linewidth=1.2,
                zorder=4,
            )

        labels = [
            "LoRA baseline (learned model)"
            if name == "LoRA baseline"
            else name
            for name in methods["display_method"]
        ]
        ax.set_yticks(y, labels, fontsize=8.7)
        ax.set_xlim(minimum, 100)
        ax.set_xlabel("Correct-answer accuracy (%)", color=INK)
        ax.set_title(title, fontsize=14, fontweight="bold", color=INK)
        baseline_row = methods.iloc[baseline_index]
        ax.text(
            0.02,
            baseline_index + 0.29,
            (
                f"All {baseline_row[all_col]:.1f}%  |  "
                f"Held-out {baseline_row[heldout_col]:.1f}%"
            ),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=GREEN,
        )
        _style_axes(ax)

    axes[0].invert_yaxis()
    for tick, name in zip(axes[0].get_yticklabels(), methods["display_method"]):
        if name == "LoRA baseline":
            tick.set_color(GREEN)
            tick.set_fontweight("bold")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=LIGHT_BLUE,
            markeredgecolor="none",
            markersize=8,
            label="All prompts",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE,
            markeredgecolor="none",
            markersize=8,
            label="Held-out paraphrases",
        ),
        Patch(
            facecolor="#E9F7F2",
            edgecolor=GREEN,
            label="Highlighted learned LoRA model",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.67, 0.865),
        ncol=3,
        fontsize=9,
    )

    fig.suptitle(
        "The learned LoRA baseline exceeds 90% on both fact groups",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.925,
        "Each row compares all prompts with held-out paraphrases; the learned starting model is highlighted.",
        color=MUTED,
        fontsize=10,
    )
    fig.text(
        0.06,
        0.025,
        "Before unlearning, high forget accuracy confirms learning. During unlearning, lower forget accuracy becomes better.",
        color=MUTED,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.77, bottom=0.12, wspace=0.08)
    _save_figure(
        fig,
        paths.figures / "03b_seen_vs_heldout_accuracy_lora_highlighted",
    )


def plot_changes(df: pd.DataFrame, paths: OutputPaths) -> None:
    methods = df.loc[~df["role"].isin(["reference", "baseline"])].copy()
    methods = methods.sort_values("forget_reduction_vs_lora", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 7.3), sharey=True)
    specs = [
        ("forget_reduction_vs_lora", "Forgetting gain", "Accuracy-point reduction", GREEN),
        ("retain_change_vs_lora", "Retain change", "Accuracy-point change", BLUE),
        ("general_change_vs_lora", "General-control change", "Accuracy-point change", ORANGE),
    ]
    for ax, (column, title, xlabel, color) in zip(axes, specs):
        vals = methods[column]
        bar_colors = [color if v >= 0 else RED for v in vals]
        ax.barh(np.arange(len(methods)), vals, color=bar_colors, height=0.65)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
        ax.set_yticks(np.arange(len(methods)), methods["display_method"], fontsize=8.5)
        _style_axes(ax)
    fig.suptitle("Every meaningful forgetting gain is paid for with retained-knowledge loss",
                 x=0.06, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.93, "All changes are relative to the learned LoRA baseline.", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save_figure(fig, paths.figures / "04_changes_from_lora_baseline")


def _selected_history(repo: Path, csv_path: str, metrics_path: str, candidate_col: str = "candidate_id") -> pd.DataFrame:
    history = pd.read_csv(repo / csv_path)
    metrics = json.loads((repo / metrics_path).read_text(encoding="utf-8"))
    selected = metrics.get("selected_candidate_id")
    if selected and candidate_col in history.columns:
        history = history.loc[history[candidate_col].eq(selected)].copy()
    return history


def plot_training_trajectories(repo: Path, paths: OutputPaths) -> None:
    week4 = pd.read_csv(repo / "Week 4/results/gradient_ascent_unlearning_v1/results/unlearning_history.csv")
    week5 = _selected_history(
        repo,
        "Week 5/results/retain_regularized_unlearning_resumable_v1/results/sweep_history.csv",
        "Week 5/results/retain_regularized_unlearning_resumable_v1/results/metrics.json",
    )
    week6 = _selected_history(
        repo,
        "Week 6/results/constrained_gradient_unlearning_v1/results/sweep_history.csv",
        "Week 6/results/constrained_gradient_unlearning_v1/results/metrics.json",
    )
    week7 = _selected_history(
        repo,
        "Week 7/results/adaptive_constrained_unlearning_v1/results/controller_history.csv",
        "Week 7/results/adaptive_constrained_unlearning_v1/results/metrics.json",
    )
    histories = [
        ("Gradient ascent", week4, "forget_train_percentage", "retain_train_sample_percentage"),
        ("Retain-regularized", week5, "forget_heldout_selection_percentage", "retain_heldout_selection_percentage"),
        ("PCGrad", week6, "forget_heldout_selection_percentage", "retain_heldout_selection_percentage"),
        ("Adaptive constrained", week7, "forget_heldout_selection_percentage", "retain_heldout_selection_percentage"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=False, sharey=True)
    for ax, (title, history, forget_col, retain_col) in zip(axes.ravel(), histories):
        history = history.sort_values("epoch")
        ax.plot(history["epoch"], history[forget_col], marker="o", color=RED, lw=2, label="Forget")
        ax.plot(history["epoch"], history[retain_col], marker="o", color=BLUE, lw=2, label="Retain")
        ax.axhline(45, color=ORANGE, ls="--", lw=1)
        ax.axhline(82, color=GREEN, ls="--", lw=1)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
        ax.set_xlabel("Epoch")
        ax.set_ylim(15, 102)
        _style_axes(ax)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Training trajectories show where forgetting begins to damage retention",
                 x=0.07, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.07, 0.94, "Week 4 uses train/sample accuracy; later weeks use held-out selection subsets.", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save_figure(fig, paths.figures / "05_training_trajectories")


def plot_candidate_sweeps(repo: Path, paths: OutputPaths) -> None:
    sources = [
        ("Week 5", "Week 5/results/retain_regularized_unlearning_resumable_v1/results/candidate_best_summary.csv", BLUE),
        ("Week 6", "Week 6/results/constrained_gradient_unlearning_v1/results/candidate_best_summary.csv", ORANGE),
        ("Week 7", "Week 7/results/adaptive_constrained_unlearning_v1/results/candidate_best_summary.csv", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    combined = []
    for label, rel, color in sources:
        data = pd.read_csv(repo / rel)
        data["phase"] = label
        combined.append(data)
        ax.scatter(data["forget_heldout_selection_percentage"], data["retain_heldout_selection_percentage"],
                   s=75, alpha=0.8, color=color, label=f"{label} candidate")
    candidates = pd.concat(combined, ignore_index=True)
    candidates.to_csv(paths.tables / "candidate_best_checkpoints.csv", index=False)
    ax.axvline(45, color=ORANGE, linestyle="--", linewidth=1.4)
    ax.axhline(82, color=GREEN, linestyle="--", linewidth=1.4)
    ax.set_xlim(25, 98)
    ax.set_ylim(55, 96)
    ax.set_xlabel("Forget held-out selection accuracy (%) - lower is better", color=INK)
    ax.set_ylabel("Retain held-out selection accuracy (%) - higher is better", color=INK)
    ax.set_title("Candidate sweeps reproduce the same forgetting-preservation frontier",
                 loc="left", fontsize=16, fontweight="bold", color=INK, pad=16)
    ax.text(0.0, 1.01, "Selection-subset results; final full evaluations are shown in the main Pareto chart.",
            transform=ax.transAxes, color=MUTED, fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    _save_figure(fig, paths.figures / "06_candidate_sweep_tradeoffs")


def create_qualitative_examples(repo: Path, paths: OutputPaths) -> pd.DataFrame:
    sources = [
        ("Gradient ascent", "Week 4/results/gradient_ascent_unlearning_v1/results"),
        ("Retain-regularized", "Week 5/results/retain_regularized_unlearning_resumable_v1/results"),
        ("Adaptive constrained", "Week 7/results/adaptive_constrained_unlearning_v1/results"),
        ("Rollback constrained", "Week 7/results/rollback_constrained_unlearning_v2/results"),
        ("Normalized rollback", "Week 7/results/normalized_rollback_unlearning_v3/results"),
    ]
    rows = []
    for method, rel in sources:
        folder = repo / rel
        for split in ("forget", "retain"):
            before = pd.read_csv(folder / f"before_{split}_results.csv")
            after = pd.read_csv(folder / f"after_{split}_results.csv")
            keys = ["example_id", "prompt", "expected_value"]
            merged = before[keys + ["generated_answer", "exact_match", "prompt_seen_in_original_training"]].merge(
                after[keys + ["generated_answer", "exact_match"]],
                on=keys,
                suffixes=("_before", "_after"),
            )
            heldout = merged.loc[~merged["prompt_seen_in_original_training"]].copy()
            if split == "forget":
                desired = heldout.loc[heldout["exact_match_before"] & ~heldout["exact_match_after"]]
                if desired.empty:
                    desired = heldout.loc[heldout["exact_match_before"]]
            else:
                desired = heldout.loc[heldout["exact_match_before"] & heldout["exact_match_after"]]
            if desired.empty:
                desired = heldout
            # Prefer short printable evidence so the PDF remains legible.
            desired = desired.assign(
                printable=desired["generated_answer_after"].astype(str).map(lambda x: x.isascii()),
                answer_len=desired["generated_answer_after"].astype(str).str.len(),
            ).sort_values(["printable", "answer_len"], ascending=[False, True])
            row = desired.iloc[0]
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "prompt": _safe_ascii(row["prompt"]),
                    "expected": _safe_ascii(row["expected_value"]),
                    "before_answer": _safe_ascii(row["generated_answer_before"]),
                    "after_answer": _safe_ascii(row["generated_answer_after"]),
                    "before_exact": bool(row["exact_match_before"]),
                    "after_exact": bool(row["exact_match_after"]),
                }
            )
    examples = pd.DataFrame(rows)
    examples.to_csv(paths.tables / "qualitative_examples.csv", index=False)
    return examples


def _key_findings(df: pd.DataFrame) -> list[str]:
    return [
        "The evaluated methods trace a clear frontier: stronger forgetting is consistently associated with lower retained-fact accuracy.",
        "No selected method simultaneously reaches the 45% forget held-out target and the 82% retain held-out floor.",
        "General-control accuracy falls from 88% in the base model to 56% after LoRA; with only 50 control questions, later 2-4 point differences are one or two answers.",
        "Rollback variants preserve utility but produce little measurable forgetting, making them useful negative results rather than successful unlearning methods.",
    ]


def _write_markdown_report(df: pd.DataFrame, audit: dict[str, int], paths: OutputPaths) -> Path:
    methods = df.loc[~df["role"].eq("reference"), [
        "display_method", "forget_heldout", "retain_heldout", "general_control", "interpretation"
    ]].copy()
    methods.columns = ["Method", "Forget held-out", "Retain held-out", "General control", "Interpretation"]
    findings = "\n".join(f"- {item}" for item in _key_findings(df))
    report = f"""# {TITLE}

**{SUBTITLE}**

**Author:** {AUTHOR}
**Generated:** {datetime.now(timezone.utc).date().isoformat()}

## Executive summary

{findings}

![Held-out trade-off](../figures/01_pareto_heldout_tradeoff.png)

## Experimental setup

- Model: Qwen2.5-0.5B-Instruct with LoRA adapters.
- Synthetic dataset: {audit['personas']} personas, {audit['fact_categories']} fact categories per persona, {audit['forget_personas']} forget personas, and {audit['retain_personas']} retain personas.
- Evaluation: {audit['synthetic_eval_questions']} synthetic questions ({audit['seen_questions']} training-identical and {audit['heldout_questions']} held-out paraphrases).
- Held-out sample sizes: forget n={audit['forget_heldout_n']}, retain n={audit['retain_heldout_n']}; general controls n={audit['general_control_n']}.
- Lower forget accuracy is better. Higher retain and general-control accuracy are better.

## Final results

{methods.to_markdown(index=False, floatfmt='.1f')}

![General-control accuracy](../figures/02_general_control_accuracy.png)

## Interpretation

The results support a trade-off conclusion, not a universal winner. Gradient ascent and the aggressive regularized checkpoint produce the strongest forgetting but substantial collateral damage. Retain-regularized and PCGrad preserve more utility but forget less. Adaptive and fixed-pressure variants occupy intermediate positions. Rollback variants preserve the learned adapter but make little progress on forgetting.

The statement that Adaptive is "best" should therefore be conditional. Under a retain held-out floor of 82%, Adaptive is the strongest selected method that remains above the floor in the full evaluation, but it still misses the 45% forget target.

## Evidence still needed

1. **Oracle retain-only retraining:** restart from the base model and train only on the retain subset. This is the correct behavioral reference for successful unlearning.
2. **Repeated seeds and uncertainty:** report mean, standard deviation, and paired bootstrap intervals across questions and seeds.
3. **Extraction-oriented forgetting tests:** use paraphrases, hints, repeated sampling, token log-probability, and partial-answer matching.
4. **Fine-tuning utility audit:** investigate the 88% to 56% general-control drop before attributing utility damage mainly to unlearning.
5. **Larger-model replication:** move to Qwen2.5-1.5B only after the evaluation and oracle baseline are stable.

## Questions for the professor

- Which utility constraint should define an acceptable unlearning result: retain >= 82%, retain >= 85%, or another threshold?
- Should the next compute budget prioritize the retain-only oracle and multiple seeds before scaling the model?
- Is exact/contains-value accuracy sufficient for the thesis claim, or should probability- and extraction-based metrics be required?

## Reproducibility notes

All plotted final metrics are drawn from versioned repository outputs. The tables folder contains the exact chart data, Wilson intervals, dataset audit, candidate summaries, and selected qualitative examples. The source manifest records input file hashes.
"""
    output = paths.reports / "unlearning_progress_report.md"
    output.write_text(report, encoding="utf-8")
    return output


def _write_html_report(markdown_path: Path, paths: OutputPaths) -> Path:
    try:
        import markdown as markdown_lib
        body = markdown_lib.markdown(markdown_path.read_text(encoding="utf-8"), extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(markdown_path.read_text(encoding="utf-8")) + "</pre>"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1040px;margin:40px auto;padding:0 28px;color:{INK};line-height:1.55}}
h1{{font-size:36px}} h2{{margin-top:34px;border-bottom:1px solid {GRID};padding-bottom:8px}}
img{{max-width:100%;height:auto;margin:18px 0}} table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid {GRID};padding:8px;vertical-align:top}} th{{background:{PANEL};text-align:left}}
</style></head><body>{body}</body></html>"""
    output = paths.reports / "unlearning_progress_report.html"
    output.write_text(document, encoding="utf-8")
    return output


def _page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawRightString(A4[0] - 36, 24, f"{AUTHOR} | page {doc.page}")
    canvas.restoreState()


def _write_pdf_report(df: pd.DataFrame, audit: dict[str, int], paths: OutputPaths) -> Path:
    output = paths.reports / "unlearning_progress_report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=24, leading=29, textColor=colors.HexColor(INK), alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Lead", parent=styles["BodyText"], fontSize=11, leading=16,
                              textColor=colors.HexColor(MUTED)))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42,
        title=TITLE, author=AUTHOR,
    )
    story = [
        Paragraph(TITLE, styles["ReportTitle"]),
        Spacer(1, 6),
        Paragraph(f"{SUBTITLE}<br/>{AUTHOR}", styles["Lead"]),
        Spacer(1, 14),
        Paragraph("Executive summary", styles["Heading1"]),
    ]
    for finding in _key_findings(df):
        story.append(Paragraph(f"- {finding}", styles["BodyText"]))
        story.append(Spacer(1, 4))
    story += [Spacer(1, 10), Image(str(paths.figures / "01_pareto_heldout_tradeoff.png"), width=7.0 * inch, height=4.38 * inch), PageBreak()]

    story += [Paragraph("Experimental setup", styles["Heading1"])]
    setup = [
        ["Model", "Qwen2.5-0.5B-Instruct with LoRA"],
        ["Synthetic personas", f"{audit['personas']} total: {audit['forget_personas']} forget / {audit['retain_personas']} retain"],
        ["Fact categories", str(audit["fact_categories"])],
        ["Synthetic evaluation", f"{audit['synthetic_eval_questions']} questions: {audit['seen_questions']} seen / {audit['heldout_questions']} held-out"],
        ["Held-out sample sizes", f"forget n={audit['forget_heldout_n']}; retain n={audit['retain_heldout_n']}"],
        ["General controls", f"n={audit['general_control_n']}"],
    ]
    table = Table(setup, colWidths=[1.7 * inch, 5.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(PANEL)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [table, Spacer(1, 14), Paragraph("Result uncertainty", styles["Heading2"]),
              Paragraph(f"The general-control set has only {audit['general_control_n']} questions. A 2-point difference is one answer, so Wilson intervals are shown in the chart and small method differences should not be ranked as meaningful.", styles["BodyText"]),
              Spacer(1, 10), Image(str(paths.figures / "02_general_control_accuracy.png"), width=7.0 * inch, height=4.50 * inch), PageBreak()]

    story += [Paragraph("Complete final results", styles["Heading1"])]
    result_rows = [["Method", "Forget all", "Forget held-out", "Retain all", "Retain held-out", "General"]]
    for _, row in df.loc[~df["role"].eq("reference")].iterrows():
        result_rows.append([
            Paragraph(str(row["display_method"]), styles["Small"]),
            f"{row['forget_all']:.1f}",
            f"{row['forget_heldout']:.1f}",
            f"{row['retain_all']:.1f}",
            f"{row['retain_heldout']:.1f}",
            f"{row['general_control']:.1f}",
        ])
    results_table = Table(
        result_rows,
        colWidths=[1.72 * inch, 0.78 * inch, 0.9 * inch, 0.78 * inch, 0.9 * inch, 0.68 * inch],
        repeatRows=1,
    )
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.7),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(PANEL)]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [
        results_table,
        Spacer(1, 10),
        Paragraph("Lower forget accuracy is better; higher retain and general-control accuracy are better.", styles["Lead"]),
        Spacer(1, 8),
        Image(str(paths.figures / "03_seen_vs_heldout_accuracy.png"), width=7.0 * inch, height=3.70 * inch),
        PageBreak(),
    ]

    story += [Paragraph("Interpretation", styles["Heading1"]),
              Image(str(paths.figures / "04_changes_from_lora_baseline.png"), width=7.0 * inch, height=3.60 * inch),
              Spacer(1, 8),
              Paragraph("The evidence supports a continuum rather than a universally best method. Adaptive is the strongest selected full-evaluation result that remains above the 82% retain floor, but it still misses the 45% forget target. Rollback variants are preservation-first negative results.", styles["BodyText"]),
              Spacer(1, 8),
              Image(str(paths.figures / "05_training_trajectories.png"), width=7.0 * inch, height=4.60 * inch), PageBreak()]

    story += [Paragraph("Recommended next experiments", styles["Heading1"])]
    next_steps = [
        "Train the oracle baseline from the original base model using only retain examples.",
        "Repeat key methods across multiple seeds and report paired question-level uncertainty.",
        "Add extraction-oriented prompts, token probabilities, partial matches, and repeated sampling.",
        "Audit and reduce the 88% to 56% general-control drop caused by synthetic-fact LoRA.",
        "Only then replicate the stable protocol with Qwen2.5-1.5B.",
    ]
    for item in next_steps:
        story += [Paragraph(f"- {item}", styles["BodyText"]), Spacer(1, 5)]
    story += [Spacer(1, 10), Paragraph("Questions for the professor", styles["Heading1"])]
    for item in [
        "Which retain threshold should define an acceptable unlearning result?",
        "Should the next compute budget prioritize the oracle and multiple seeds before model scaling?",
        "Which evidence is required to support a claim of information removal rather than prompt-specific suppression?",
    ]:
        story += [Paragraph(f"- {item}", styles["BodyText"]), Spacer(1, 5)]
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return output


def _slide_header(canvas, title: str, kicker: str | None = None) -> None:
    width, height = landscape(letter)
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor(BLUE))
    canvas.rect(0, height - 13, width, 13, fill=1, stroke=0)
    if kicker:
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(colors.HexColor(BLUE))
        canvas.drawString(44, height - 42, kicker.upper())
    font_size = 25
    while canvas.stringWidth(title, "Helvetica-Bold", font_size) > width - 88 and font_size > 18:
        font_size -= 1
    canvas.setFont("Helvetica-Bold", font_size)
    canvas.setFillColor(colors.HexColor(INK))
    canvas.drawString(44, height - 78, title)


def _slide_footer(canvas, number: int) -> None:
    width, _ = landscape(letter)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(44, 22, AUTHOR)
    canvas.drawRightString(width - 44, 22, str(number))


def _draw_wrapped(canvas, text: str, x: float, y: float, width_chars: int, leading: float = 18,
                  font: str = "Helvetica", size: float = 13, color: str = INK) -> float:
    canvas.setFont(font, size)
    canvas.setFillColor(colors.HexColor(color))
    for line in textwrap.wrap(text, width=width_chars):
        canvas.drawString(x, y, line)
        y -= leading
    return y


def _write_presentation_pdf(df: pd.DataFrame, audit: dict[str, int], examples: pd.DataFrame,
                            paths: OutputPaths) -> Path:
    from reportlab.pdfgen import canvas as canvas_module

    output = paths.reports / "professor_progress_presentation.pdf"
    width, height = landscape(letter)
    c = canvas_module.Canvas(str(output), pagesize=(width, height))
    c.setTitle("First professor presentation - machine unlearning")
    c.setAuthor(AUTHOR)

    # 1 - Title
    c.setFillColor(colors.HexColor(INK)); c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColor(colors.HexColor(LIGHT_BLUE)); c.rect(0, height - 16, width, 16, fill=1, stroke=0)
    title_text = "Selective forgetting in a small language model"
    title_size = 34
    while c.stringWidth(title_text, "Helvetica-Bold", title_size) > width - 108 and title_size > 24:
        title_size -= 1
    c.setFont("Helvetica-Bold", title_size); c.setFillColor(colors.white)
    c.drawString(54, height - 165, title_text)
    c.setFont("Helvetica", 18); c.setFillColor(colors.HexColor("#D6DEEC"))
    c.drawString(56, height - 205, "Progress, trade-offs, negative results, and the next decision")
    c.setFont("Helvetica-Bold", 14); c.setFillColor(colors.HexColor(LIGHT_BLUE))
    c.drawString(56, 72, AUTHOR)
    c.showPage()

    # 2 - Question and pipeline
    _slide_header(c, "Can a model forget selected facts without losing useful behavior?", "Research question")
    steps = [
        ("1", "Start", "Qwen2.5-0.5B-Instruct"),
        ("2", "Learn", "LoRA on synthetic facts"),
        ("3", "Unlearn", "Eight objectives / controls"),
        ("4", "Evaluate", "Forget, retain, general"),
    ]
    box_w = 160; gap = 22; x = 44; y = 250
    for num, label, detail in steps:
        c.setFillColor(colors.HexColor(PANEL)); c.roundRect(x, y, box_w, 125, 8, fill=1, stroke=0)
        c.setFillColor(colors.HexColor(BLUE)); c.circle(x + 24, y + 96, 14, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 11); c.drawCentredString(x + 24, y + 92, num)
        c.setFillColor(colors.HexColor(INK)); c.setFont("Helvetica-Bold", 16); c.drawString(x + 16, y + 62, label)
        _draw_wrapped(c, detail, x + 16, y + 40, 22, 15, "Helvetica", 10.5, MUTED)
        if num != "4":
            c.setStrokeColor(colors.HexColor(GRID)); c.setLineWidth(2); c.line(x + box_w + 4, y + 62, x + box_w + gap - 4, y + 62)
        x += box_w + gap
    _draw_wrapped(c, "Success requires low forget accuracy and high retain/general accuracy at the same time.", 44, 175, 82, 22, "Helvetica-Bold", 16, INK)
    _slide_footer(c, 2); c.showPage()

    # 3 - Setup
    _slide_header(c, "The controlled benchmark separates memorization from preservation", "Experimental setup")
    metrics = [
        (str(audit["personas"]), "synthetic personas"),
        (f"{audit['forget_personas']} / {audit['retain_personas']}", "forget / retain personas"),
        (str(audit["fact_categories"]), "fact categories per persona"),
        (f"{audit['seen_questions']} / {audit['heldout_questions']}", "seen / held-out questions"),
        (str(audit["general_control_n"]), "general-control questions"),
    ]
    x = 46
    for value, label in metrics:
        c.setFont("Helvetica-Bold", 25); c.setFillColor(colors.HexColor(BLUE)); c.drawString(x, 330, value)
        _draw_wrapped(c, label, x, 300, 18, 16, "Helvetica", 11, MUTED)
        x += 144
    _draw_wrapped(c, "Held-out questions paraphrase the same facts. They test prompt generalization, not whether the information is irrecoverable under adversarial extraction.", 46, 205, 93, 22, "Helvetica", 15, INK)
    _slide_footer(c, 3); c.showPage()

    # 4 - Pareto
    _slide_header(c, "No method reaches both the forgetting target and retain floor", "Main result")
    c.drawImage(str(paths.figures / "01_pareto_heldout_tradeoff.png"), 52, 55, width=690, height=432, preserveAspectRatio=True, anchor="c")
    _slide_footer(c, 4); c.showPage()

    # 5 - General controls
    _slide_header(c, "The largest utility loss happens before unlearning", "Important confound")
    c.drawImage(str(paths.figures / "02_general_control_accuracy.png"), 52, 55, width=690, height=430, preserveAspectRatio=True, anchor="c")
    _slide_footer(c, 5); c.showPage()

    # 6 - Trajectories
    _slide_header(c, "Training longer strengthens forgetting and eventually breaks retention", "Optimization behavior")
    c.drawImage(str(paths.figures / "05_training_trajectories.png"), 55, 62, width=680, height=420, preserveAspectRatio=True, anchor="c")
    _slide_footer(c, 6); c.showPage()

    # 7 - Qualitative evidence
    _slide_header(c, "Accuracy changes correspond to visibly different model behavior", "Examples")
    chosen = examples.loc[(examples["split"] == "forget") & examples["method"].isin(["Gradient ascent", "Rollback constrained"])].head(2)
    y = 390
    for _, row in chosen.iterrows():
        c.setFillColor(colors.HexColor(PANEL)); c.roundRect(44, y - 122, 704, 112, 8, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 14); c.setFillColor(colors.HexColor(INK)); c.drawString(60, y - 34, row["method"])
        _draw_wrapped(c, f"Prompt: {row['prompt']}", 60, y - 55, 92, 14, "Helvetica", 10, MUTED)
        _draw_wrapped(c, f"Before: {row['before_answer']} | After: {row['after_answer']}", 60, y - 84, 105, 14, "Helvetica", 10, INK)
        y -= 140
    _draw_wrapped(c, "The full qualitative table is exported as CSV. These examples illustrate behavior; they do not replace extraction-oriented privacy tests.", 48, 78, 105, 16, "Helvetica", 10.5, MUTED)
    _slide_footer(c, 7); c.showPage()

    # 8 - Next decisions
    _slide_header(c, "The next milestone is stronger evidence, not another optimizer", "Recommendation")
    left = [
        "1. Train a retain-only oracle baseline.",
        "2. Repeat key methods across seeds.",
        "3. Add probability and extraction tests.",
        "4. Diagnose the LoRA general-control drop.",
        "5. Scale to 1.5B after the protocol stabilizes.",
    ]
    y = 390
    for item in left:
        y = _draw_wrapped(c, item, 54, y, 50, 27, "Helvetica-Bold", 15, INK) - 5
    c.setStrokeColor(colors.HexColor(GRID)); c.line(405, 120, 405, 405)
    c.setFont("Helvetica-Bold", 16); c.setFillColor(colors.HexColor(BLUE)); c.drawString(440, 390, "Decisions requested from the professor")
    y = 350
    for item in [
        "What retain threshold defines success?",
        "Oracle and seeds first, or scale the model now?",
        "What evidence is sufficient to claim removal?",
    ]:
        y = _draw_wrapped(c, "- " + item, 440, y, 43, 22, "Helvetica", 13, INK) - 12
    _slide_footer(c, 8); c.showPage()

    c.save()
    return output


def write_supporting_documents(paths: OutputPaths) -> None:
    (paths.reports / "speaker_notes.md").write_text(
        """# Speaker notes

## Slide 1
Frame this as a progress review and decision meeting, not a claim that unlearning has been solved.

## Slide 2
Define success as a joint condition: forgetting must improve without unacceptable retain or general degradation.

## Slide 3
Explain that the 1,000 held-out questions are paraphrases of the same facts. This is valuable, but not a privacy attack.

## Slide 4
Lead with the upper-left ideal. Emphasize that methods occupy a frontier and none enters the target region.

## Slide 5
State that the 88% to 56% decline occurs during LoRA. With 50 controls, 2 points equals one answer.

## Slide 6
Use the trajectories to motivate checkpoint selection and early stopping as part of the method, not an afterthought.

## Slide 7
Contrast a method that changes forgotten answers with rollback, which often keeps producing the original fact.

## Slide 8
Ask for an explicit priority decision: strengthen the evaluation protocol before spending compute on a larger model.
""",
        encoding="utf-8",
    )
    (paths.reports / "oracle_retraining_protocol.md").write_text(
        """# Retain-only oracle protocol

1. Start from the same Qwen2.5-0.5B-Instruct base checkpoint used for the LoRA baseline.
2. Use the same LoRA target modules, rank, optimizer, scheduler, maximum tokens, and scoring code.
3. Remove every forget-person training example; train only on the retain-person subset.
4. Select checkpoints using retain validation and general validation only. Never use forget evaluation for checkpoint selection.
5. Evaluate on the unchanged forget, retain, and general final sets.
6. Repeat with the same seeds used for the main unlearning methods.
7. Compare every unlearned model with this oracle using accuracy, token log-probability, output distributions, and paired per-question differences.

The original base model is not a sufficient oracle because it lacks both forget and retain synthetic facts.
""",
        encoding="utf-8",
    )
    (paths.reports / "experiment_checklist.md").write_text(
        """# Evidence checklist for the next professor update

- [ ] Retain-only oracle trained and evaluated
- [ ] At least three seeds for the principal baselines
- [ ] Mean, standard deviation, and paired question-level intervals
- [ ] Token log-probability of forgotten values
- [ ] Multiple paraphrase and hint-based extraction prompts
- [ ] Repeated sampling / top-k extraction test
- [ ] Partial-answer and normalized exact-match metrics
- [ ] LoRA-only general-control degradation ablation
- [ ] Hyperparameters, checkpoint policy, runtime, and GPU hours documented
- [ ] Qwen2.5-1.5B replication after the protocol is stable
""",
        encoding="utf-8",
    )


def _write_manifest(repo: Path, paths: OutputPaths, input_paths: Iterable[Path]) -> Path:
    outputs = [
        p for p in paths.root.rglob("*")
        if p.is_file() and p.name not in {"manifest.json", "unlearning_presentation_bundle.zip"}
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "presentations/first/presentation_pipeline.py",
        "inputs": [
            {"path": str(path.relative_to(repo)).replace("\\", "/"), "sha256": _sha256(path)}
            for path in input_paths if path.exists()
        ],
        "outputs": [
            {"path": str(path.relative_to(paths.root)).replace("\\", "/"), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(outputs)
        ],
    }
    output = paths.root / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def _make_bundle(paths: OutputPaths) -> Path:
    archive_base = paths.root / "unlearning_presentation_bundle"
    existing = archive_base.with_suffix(".zip")
    if existing.exists():
        existing.unlink()
    # Build outside the results tree so the archive cannot include itself.
    with tempfile.TemporaryDirectory(prefix="unlearning-presentation-") as temp_dir:
        temporary_base = Path(temp_dir) / "unlearning_presentation_bundle"
        temporary_archive = Path(
            shutil.make_archive(str(temporary_base), "zip", root_dir=paths.root, base_dir=".", logger=None)
        )
        shutil.move(str(temporary_archive), str(existing))
    return existing


def run(project_dir: str | Path | None = None) -> dict[str, str]:
    """Generate all assets and return their key paths."""
    project_dir = Path(project_dir or Path.cwd()).resolve()
    if project_dir.name == "results":
        project_dir = project_dir.parent
    repo = _repo_root(project_dir)
    project_dir = repo / "presentations" / "first"
    paths = _prepare_dirs(project_dir)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.facecolor": "white",
    })

    df = load_metrics(repo)
    audit = load_dataset_audit(repo)
    create_tables(df, audit, paths)
    plot_pareto(df, paths)
    plot_general_control(df, audit, paths)
    plot_seen_heldout(df, paths)
    plot_seen_heldout_highlighted(df, paths)
    plot_changes(df, paths)
    plot_training_trajectories(repo, paths)
    plot_candidate_sweeps(repo, paths)
    examples = create_qualitative_examples(repo, paths)

    markdown_report = _write_markdown_report(df, audit, paths)
    html_report = _write_html_report(markdown_report, paths)
    pdf_report = _write_pdf_report(df, audit, paths)
    presentation_pdf = _write_presentation_pdf(df, audit, examples, paths)
    write_supporting_documents(paths)

    inputs = [
        repo / "reports/week4-week7-master-comparison.csv",
        repo / "Week 4/results/gradient_ascent_unlearning_v1/results/unlearning_history.csv",
        repo / "Week 5/results/retain_regularized_unlearning_resumable_v1/results/sweep_history.csv",
        repo / "Week 6/results/constrained_gradient_unlearning_v1/results/sweep_history.csv",
        repo / "Week 7/results/adaptive_constrained_unlearning_v1/results/controller_history.csv",
    ]
    manifest = _write_manifest(repo, paths, inputs)
    bundle = _make_bundle(paths)

    return {
        "results": str(paths.root),
        "presentation_pdf": str(presentation_pdf),
        "report_pdf": str(pdf_report),
        "report_html": str(html_report),
        "report_markdown": str(markdown_report),
        "manifest": str(manifest),
        "bundle": str(bundle),
    }


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).resolve().parent), indent=2))
