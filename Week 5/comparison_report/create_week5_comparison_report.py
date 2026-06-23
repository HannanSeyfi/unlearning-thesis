"""Create Week 4 vs Week 5 comparison tables and plots.

This script is intentionally Colab-friendly: it only needs pandas, numpy, and
matplotlib, and it reads result files already committed to the thesis repo.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_ORDER = [
    "forget_all",
    "forget_heldout_paraphrases",
    "retain_all",
    "retain_heldout_paraphrases",
    "general",
]

METRIC_LABELS = {
    "forget_all": "Forget all",
    "forget_heldout_paraphrases": "Forget heldout",
    "retain_all": "Retain all",
    "retain_heldout_paraphrases": "Retain heldout",
    "general": "General",
}

LOWER_IS_BETTER = {"forget_all", "forget_heldout_paraphrases"}


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    return pd.read_csv(path)


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def pct(value: float) -> str:
    return f"{value:.1f}%"


def df_to_markdown(df: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    table = df.loc[:, list(columns)] if columns is not None else df
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        rendered = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                rendered.append(f"{value:.1f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def build_final_metrics_table(week4_metrics: dict, week5_metrics: dict) -> pd.DataFrame:
    rows = []
    week4_after = week4_metrics["after_unlearning"]
    week5_after = week5_metrics["after_unlearning"]
    for metric in METRIC_ORDER:
        week4_value = float(week4_after[metric])
        week5_value = float(week5_after[metric])
        delta = week5_value - week4_value
        lower_is_better = metric in LOWER_IS_BETTER
        if lower_is_better:
            winner = "Week 5" if week5_value < week4_value else "Week 4"
            direction = "lower is better"
        else:
            winner = "Week 5" if week5_value > week4_value else "Week 4"
            direction = "higher is better"
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "week4_after": week4_value,
                "week5_after": week5_value,
                "week5_minus_week4": delta,
                "direction": direction,
                "winner": winner,
            }
        )
    return pd.DataFrame(rows)


def summarize_categories(all_results_path: Path, model_stage: str, week_label: str) -> pd.DataFrame:
    frame = read_csv(all_results_path)
    frame = frame[frame["model_stage"] == model_stage].copy()
    if frame.empty:
        raise ValueError(f"No rows found for model_stage={model_stage} in {all_results_path}")

    frame["contains_value_bool"] = as_bool(frame["contains_value"])
    grouped = (
        frame.groupby(["eval_split", "category"], dropna=False)
        .agg(
            num_questions=("contains_value_bool", "size"),
            num_correct=("contains_value_bool", "sum"),
            contains_value_percentage=("contains_value_bool", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )
    grouped["week"] = week_label
    return grouped


def build_category_table(repo_root: Path) -> pd.DataFrame:
    week4_path = repo_root / "Week 4" / "results" / "gradient_ascent_unlearning_v1" / "results" / "all_before_after_results.csv"
    week5_path = repo_root / "Week 5" / "results" / "retain_regularized_unlearning_resumable_v1" / "results" / "all_before_after_results.csv"
    week4 = summarize_categories(week4_path, "after_gradient_ascent", "week4")
    week5 = summarize_categories(week5_path, "after_retain_regularized_unlearning", "week5")

    week4 = week4.rename(
        columns={
            "num_questions": "week4_num_questions",
            "num_correct": "week4_num_correct",
            "contains_value_percentage": "week4_percentage",
        }
    ).drop(columns=["week"])
    week5 = week5.rename(
        columns={
            "num_questions": "week5_num_questions",
            "num_correct": "week5_num_correct",
            "contains_value_percentage": "week5_percentage",
        }
    ).drop(columns=["week"])

    merged = week4.merge(week5, on=["eval_split", "category"], how="outer")
    merged["week5_minus_week4"] = merged["week5_percentage"] - merged["week4_percentage"]
    return merged.sort_values(["eval_split", "category"]).reset_index(drop=True)


def save_final_metric_plot(final_metrics: pd.DataFrame, output_path: Path) -> None:
    labels = final_metrics["label"].tolist()
    week4 = final_metrics["week4_after"].to_numpy()
    week5 = final_metrics["week5_after"].to_numpy()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars4 = ax.bar(x - width / 2, week4, width, label="Week 4", color="#3b6ea8")
    bars5 = ax.bar(x + width / 2, week5, width, label="Week 5", color="#2a9d8f")
    ax.set_ylabel("Contains-value accuracy (%)")
    ax.set_title("Week 4 vs Week 5 Final Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for bars in (bars4, bars5):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.3,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_tradeoff_plot(
    sweep_history: pd.DataFrame,
    week4_metrics: dict,
    week5_metrics: dict,
    output_path: Path,
) -> None:
    sweep = sweep_history.copy()
    sweep["retain_eligible_bool"] = as_bool(sweep["retain_eligible"])
    selected_id = week5_metrics["selected_candidate_id"]
    selected_epoch = int(week5_metrics["selected_epoch"])

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ineligible = sweep[~sweep["retain_eligible_bool"]]
    eligible = sweep[sweep["retain_eligible_bool"]]

    ax.scatter(
        ineligible["retain_heldout_selection_percentage"],
        ineligible["forget_heldout_selection_percentage"],
        s=44,
        color="#9a9a9a",
        alpha=0.6,
        label="Week 5 below retain threshold",
    )
    ax.scatter(
        eligible["retain_heldout_selection_percentage"],
        eligible["forget_heldout_selection_percentage"],
        s=52,
        color="#2a9d8f",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.9,
        label="Week 5 eligible",
    )

    selected = sweep[
        (sweep["candidate_id"] == selected_id)
        & (sweep["epoch"].astype(int) == selected_epoch)
    ]
    if not selected.empty:
        row = selected.iloc[0]
        ax.scatter(
            [row["retain_heldout_selection_percentage"]],
            [row["forget_heldout_selection_percentage"]],
            marker="*",
            s=260,
            color="#f4a261",
            edgecolors="#663300",
            linewidths=0.8,
            label=f"Selected Week 5: {selected_id} e{selected_epoch}",
            zorder=5,
        )
        ax.annotate(
            f"{selected_id} e{selected_epoch}",
            (
                row["retain_heldout_selection_percentage"],
                row["forget_heldout_selection_percentage"],
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax.scatter(
        [week4_metrics["after_unlearning"]["retain_heldout_paraphrases"]],
        [week4_metrics["after_unlearning"]["forget_heldout_paraphrases"]],
        marker="X",
        s=150,
        color="#d1495b",
        edgecolors="white",
        linewidths=0.8,
        label="Week 4 final",
        zorder=4,
    )

    ax.axvline(85.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(85.4, 96, "Week 5 retain threshold", fontsize=8, color="#444444")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Retain heldout accuracy (%) - higher is better")
    ax.set_ylabel("Forget heldout accuracy (%) - lower is better")
    ax.set_title("Week 5 Forget/Retain Trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_candidate_paths_plot(sweep_history: pd.DataFrame, output_path: Path) -> None:
    sweep = sweep_history.copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    cmap = plt.get_cmap("tab10")

    for index, (candidate_id, group) in enumerate(sweep.groupby("candidate_id")):
        group = group.sort_values("epoch")
        color = cmap(index % 10)
        ax.plot(
            group["retain_heldout_selection_percentage"],
            group["forget_heldout_selection_percentage"],
            marker="o",
            markersize=4,
            linewidth=1.3,
            color=color,
            alpha=0.85,
            label=candidate_id,
        )
        last = group.iloc[-1]
        ax.text(
            last["retain_heldout_selection_percentage"] + 0.6,
            last["forget_heldout_selection_percentage"],
            candidate_id.split("_", 1)[0],
            fontsize=7,
            color=color,
        )

    ax.axvline(85.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Retain heldout accuracy (%) - higher is better")
    ax.set_ylabel("Forget heldout accuracy (%) - lower is better")
    ax.set_title("Week 5 Candidate Trajectories Across Epochs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_category_plot(category_table: pd.DataFrame, output_path: Path) -> None:
    figure_splits = ["forget", "retain"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    width = 0.36

    for ax, eval_split in zip(axes, figure_splits):
        subset = category_table[category_table["eval_split"] == eval_split].copy()
        subset = subset.sort_values("category")
        labels = subset["category"].tolist()
        x = np.arange(len(labels))
        ax.bar(x - width / 2, subset["week4_percentage"], width, label="Week 4", color="#3b6ea8")
        ax.bar(x + width / 2, subset["week5_percentage"], width, label="Week 5", color="#2a9d8f")
        ax.set_title(f"{eval_split.title()} Categories")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 105)
        if ax is axes[0]:
            ax.set_ylabel("Contains-value accuracy (%)")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Category Breakdown: Week 4 vs Week 5")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    final_metrics: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    category_table: pd.DataFrame,
    week4_metrics: dict,
    week5_metrics: dict,
) -> None:
    selected_id = week5_metrics["selected_candidate_id"]
    selected_epoch = int(week5_metrics["selected_epoch"])
    selected_config = week5_metrics["selected_config"]
    before = week5_metrics["before_unlearning"]
    after = week5_metrics["after_unlearning"]

    final_display = final_metrics.copy()
    for column in ["week4_after", "week5_after", "week5_minus_week4"]:
        final_display[column] = final_display[column].map(lambda value: f"{value:+.1f}" if "minus" in column else f"{value:.1f}")

    category_display = category_table[
        category_table["eval_split"].isin(["forget", "retain"])
    ].copy()
    for column in ["week4_percentage", "week5_percentage", "week5_minus_week4"]:
        category_display[column] = category_display[column].map(lambda value: f"{value:+.1f}" if "minus" in column else f"{value:.1f}")

    top_candidates = candidate_summary.head(9).copy()
    top_candidates["learning_rate"] = top_candidates["learning_rate"].map(lambda value: f"{float(value):.0e}")
    for column in [
        "forget_heldout_selection_percentage",
        "retain_heldout_selection_percentage",
        "selection_score",
    ]:
        top_candidates[column] = top_candidates[column].map(lambda value: f"{float(value):.1f}")

    text = f"""# Week 5 Comparison Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Headline

Week 5 did not reproduce Week 4's strongest forgetting because it selected a
preservation-oriented checkpoint. The selected Week 5 adapter keeps retain and
general performance much higher, but it leaves more forget facts recoverable.

Selected Week 5 checkpoint:

- candidate: `{selected_id}`
- epoch: `{selected_epoch}`
- learning rate: `{selected_config['learning_rate']}`
- retain weight: `{selected_config['retain_weight']}`
- KL weight: `{selected_config['kl_weight']}`

Week 5 changed forget heldout accuracy from {pct(before['forget_heldout_paraphrases'])}
before unlearning to {pct(after['forget_heldout_paraphrases'])} after unlearning.
It kept retain heldout at {pct(after['retain_heldout_paraphrases'])} and general
control at {pct(after['general'])}.

## Final Metrics

{df_to_markdown(final_display, ['label', 'week4_after', 'week5_after', 'week5_minus_week4', 'direction', 'winner'])}

## Candidate Ranking

{df_to_markdown(top_candidates, ['candidate_id', 'epoch', 'learning_rate', 'retain_weight', 'kl_weight', 'forget_heldout_selection_percentage', 'retain_heldout_selection_percentage', 'selection_score'])}

## Category Breakdown

{df_to_markdown(category_display, ['eval_split', 'category', 'week4_percentage', 'week5_percentage', 'week5_minus_week4'])}

## Interpretation

Week 4 is the stronger forgetting baseline: it pushes forget heldout accuracy
down to {pct(week4_metrics['after_unlearning']['forget_heldout_paraphrases'])}.
The cost is collateral damage: retain heldout falls to
{pct(week4_metrics['after_unlearning']['retain_heldout_paraphrases'])}.

Week 5 is the preservation baseline: retain heldout remains at
{pct(week5_metrics['after_unlearning']['retain_heldout_paraphrases'])}, and
general control improves relative to Week 4. The trade-off is weaker forgetting,
especially in categories that remain easy for the model to answer.

The sweep history shows that more aggressive Week 5 candidates could forget
harder, but they crossed below the 85% retain-selection threshold. This makes
Week 5 useful as a trade-off map rather than as a single winning checkpoint.

## Generated Figures

- `week4_week5_final_metrics.png`
- `week5_forget_retain_tradeoff.png`
- `week5_candidate_tradeoff_paths.png`
- `week4_week5_category_breakdown.png`
"""
    output_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "Week 5" / "comparison_report" / "report_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    week4_results = repo_root / "Week 4" / "results" / "gradient_ascent_unlearning_v1" / "results"
    week5_results = repo_root / "Week 5" / "results" / "retain_regularized_unlearning_resumable_v1" / "results"

    week4_metrics = read_json(week4_results / "metrics.json")
    week5_metrics = read_json(week5_results / "metrics.json")
    sweep_history = read_csv(week5_results / "sweep_history.csv")
    candidate_summary = read_csv(week5_results / "candidate_best_summary.csv")
    candidate_summary = candidate_summary.sort_values("selection_score", ascending=False).reset_index(drop=True)
    candidate_summary.insert(0, "rank", np.arange(1, len(candidate_summary) + 1))

    final_metrics = build_final_metrics_table(week4_metrics, week5_metrics)
    category_table = build_category_table(repo_root)

    final_metrics.to_csv(output_dir / "week4_week5_final_metrics.csv", index=False)
    candidate_summary.to_csv(output_dir / "week5_candidate_best_summary_ranked.csv", index=False)
    category_table.to_csv(output_dir / "week4_week5_category_breakdown.csv", index=False)

    save_final_metric_plot(final_metrics, output_dir / "week4_week5_final_metrics.png")
    save_tradeoff_plot(
        sweep_history,
        week4_metrics,
        week5_metrics,
        output_dir / "week5_forget_retain_tradeoff.png",
    )
    save_candidate_paths_plot(
        sweep_history,
        output_dir / "week5_candidate_tradeoff_paths.png",
    )
    save_category_plot(
        category_table,
        output_dir / "week4_week5_category_breakdown.png",
    )
    write_report(
        output_dir / "week5_comparison_report.md",
        final_metrics,
        candidate_summary,
        category_table,
        week4_metrics,
        week5_metrics,
    )

    print(f"Wrote Week 5 comparison report outputs to: {output_dir}")
    print("Key files:")
    for path in sorted(output_dir.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
