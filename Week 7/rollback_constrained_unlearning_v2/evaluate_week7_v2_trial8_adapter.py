"""Restore and fully evaluate the preserved Week 7 v2 trial-8 adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


AUDIT_RUN_NAME = "rollback_constrained_unlearning_v2_trial8_audit"
SOURCE_RUN_NAME = "rollback_constrained_unlearning_v2"
SOURCE_CANDIDATE_ID = "r02_rollback_lab_guarded"
EXPECTED_TRIAL = 8
AUDIT_STAGE = "week7_v2_trial8_accepted"
MIN_MATERIAL_FORGET_IMPROVEMENT = 1.25
RETAIN_FLOOR = 84.0
GENERAL_FLOOR = 52.0


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing helper module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row_for_stage(frame: pd.DataFrame, stage: str) -> dict[str, Any]:
    matches = frame[frame["model_stage"] == stage]
    if matches.empty:
        raise KeyError(f"Comparison row not found: {stage}")
    return matches.iloc[-1].to_dict()


def numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or pd.isna(value):
        raise ValueError(f"Missing numeric metric {key!r}")
    return float(value)


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    audit_metrics: dict[str, Any],
) -> None:
    trial8 = audit_metrics["trial8_full_evaluation"]
    deltas = audit_metrics["deltas"]
    verdict = audit_metrics["verdict"]
    display_stages = {
        "before_unlearning_week35_adapter",
        "week6_constrained_gradient_selected",
        "week7_v1_adaptive_selected",
        "week7_v2_rollback_selected",
        AUDIT_STAGE,
    }
    display = comparison[comparison["model_stage"].isin(display_stages)].copy()
    lines = [
        "# Week 7 V2 Trial-8 Audit",
        "",
        f"Source candidate: `{SOURCE_CANDIDATE_ID}`",
        f"Preserved accepted trial: `{EXPECTED_TRIAL}`",
        f"Release asset: `{audit_metrics['source_release']['release_asset']}`",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Full Evaluation",
        "",
        "| model_stage | forget_heldout | retain_heldout | general |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"| {row['model_stage']} | "
            f"{float(row['forget_heldout_paraphrases']):.1f}% | "
            f"{float(row['retain_heldout_paraphrases']):.1f}% | "
            f"{float(row['general']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "Lower forget accuracy is better; higher retain and general accuracy are better.",
            "",
            "## Trial-8 Metrics",
            "",
            f"- forget held-out: `{trial8['forget_heldout_paraphrases']:.1f}%`",
            f"- retain held-out: `{trial8['retain_heldout_paraphrases']:.1f}%`",
            f"- general: `{trial8['general']:.1f}%`",
            f"- full lab-number retain: `{audit_metrics['trial8_lab_retain_percentage']:.1f}%`",
            "",
            "## Deltas",
            "",
            f"- forget vs baseline: `{deltas['vs_baseline']['forget_heldout_paraphrases']:+.1f}` points",
            f"- forget vs selected v2 trial 3: `{deltas['vs_v2_selected']['forget_heldout_paraphrases']:+.1f}` points",
            f"- retain vs selected v2 trial 3: `{deltas['vs_v2_selected']['retain_heldout_paraphrases']:+.1f}` points",
            f"- general vs selected v2 trial 3: `{deltas['vs_v2_selected']['general']:+.1f}` points",
            "",
            "## Source Selection Record",
            "",
            f"- selection forget: `{audit_metrics['source_trial_record']['forget_selection_percentage']:.1f}%`",
            f"- selection retain: `{audit_metrics['source_trial_record']['retain_selection_percentage']:.1f}%`",
            f"- selection general: `{audit_metrics['source_trial_record']['general_selection_percentage']:.1f}%`",
            f"- selection lab retain: `{audit_metrics['source_trial_record']['lab_retain_selection_percentage']:.1f}%`",
            "",
            "## Files",
            "",
            "- `metrics.json`",
            "- `trial8_cross_week_comparison.csv`",
            "- `trial8_full_evaluation/`",
            "- `percentage_summary.csv`",
            "- `category_summary.csv`",
            "- `identity_summary.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--audit-run-name", default=AUDIT_RUN_NAME)
    parser.add_argument("--push-results", action="store_true")
    parser.add_argument("--push-branch", default="main")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    v2 = load_module(
        repo_root
        / "Week 7"
        / "rollback_constrained_unlearning_v2"
        / "train_week7_rollback_constrained_unlearning_v2.py",
        "week7_v2_trial8_audit_helpers",
    )
    w6, w7 = v2.load_helpers(repo_root)

    audit_run_dir = repo_root / "Week 7" / "results" / args.audit_run_name
    output_dir = audit_run_dir / "results"
    cache_dir = audit_run_dir / "_restore_cache"
    adapter_cache = cache_dir / "trial8_adapter"
    if args.reset:
        w6.safe_rmtree(audit_run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (audit_run_dir / "failure.json").unlink(missing_ok=True)

    source_results_dir = repo_root / "Week 7" / "results" / SOURCE_RUN_NAME
    source_state_path = (
        source_results_dir
        / "resume_state"
        / "candidates"
        / SOURCE_CANDIDATE_ID
        / "state.json"
    )
    source_state = w6.read_json(source_state_path)
    source_epoch = int(source_state.get("epoch", -1))
    if source_epoch != EXPECTED_TRIAL:
        raise RuntimeError(
            f"Expected preserved accepted trial {EXPECTED_TRIAL}, found release epoch {source_epoch}."
        )
    if not source_state.get("release_asset"):
        raise RuntimeError("Trial-8 state does not contain a GitHub Release asset.")

    w6.safe_rmtree(adapter_cache)
    print("Restoring preserved trial-8 adapter:", source_state["release_asset"], flush=True)
    restored = w7.restore_adapter_release(
        repo_root=repo_root,
        metadata=source_state,
        destination_dir=adapter_cache,
    )
    if not restored:
        raise RuntimeError("Could not restore the preserved trial-8 Release adapter.")
    for required_name in ["adapter_config.json", "adapter_model.safetensors"]:
        if not (adapter_cache / required_name).exists():
            raise FileNotFoundError(f"Restored adapter is missing {required_name}")

    data_dir = repo_root / "Week 3.5" / "data" / "synthetic_facts_v1"
    general_dir = repo_root / "Week 3.5" / "data" / "general_controls_v1"
    train_forget = w6.read_jsonl(data_dir / "train_forget.jsonl")
    train_retain = w6.read_jsonl(data_dir / "train_retain.jsonl")
    eval_forget = w6.read_jsonl(data_dir / "eval_forget.jsonl")
    eval_retain = w6.read_jsonl(data_dir / "eval_retain.jsonl")
    general_controls = w6.read_jsonl(general_dir / "general_control.jsonl")
    is_seen_prompt = w6.make_seen_prompt_checker(train_forget, train_retain)
    _, _, selection_ids = w6.make_selection_splits(eval_forget, eval_retain, is_seen_prompt)
    _, general_selection_prompts = w7.split_general_controls(general_controls)

    tokenizer = w6.load_tokenizer()
    try:
        trial8_metrics, frames = w7.evaluate_finalist(
            w6=w6,
            candidate_id=AUDIT_STAGE,
            adapter_dir=adapter_cache,
            tokenizer=tokenizer,
            eval_forget=eval_forget,
            eval_retain=eval_retain,
            general_controls=general_controls,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            general_selection_prompts=general_selection_prompts,
            finalist_output_dir=output_dir / "trial8_full_evaluation",
        )
    finally:
        w6.safe_rmtree(cache_dir)

    baseline_frames = []
    source_output_dir = source_results_dir / "results"
    for filename in [
        "before_forget_results.csv",
        "before_retain_results.csv",
        "before_general_results.csv",
    ]:
        source = source_output_dir / filename
        destination = output_dir / filename
        shutil.copy2(source, destination)
        baseline_frames.append(pd.read_csv(destination))
    w6.summarize_outputs(
        output_dir,
        [*baseline_frames, frames["forget"], frames["retain"], frames["general"]],
    )

    source_comparison = pd.read_csv(source_output_dir / "week7_v2_cross_week_comparison.csv")
    comparison = source_comparison[source_comparison["model_stage"] != AUDIT_STAGE].copy()
    comparison = pd.concat(
        [comparison, pd.DataFrame([w6.metrics_row(AUDIT_STAGE, trial8_metrics)])],
        ignore_index=True,
    )
    comparison.to_csv(output_dir / "trial8_cross_week_comparison.csv", index=False)

    baseline = row_for_stage(comparison, "before_unlearning_week35_adapter")
    v2_selected = row_for_stage(comparison, "week7_v2_rollback_selected")
    trial_history = pd.read_csv(source_output_dir / "trial_history.csv")
    trial_record_rows = trial_history[
        (trial_history["candidate_id"] == SOURCE_CANDIDATE_ID)
        & (trial_history["trial"].astype(int) == EXPECTED_TRIAL)
    ]
    if trial_record_rows.empty:
        raise RuntimeError("Trial 8 is missing from the source trial history.")
    trial_record = trial_record_rows.iloc[-1].to_dict()
    trial_record = {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in trial_record.items()
    }

    lab_retain = w6.percentage(frames["retain"][frames["retain"]["category"] == "lab_number"])
    deltas = {
        "vs_baseline": {
            key: float(trial8_metrics[key]) - numeric(baseline, key)
            for key in ["forget_heldout_paraphrases", "retain_heldout_paraphrases", "general"]
        },
        "vs_v2_selected": {
            key: float(trial8_metrics[key]) - numeric(v2_selected, key)
            for key in ["forget_heldout_paraphrases", "retain_heldout_paraphrases", "general"]
        },
    }
    materially_better = (
        deltas["vs_v2_selected"]["forget_heldout_paraphrases"]
        <= -MIN_MATERIAL_FORGET_IMPROVEMENT
        and float(trial8_metrics["retain_heldout_paraphrases"]) >= RETAIN_FLOOR
        and float(trial8_metrics["general"]) >= GENERAL_FLOOR
    )
    if materially_better:
        verdict = (
            "Trial 8 materially improves forgetting over the selected v2 checkpoint while "
            "meeting the aggregate retain and general floors."
        )
    else:
        verdict = (
            "Trial 8 does not materially improve forgetting over the selected v2 checkpoint "
            "while meeting the aggregate utility floors."
        )

    audit_metrics = {
        "created_at_utc": w6.now_utc(),
        "audit_run_name": args.audit_run_name,
        "source_run_name": SOURCE_RUN_NAME,
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "source_trial": EXPECTED_TRIAL,
        "source_release": {
            key: source_state.get(key)
            for key in ["release_tag", "release_asset", "role", "epoch", "accepted_blocks"]
        },
        "source_trial_record": trial_record,
        "trial8_full_evaluation": trial8_metrics,
        "trial8_lab_retain_percentage": lab_retain,
        "material_improvement_threshold_points": MIN_MATERIAL_FORGET_IMPROVEMENT,
        "utility_floors": {
            "retain_heldout_percentage": RETAIN_FLOOR,
            "general_percentage": GENERAL_FLOOR,
        },
        "deltas": deltas,
        "materially_better_than_selected_v2": materially_better,
        "verdict": verdict,
    }
    w6.write_json(output_dir / "metrics.json", audit_metrics)
    w6.write_json(output_dir / "source_trial8_release_metadata.json", source_state)
    write_report(output_dir / "trial8_audit_report.md", comparison, audit_metrics)

    print("Trial-8 audit results:", trial8_metrics, flush=True)
    print(verdict, flush=True)
    print("Wrote audit outputs to:", audit_run_dir, flush=True)
    if args.push_results:
        sys.path.insert(0, str(repo_root))
        from Tools.github_colab_sync import commit_and_push

        commit_and_push(
            audit_run_dir,
            "Colab: save Week 7 v2 trial-8 audit",
            repo_dir=repo_root,
            branch=args.push_branch,
        )


def record_failure(args: argparse.Namespace, error: Exception) -> None:
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    failure_path = repo_root / "Week 7" / "results" / args.audit_run_name / "failure.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "source_results_unchanged": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Wrote trial-8 audit failure diagnostics:", failure_path, file=sys.stderr, flush=True)
    if args.push_results:
        try:
            sys.path.insert(0, str(repo_root))
            from Tools.github_colab_sync import commit_and_push

            commit_and_push(
                failure_path,
                "Colab: record Week 7 v2 trial-8 audit failure",
                repo_dir=repo_root,
                branch=args.push_branch,
            )
        except Exception as sync_error:
            print(f"Could not push audit failure diagnostics: {sync_error}", file=sys.stderr)


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        run_audit(parsed_args)
    except Exception as audit_error:
        record_failure(parsed_args, audit_error)
        raise
