"""Run Week 7 v3 normalized-gradient rollback unlearning."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SEED = 42
RUN_NAME = "normalized_rollback_unlearning_v3"
TRIAL_FORGET_EXAMPLES = 50
TARGET_FORGET_HELDOUT = 55.0
STRETCH_FORGET_HELDOUT = 45.0
MIN_FORGET_GAIN = 1.25
MEANINGFUL_FORGET_GAIN = 2.50
RETAIN_BASELINE_DROP = 8.50
GENERAL_BASELINE_DROP = 4.00
LAB_BASELINE_DROP = 3.125
MIN_RETAIN_FLOOR = 82.0
MIN_GENERAL_FLOOR = 50.0
MAX_TRIALS = 18
MAX_ACCEPTED_BLOCKS = 8
MAX_CONSECUTIVE_REJECTIONS = 5
MAX_NORMALIZATION_MULTIPLIER = 100.0
RETAIN_WEIGHT_STEP = 0.25
V3_RELEASE_TAG = "week7-v3-normalized-resume-state"
V3_RELEASE_NAME = "Week 7 V3 Normalized Resume State"


FOCUSED_CANDIDATES = [
    {
        "label": "normalized_projected_balanced",
        "learning_rate": 2e-5,
        "min_learning_rate": 5e-6,
        "max_learning_rate": 4e-5,
        "initial_forget_norm_ratio": 1.0,
        "min_forget_norm_ratio": 0.50,
        "max_forget_norm_ratio": 4.0,
        "no_progress_growth": 1.50,
        "accept_growth": 1.15,
        "utility_backoff": 0.70,
        "learning_rate_growth": 1.15,
        "learning_rate_backoff": 0.75,
        "retain_weight": 1.0,
        "max_retain_weight": 3.0,
        "kl_weight": 0.5,
        "lab_retain_oversample": 2,
        "projection_strength": 1.0,
    },
    {
        "label": "normalized_direct_stronger",
        "learning_rate": 2.5e-5,
        "min_learning_rate": 5e-6,
        "max_learning_rate": 5e-5,
        "initial_forget_norm_ratio": 1.25,
        "min_forget_norm_ratio": 0.50,
        "max_forget_norm_ratio": 5.0,
        "no_progress_growth": 1.50,
        "accept_growth": 1.15,
        "utility_backoff": 0.65,
        "learning_rate_growth": 1.15,
        "learning_rate_backoff": 0.70,
        "retain_weight": 1.0,
        "max_retain_weight": 3.5,
        "kl_weight": 0.5,
        "lab_retain_oversample": 2,
        "projection_strength": 0.0,
    },
]


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


def load_helpers(repo_root: Path):
    v2 = load_module(
        repo_root
        / "Week 7"
        / "rollback_constrained_unlearning_v2"
        / "train_week7_rollback_constrained_unlearning_v2.py",
        "week7_v2_shared_for_v3",
    )
    w6, w7 = v2.load_helpers(repo_root)
    w7.RESUME_RELEASE_TAG = V3_RELEASE_TAG
    w7.RESUME_RELEASE_NAME = V3_RELEASE_NAME
    return v2, w6, w7


def candidate_id(index: int, config: dict[str, Any]) -> str:
    return f"n{index:02d}_{config['label']}"


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def make_forget_trial_block(
    rows: list[dict[str, Any]],
    *,
    config_index: int,
    accepted_blocks: int,
) -> list[dict[str, Any]]:
    blocks_per_epoch = max(1, math.ceil(len(rows) / TRIAL_FORGET_EXAMPLES))
    virtual_epoch = accepted_blocks // blocks_per_epoch
    block_index = accepted_blocks % blocks_per_epoch
    shuffled = list(rows)
    random.Random(SEED + config_index * 1000 + virtual_epoch).shuffle(shuffled)
    start = block_index * TRIAL_FORGET_EXAMPLES
    block = shuffled[start : start + TRIAL_FORGET_EXAMPLES]
    if len(block) < TRIAL_FORGET_EXAMPLES:
        block.extend(shuffled[: TRIAL_FORGET_EXAMPLES - len(block)])
    return block


def derive_guardrails(baseline: dict[str, float]) -> dict[str, float]:
    return {
        "retain": max(MIN_RETAIN_FLOOR, baseline["retain"] - RETAIN_BASELINE_DROP),
        "general": max(MIN_GENERAL_FLOOR, baseline["general"] - GENERAL_BASELINE_DROP),
        "lab_retain": max(0.0, baseline["lab_retain"] - LAB_BASELINE_DROP),
    }


def utility_feasible(metrics: dict[str, float], floors: dict[str, float]) -> bool:
    return (
        metrics["retain"] >= floors["retain"]
        and metrics["general"] >= floors["general"]
        and metrics["lab_retain"] >= floors["lab_retain"]
    )


def proposal_decision(
    metrics: dict[str, float],
    floors: dict[str, float],
    accepted_forget_percentage: float,
) -> dict[str, float | bool]:
    forget_gain = accepted_forget_percentage - metrics["forget"]
    feasible = utility_feasible(metrics, floors)
    progress = forget_gain >= MIN_FORGET_GAIN - 1e-9
    return {
        "forget_gain": forget_gain,
        "utility_feasible": feasible,
        "forget_progress": progress,
        "accepted": bool(feasible and progress),
    }


def selection_score(metrics: dict[str, float], floors: dict[str, float]) -> float:
    return (
        100.0
        - metrics["forget"]
        + 0.30 * (metrics["retain"] - floors["retain"])
        + 0.15 * (metrics["general"] - floors["general"])
        + 0.10 * (metrics["lab_retain"] - floors["lab_retain"])
        + (5.0 if metrics["forget"] <= TARGET_FORGET_HELDOUT else 0.0)
    )


def scale_grads(
    grads: list[torch.Tensor | None],
    scale: float,
) -> list[torch.Tensor | None]:
    return [None if grad is None else grad * scale for grad in grads]


def gradient_norm(w6, grads: list[torch.Tensor | None]) -> float:
    norm_sq = w6.grad_norm_sq(grads)
    return math.sqrt(max(float(norm_sq.detach().cpu()), 0.0))


def normalized_gradient_update(
    *,
    w6,
    forget_grads: list[torch.Tensor | None],
    preserve_grads: list[torch.Tensor | None],
    forget_norm_ratio: float,
    projection_strength: float,
) -> tuple[list[torch.Tensor | None], dict[str, float | bool]]:
    projected_forget, projection = w6.project_forget_gradient(
        forget_grads,
        preserve_grads,
        projection_strength=projection_strength,
    )
    projected_norm = gradient_norm(w6, projected_forget)
    preserve_norm = gradient_norm(w6, preserve_grads)
    if projected_norm <= 1e-12 or preserve_norm <= 1e-12:
        multiplier = 1.0
    else:
        multiplier = min(
            MAX_NORMALIZATION_MULTIPLIER,
            forget_norm_ratio * preserve_norm / projected_norm,
        )
    balanced_forget = scale_grads(projected_forget, multiplier)
    combined = w6.combine_grads(balanced_forget, preserve_grads)
    return combined, {
        "raw_forget_norm": float(projection.forget_norm),
        "preserve_norm": preserve_norm,
        "projected_forget_norm": projected_norm,
        "balanced_forget_norm": gradient_norm(w6, balanced_forget),
        "normalization_multiplier": multiplier,
        "gradient_cosine": float(projection.cosine),
        "gradient_conflict": bool(projection.conflict),
        "projection_applied": bool(projection.projection_applied),
    }


def train_trial(
    *,
    w6,
    adapter_to_load: Path,
    tokenizer,
    teacher_model,
    forget_rows: list[dict[str, Any]],
    retain_rows: list[dict[str, Any]],
    learning_rate: float,
    forget_norm_ratio: float,
    retain_weight: float,
    kl_weight: float,
    projection_strength: float,
    seed: int,
):
    model = w6.load_adapter(adapter_to_load, trainable=True)
    parameters = w6.trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    device = w6.model_device(model)
    forget_loader = w6.make_loader(forget_rows, tokenizer, seed=seed, shuffle=True)
    retain_loader = w6.make_loader(retain_rows, tokenizer, seed=seed + 10000, shuffle=True)
    retain_iterator = iter(retain_loader)
    optimizer.zero_grad(set_to_none=True)
    history: dict[str, list[float]] = {
        "forget_loss": [],
        "retain_loss": [],
        "kl_loss": [],
        "raw_forget_norm": [],
        "preserve_norm": [],
        "balanced_forget_norm": [],
        "normalization_multiplier": [],
        "gradient_cosine": [],
        "gradient_conflict": [],
        "projection_applied": [],
    }

    for step, forget_batch in enumerate(forget_loader, 1):
        try:
            retain_batch = next(retain_iterator)
        except StopIteration:
            retain_iterator = iter(retain_loader)
            retain_batch = next(retain_iterator)
        forget_batch = w6.move_batch(forget_batch, device)
        retain_batch = w6.move_batch(retain_batch, device)

        forget_outputs = model(**forget_batch)
        forget_loss = forget_outputs.loss.clamp(max=w6.TARGET_FORGET_MAX)
        forget_grads = w6.detached_grads(-forget_loss, parameters)

        retain_outputs = model(**retain_batch)
        retain_loss = retain_outputs.loss
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=retain_batch["input_ids"],
                attention_mask=retain_batch["attention_mask"],
            )
        kl_loss = w6.masked_token_kl(
            retain_outputs.logits,
            teacher_outputs.logits,
            retain_batch["labels"],
        )
        preserve_objective = retain_weight * retain_loss + kl_weight * kl_loss
        preserve_grads = w6.detached_grads(preserve_objective, parameters)
        combined_grads, stats = normalized_gradient_update(
            w6=w6,
            forget_grads=forget_grads,
            preserve_grads=preserve_grads,
            forget_norm_ratio=forget_norm_ratio,
            projection_strength=projection_strength,
        )
        w6.accumulate_grads(
            parameters,
            combined_grads,
            scale=1.0 / w6.GRADIENT_ACCUMULATION_STEPS,
        )

        history["forget_loss"].append(float(forget_loss.detach().cpu()))
        history["retain_loss"].append(float(retain_loss.detach().cpu()))
        history["kl_loss"].append(float(kl_loss.detach().cpu()))
        for key in [
            "raw_forget_norm",
            "preserve_norm",
            "balanced_forget_norm",
            "normalization_multiplier",
            "gradient_cosine",
        ]:
            history[key].append(float(stats[key]))
        history["gradient_conflict"].append(1.0 if stats["gradient_conflict"] else 0.0)
        history["projection_applied"].append(1.0 if stats["projection_applied"] else 0.0)

        if step % w6.GRADIENT_ACCUMULATION_STEPS == 0 or step == len(forget_loader):
            torch.nn.utils.clip_grad_norm_(parameters, w6.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return model, {key: float(np.mean(values)) for key, values in history.items()}


def baseline_summary(
    cid: str,
    config: dict[str, Any],
    baseline: dict[str, float],
) -> dict[str, Any]:
    return {
        "candidate_id": cid,
        "best_trial": 0,
        "accepted_blocks": 0,
        "selection_forget_percentage": baseline["forget"],
        "selection_retain_percentage": baseline["retain"],
        "selection_general_percentage": baseline["general"],
        "selection_lab_retain_percentage": baseline["lab_retain"],
        "feasible": True,
        "meaningful_forgetting": False,
        "selection_score": -100.0,
        "best_adapter_kind": "source_week35",
        "learning_rate": float(config["learning_rate"]),
        "forget_norm_ratio": 0.0,
        "retain_weight": float(config["retain_weight"]),
        "kl_weight": float(config["kl_weight"]),
        "projection_strength": float(config["projection_strength"]),
    }


def run_candidate(
    *,
    v2,
    w6,
    w7,
    config_index: int,
    config: dict[str, Any],
    repo_root: Path,
    run_dir: Path,
    output_dir: Path,
    resume_dir: Path,
    selection_dir: Path,
    candidate_adapter_root: Path,
    source_adapter_dir: Path,
    tokenizer,
    teacher_model,
    train_forget: list[dict[str, Any]],
    weighted_retain: list[dict[str, Any]],
    forget_selection: list[dict[str, Any]],
    retain_selection: list[dict[str, Any]],
    general_selection: list[dict[str, Any]],
    is_seen_prompt,
    selection_ids: set[str],
    baseline: dict[str, float],
    floors: dict[str, float],
    max_trials: int,
    max_accepted_blocks: int,
    push_each_trial: bool,
    push_branch: str,
    trial_history: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cid = candidate_id(config_index, config)
    candidate_resume_dir = resume_dir / "candidates" / cid
    state_path = candidate_resume_dir / "state.json"
    best_state_path = candidate_resume_dir / "best.json"
    accepted_adapter_dir = candidate_resume_dir / "accepted_adapter"
    best_adapter_dir = candidate_adapter_root / cid
    candidate_resume_dir.mkdir(parents=True, exist_ok=True)

    prior_state = v2.state_from_disk(w6, state_path)
    if prior_state:
        state = prior_state
        restored = v2.restore_state_adapter(
            w7=w7,
            repo_root=repo_root,
            state=state,
            destination=accepted_adapter_dir,
        )
        if int(state.get("accepted_blocks", 0)) > 0 and not restored:
            raise RuntimeError(f"Could not restore the durable v3 accepted adapter for {cid}.")
    else:
        state = {
            "candidate_id": cid,
            "last_committed_trial": 0,
            "accepted_blocks": 0,
            "accepted_forget_percentage": baseline["forget"],
            "current_learning_rate": float(config["learning_rate"]),
            "current_forget_norm_ratio": float(config["initial_forget_norm_ratio"]),
            "current_retain_weight": float(config["retain_weight"]),
            "consecutive_rejections": 0,
            "accepted_adapter_kind": "source_week35",
            "guardrails": floors,
        }

    summary = next(
        (row.copy() for row in candidate_summaries if row.get("candidate_id") == cid),
        baseline_summary(cid, config, baseline),
    )
    start_trial = int(state.get("last_committed_trial", 0)) + 1
    accepted_blocks = int(state.get("accepted_blocks", 0))
    accepted_forget = float(state.get("accepted_forget_percentage", baseline["forget"]))
    current_lr = float(state.get("current_learning_rate", config["learning_rate"]))
    current_ratio = float(
        state.get("current_forget_norm_ratio", config["initial_forget_norm_ratio"])
    )
    current_retain_weight = float(state.get("current_retain_weight", config["retain_weight"]))
    consecutive_rejections = int(state.get("consecutive_rejections", 0))
    state_persisted = True

    if (
        start_trial > max_trials
        or accepted_blocks >= max_accepted_blocks
        or consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS
        or accepted_forget <= TARGET_FORGET_HELDOUT
    ):
        print("Skipping completed v3 candidate:", cid)
        return trial_history, candidate_summaries

    for trial in range(start_trial, max_trials + 1):
        if (
            accepted_blocks >= max_accepted_blocks
            or consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS
            or accepted_forget <= TARGET_FORGET_HELDOUT
        ):
            break
        adapter_to_load = accepted_adapter_dir if accepted_blocks > 0 else source_adapter_dir
        forget_block = make_forget_trial_block(
            train_forget,
            config_index=config_index,
            accepted_blocks=accepted_blocks,
        )
        model, train_metrics = train_trial(
            w6=w6,
            adapter_to_load=adapter_to_load,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            forget_rows=forget_block,
            retain_rows=weighted_retain,
            learning_rate=current_lr,
            forget_norm_ratio=current_ratio,
            retain_weight=current_retain_weight,
            kl_weight=float(config["kl_weight"]),
            projection_strength=float(config["projection_strength"]),
            seed=SEED + config_index * 10000 + trial,
        )
        model.eval()
        guard_metrics, guard_frames = v2.evaluate_guards(
            w6=w6,
            model=model,
            tokenizer=tokenizer,
            forget_selection=forget_selection,
            retain_selection=retain_selection,
            general_selection=general_selection,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            stage=f"{cid}_trial_{trial}",
        )
        v2.save_guard_frames(selection_dir, cid, trial, guard_frames)

        trial_lr = current_lr
        trial_ratio = current_ratio
        trial_retain_weight = current_retain_weight
        decision = proposal_decision(guard_metrics, floors, accepted_forget)
        forget_gain = float(decision["forget_gain"])
        feasible = bool(decision["utility_feasible"])
        progress = bool(decision["forget_progress"])
        accepted = bool(decision["accepted"])
        accepted_sync_ok = False
        best_sync_ok = False

        if accepted:
            action = "accept_forgetting_progress"
            accepted_blocks += 1
            accepted_forget = guard_metrics["forget"]
            consecutive_rejections = 0
            w6.safe_rmtree(accepted_adapter_dir)
            accepted_adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(accepted_adapter_dir)
            current_ratio = min(
                float(config["max_forget_norm_ratio"]),
                current_ratio * float(config["accept_growth"]),
            )
            state.update(
                {
                    "candidate_id": cid,
                    "last_committed_trial": trial,
                    "accepted_blocks": accepted_blocks,
                    "accepted_forget_percentage": accepted_forget,
                    "current_learning_rate": current_lr,
                    "current_forget_norm_ratio": current_ratio,
                    "current_retain_weight": current_retain_weight,
                    "consecutive_rejections": 0,
                    "accepted_adapter_kind": "release_asset",
                    "guardrails": floors,
                    "updated_at_utc": w6.now_utc(),
                }
            )
            if push_each_trial:
                accepted_sync_ok, synced_state = v2.sync_adapter_state(
                    w6=w6,
                    w7=w7,
                    repo_root=repo_root,
                    branch=push_branch,
                    run_name=run_dir.name,
                    candidate=cid,
                    role="accepted",
                    trial=trial,
                    adapter_dir=accepted_adapter_dir,
                    state=state,
                    state_path=state_path,
                )
                if accepted_sync_ok:
                    state = synced_state
                    state_persisted = True
                else:
                    state_persisted = False
            else:
                w6.write_json(state_path, state)
                accepted_sync_ok = True
                state_persisted = True

            score = selection_score(guard_metrics, floors)
            meaningful = baseline["forget"] - guard_metrics["forget"] >= MEANINGFUL_FORGET_GAIN
            if score > float(summary["selection_score"]):
                w6.safe_rmtree(best_adapter_dir)
                shutil.copytree(accepted_adapter_dir, best_adapter_dir)
                proposed_summary = {
                    "candidate_id": cid,
                    "best_trial": trial,
                    "accepted_blocks": accepted_blocks,
                    "selection_forget_percentage": guard_metrics["forget"],
                    "selection_retain_percentage": guard_metrics["retain"],
                    "selection_general_percentage": guard_metrics["general"],
                    "selection_lab_retain_percentage": guard_metrics["lab_retain"],
                    "feasible": True,
                    "meaningful_forgetting": meaningful,
                    "selection_score": score,
                    "best_adapter_kind": "release_asset",
                    "learning_rate": trial_lr,
                    "forget_norm_ratio": trial_ratio,
                    "retain_weight": trial_retain_weight,
                    "kl_weight": float(config["kl_weight"]),
                    "projection_strength": float(config["projection_strength"]),
                }
                best_state = {**proposed_summary, "updated_at_utc": w6.now_utc()}
                if push_each_trial:
                    best_sync_ok, _ = v2.sync_adapter_state(
                        w6=w6,
                        w7=w7,
                        repo_root=repo_root,
                        branch=push_branch,
                        run_name=run_dir.name,
                        candidate=cid,
                        role="best",
                        trial=trial,
                        adapter_dir=best_adapter_dir,
                        state=best_state,
                        state_path=best_state_path,
                    )
                else:
                    w6.write_json(best_state_path, best_state)
                    best_sync_ok = True
                if best_sync_ok:
                    summary = proposed_summary
        else:
            consecutive_rejections += 1
            if feasible:
                action = "reject_no_forget_progress_increase_intensity"
                current_ratio = min(
                    float(config["max_forget_norm_ratio"]),
                    current_ratio * float(config["no_progress_growth"]),
                )
                current_lr = min(
                    float(config["max_learning_rate"]),
                    current_lr * float(config["learning_rate_growth"]),
                )
            else:
                action = "reject_utility_rollback_backoff"
                current_ratio = max(
                    float(config["min_forget_norm_ratio"]),
                    current_ratio * float(config["utility_backoff"]),
                )
                current_lr = max(
                    float(config["min_learning_rate"]),
                    current_lr * float(config["learning_rate_backoff"]),
                )
                current_retain_weight = min(
                    float(config["max_retain_weight"]),
                    current_retain_weight + RETAIN_WEIGHT_STEP,
                )
            state.update(
                {
                    "candidate_id": cid,
                    "last_committed_trial": trial,
                    "accepted_blocks": accepted_blocks,
                    "accepted_forget_percentage": accepted_forget,
                    "current_learning_rate": current_lr,
                    "current_forget_norm_ratio": current_ratio,
                    "current_retain_weight": current_retain_weight,
                    "consecutive_rejections": consecutive_rejections,
                    "guardrails": floors,
                    "updated_at_utc": w6.now_utc(),
                }
            )
            if state_persisted:
                w6.write_json(state_path, state)
                accepted_sync_ok = True
            score = selection_score(guard_metrics, floors) if feasible else -1000.0
            meaningful = False

        row = {
            "candidate_id": cid,
            "trial": trial,
            "accepted_blocks_before": accepted_blocks - (1 if accepted else 0),
            "accepted_blocks_after": accepted_blocks,
            "accepted_forget_before": accepted_forget + (forget_gain if accepted else 0.0),
            "accepted_forget_after": accepted_forget,
            "accepted": accepted,
            "action": action,
            "trial_learning_rate": trial_lr,
            "trial_forget_norm_ratio": trial_ratio,
            "trial_retain_weight": trial_retain_weight,
            "kl_weight": float(config["kl_weight"]),
            "projection_strength": float(config["projection_strength"]),
            "lab_retain_oversample": int(config["lab_retain_oversample"]),
            "next_learning_rate": current_lr,
            "next_forget_norm_ratio": current_ratio,
            "next_retain_weight": current_retain_weight,
            "consecutive_rejections": consecutive_rejections,
            "forget_gain": forget_gain,
            "forget_progress": progress,
            "forget_selection_percentage": guard_metrics["forget"],
            "retain_selection_percentage": guard_metrics["retain"],
            "general_selection_percentage": guard_metrics["general"],
            "lab_retain_selection_percentage": guard_metrics["lab_retain"],
            "utility_feasible": feasible,
            "meaningful_forgetting": meaningful,
            "selection_score": score,
            **{f"mean_{key}": value for key, value in train_metrics.items()},
            "accepted_adapter_synced": accepted_sync_ok,
            "best_adapter_synced": best_sync_ok,
            "updated_at_utc": w6.now_utc(),
        }
        trial_history = [
            old
            for old in trial_history
            if not (old.get("candidate_id") == cid and int(old.get("trial", -1)) == trial)
        ]
        trial_history.append(row)
        candidate_summaries = [old for old in candidate_summaries if old.get("candidate_id") != cid]
        candidate_summaries.append(summary)
        v2.write_progress_tables(output_dir, trial_history, candidate_summaries)
        print(row, flush=True)

        progress_paths: list[Path] = [output_dir, selection_dir / cid, state_path]
        if best_state_path.exists():
            progress_paths.append(best_state_path)
        v2.sync_trial_progress(
            w6=w6,
            enabled=push_each_trial,
            paths=progress_paths,
            message=f"Colab: save Week 7 v3 normalized trial {cid} {trial:02d}",
            repo_root=repo_root,
            branch=push_branch,
        )
        del model
        v2.cleanup_cuda()

    return trial_history, candidate_summaries


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    candidate_df: pd.DataFrame,
    final_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    lines = [
        "# Week 7 V3 Normalized-Gradient Rollback",
        "",
        f"Selected candidate: `{metrics['selected_candidate_id']}`",
        f"Selected trial: `{metrics['selected_trial']}`",
        "",
        "V3 norm-balances forget-ascent and preservation gradients, rolls back every rejected proposal, and requires measurable forgetting progress before accepting a checkpoint.",
        "",
        "## Cross-Week Comparison",
        "",
        "| model_stage | forget_heldout | retain_heldout | general |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['model_stage']} | {float(row['forget_heldout_paraphrases']):.1f}% | "
            f"{float(row['retain_heldout_paraphrases']):.1f}% | {float(row['general']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Full Candidate Evaluation",
            "",
            "| candidate_id | trial | forget_heldout | retain_heldout | general |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in final_df.iterrows():
        lines.append(
            f"| {row['candidate_id']} | {int(row['best_trial'])} | "
            f"{float(row['forget_heldout_paraphrases']):.1f}% | "
            f"{float(row['retain_heldout_paraphrases']):.1f}% | {float(row['general']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Selection Ranking",
            "",
            "| candidate_id | trial | accepted_blocks | forget | retain | general | lab_retain | meaningful | score |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for _, row in candidate_df.sort_values("selection_score", ascending=False).iterrows():
        lines.append(
            f"| {row['candidate_id']} | {int(row['best_trial'])} | {int(row['accepted_blocks'])} | "
            f"{float(row['selection_forget_percentage']):.1f}% | "
            f"{float(row['selection_retain_percentage']):.1f}% | "
            f"{float(row['selection_general_percentage']):.1f}% | "
            f"{float(row['selection_lab_retain_percentage']):.1f}% | "
            f"{bool(row['meaningful_forgetting'])} | {float(row['selection_score']):.2f} |"
        )
    floors = metrics["guardrails"]
    lines.extend(
        [
            "",
            "## Runtime Guardrails",
            "",
            f"- retain selection floor: `{floors['retain_floor_percentage']:.3f}%`",
            f"- general selection floor: `{floors['general_floor_percentage']:.3f}%`",
            f"- lab-number retain floor: `{floors['lab_retain_floor_percentage']:.3f}%`",
            f"- minimum accepted forget gain: `{MIN_FORGET_GAIN:.2f}` points",
            f"- primary forget target: `{TARGET_FORGET_HELDOUT:.1f}%`",
            f"- stretch forget target: `{STRETCH_FORGET_HELDOUT:.1f}%`",
            "",
            "## Files",
            "",
            "- `trial_history.csv`",
            "- `candidate_best_summary.csv`",
            "- `candidate_final_evaluations.csv`",
            "- `week7_v3_cross_week_comparison.csv`",
            "- `gradient_diagnostics.csv`",
            "- `metrics.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS)
    parser.add_argument("--max-accepted-blocks", type=int, default=MAX_ACCEPTED_BLOCKS)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--push-each-trial", action="store_true")
    parser.add_argument("--push-branch", default="main")
    return parser.parse_args()


def record_failure(args: argparse.Namespace, error: Exception) -> None:
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    run_dir = repo_root / "Week 7" / "results" / args.run_name
    failure_path = run_dir / "failure.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "resume_note": "Rerun with RESET_EXISTING_RUN = False; v1, v2, audit, and v3 use separate folders.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Wrote Week 7 v3 failure diagnostics:", failure_path, file=sys.stderr, flush=True)
    if args.push_each_trial:
        try:
            sys.path.insert(0, str(repo_root))
            from Tools.github_colab_sync import commit_and_push

            commit_and_push(
                failure_path,
                "Colab: record Week 7 v3 failure diagnostics",
                repo_dir=repo_root,
                branch=args.push_branch,
            )
        except Exception as sync_error:
            print(f"Could not push v3 failure diagnostics: {sync_error}", file=sys.stderr)


def main(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    v2, w6, w7 = load_helpers(repo_root)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    run_dir = repo_root / "Week 7" / "results" / args.run_name
    selected_adapter_dir = run_dir / "best_week7_v3_adapter"
    candidate_adapter_root = run_dir / "candidate_adapters"
    output_dir = run_dir / "results"
    resume_dir = run_dir / "resume_state"
    selection_dir = resume_dir / "selection_results"
    if args.reset:
        w6.safe_rmtree(run_dir)
    for folder in [selected_adapter_dir, candidate_adapter_root, output_dir, resume_dir, selection_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    (run_dir / "failure.json").unlink(missing_ok=True)

    data_dir = repo_root / "Week 3.5" / "data" / "synthetic_facts_v1"
    general_dir = repo_root / "Week 3.5" / "data" / "general_controls_v1"
    source_adapter_dir = (
        repo_root / "Week 3.5" / "results" / "qwen05_high_accuracy_baseline" / "adapter"
    )
    if not source_adapter_dir.exists():
        fallback = repo_root / "Week 3.5" / "results" / "reference_successful_run" / "adapter"
        if fallback.exists():
            source_adapter_dir = fallback

    train_forget = w6.read_jsonl(data_dir / "train_forget.jsonl")
    train_retain = w6.read_jsonl(data_dir / "train_retain.jsonl")
    eval_forget = w6.read_jsonl(data_dir / "eval_forget.jsonl")
    eval_retain = w6.read_jsonl(data_dir / "eval_retain.jsonl")
    general_controls = w6.read_jsonl(general_dir / "general_control.jsonl")
    is_seen_prompt = w6.make_seen_prompt_checker(train_forget, train_retain)
    forget_selection, retain_selection, selection_ids = w6.make_selection_splits(
        eval_forget,
        eval_retain,
        is_seen_prompt,
    )
    general_selection, general_selection_prompts = w7.split_general_controls(general_controls)

    print("Week 7 v3 run folder:", run_dir)
    print("V1, v2, and trial-8 audit remain unchanged.")
    print("Candidates:", len(FOCUSED_CANDIDATES))
    print("Half-epoch forget examples:", TRIAL_FORGET_EXAMPLES)

    tokenizer = w6.load_tokenizer()
    teacher_model = w6.load_adapter(source_adapter_dir, trainable=False)
    teacher_model.eval()
    baseline, baseline_frames = v2.evaluate_guards(
        w6=w6,
        model=teacher_model,
        tokenizer=tokenizer,
        forget_selection=forget_selection,
        retain_selection=retain_selection,
        general_selection=general_selection,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
        stage="week35_baseline_v3_guard",
    )
    floors = derive_guardrails(baseline)
    v2.save_guard_frames(selection_dir, "baseline", 0, baseline_frames)
    w6.write_json(output_dir / "baseline_guard_metrics.json", baseline)
    w6.write_json(output_dir / "derived_guardrails.json", floors)
    print("Baseline guard metrics:", baseline)
    print("Derived v3 guardrails:", floors)

    history_df = load_table(output_dir / "trial_history.csv")
    summary_df = load_table(output_dir / "candidate_best_summary.csv")
    trial_history = history_df.to_dict("records") if not history_df.empty else []
    candidate_summaries = summary_df.to_dict("records") if not summary_df.empty else []

    for index, config in enumerate(FOCUSED_CANDIDATES, 1):
        trial_history, candidate_summaries = run_candidate(
            v2=v2,
            w6=w6,
            w7=w7,
            config_index=index,
            config=config,
            repo_root=repo_root,
            run_dir=run_dir,
            output_dir=output_dir,
            resume_dir=resume_dir,
            selection_dir=selection_dir,
            candidate_adapter_root=candidate_adapter_root,
            source_adapter_dir=source_adapter_dir,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            train_forget=train_forget,
            weighted_retain=v2.weighted_retain_rows(
                train_retain,
                int(config["lab_retain_oversample"]),
            ),
            forget_selection=forget_selection,
            retain_selection=retain_selection,
            general_selection=general_selection,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            baseline=baseline,
            floors=floors,
            max_trials=args.max_trials,
            max_accepted_blocks=args.max_accepted_blocks,
            push_each_trial=args.push_each_trial,
            push_branch=args.push_branch,
            trial_history=trial_history,
            candidate_summaries=candidate_summaries,
        )

    history_df, candidate_df = v2.write_progress_tables(
        output_dir,
        trial_history,
        candidate_summaries,
    )
    if candidate_df.empty:
        raise RuntimeError("No Week 7 v3 candidate summaries were produced.")
    diagnostic_columns = [
        column
        for column in history_df.columns
        if column.startswith("mean_")
        or column
        in {
            "candidate_id",
            "trial",
            "accepted",
            "action",
            "trial_forget_norm_ratio",
            "forget_gain",
            "utility_feasible",
        }
    ]
    history_df[diagnostic_columns].to_csv(output_dir / "gradient_diagnostics.csv", index=False)

    meaningful = candidate_df[w7.boolean_mask(candidate_df["meaningful_forgetting"])]
    accepted = candidate_df[candidate_df["accepted_blocks"].astype(int) > 0]
    ranking_pool = meaningful if not meaningful.empty else accepted
    if ranking_pool.empty:
        ranking_pool = candidate_df
    selected_row = ranking_pool.sort_values("selection_score", ascending=False).iloc[0].to_dict()

    del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_paths: dict[str, Path] = {}
    final_rows: list[dict[str, Any]] = []
    final_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for _, row in candidate_df.iterrows():
        summary = row.to_dict()
        cid = str(summary["candidate_id"])
        adapter_paths[cid] = v2.ensure_best_adapter(
            w6=w6,
            w7=w7,
            repo_root=repo_root,
            resume_dir=resume_dir,
            candidate_adapter_root=candidate_adapter_root,
            source_adapter_dir=source_adapter_dir,
            summary=summary,
        )
        full_metrics, frames = w7.evaluate_finalist(
            w6=w6,
            candidate_id=cid,
            adapter_dir=adapter_paths[cid],
            tokenizer=tokenizer,
            eval_forget=eval_forget,
            eval_retain=eval_retain,
            general_controls=general_controls,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            general_selection_prompts=general_selection_prompts,
            finalist_output_dir=output_dir / "candidate_finalists" / cid,
        )
        final_frames[cid] = frames
        final_rows.append(
            {
                "candidate_id": cid,
                "best_trial": int(summary["best_trial"]),
                **full_metrics,
            }
        )
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(output_dir / "candidate_final_evaluations.csv", index=False)

    selected_id = str(selected_row["candidate_id"])
    selected_metrics = next(row for row in final_rows if row["candidate_id"] == selected_id)
    selected_frames = final_frames[selected_id]
    w6.safe_rmtree(selected_adapter_dir)
    shutil.copytree(adapter_paths[selected_id], selected_adapter_dir)
    tokenizer.save_pretrained(selected_adapter_dir)
    selected_frames["forget"].to_csv(output_dir / "after_forget_results.csv", index=False)
    selected_frames["retain"].to_csv(output_dir / "after_retain_results.csv", index=False)
    selected_frames["general"].to_csv(output_dir / "after_general_results.csv", index=False)
    selected_frames["forget_excluding_selection"].to_csv(
        output_dir / "after_forget_final_excluding_selection_results.csv",
        index=False,
    )
    selected_frames["retain_excluding_selection"].to_csv(
        output_dir / "after_retain_final_excluding_selection_results.csv",
        index=False,
    )
    selected_frames["general_excluding_selection"].to_csv(
        output_dir / "after_general_final_excluding_selection_results.csv",
        index=False,
    )

    v2_output_dir = (
        repo_root / "Week 7" / "results" / "rollback_constrained_unlearning_v2" / "results"
    )
    before_frames = []
    for filename in [
        "before_forget_results.csv",
        "before_retain_results.csv",
        "before_general_results.csv",
    ]:
        source = v2_output_dir / filename
        destination = output_dir / filename
        shutil.copy2(source, destination)
        before_frames.append(pd.read_csv(destination))
    w6.summarize_outputs(
        output_dir,
        [*before_frames, selected_frames["forget"], selected_frames["retain"], selected_frames["general"]],
    )

    comparison_df = pd.read_csv(v2_output_dir / "week7_v2_cross_week_comparison.csv")
    comparison_df = comparison_df[
        comparison_df["model_stage"] != "week7_v3_normalized_selected"
    ].copy()
    comparison_df = pd.concat(
        [
            comparison_df,
            pd.DataFrame([w6.metrics_row("week7_v3_normalized_selected", selected_metrics)]),
        ],
        ignore_index=True,
    )
    comparison_df.to_csv(output_dir / "week7_v3_cross_week_comparison.csv", index=False)

    metrics = {
        "created_at_utc": w6.now_utc(),
        "run_name": args.run_name,
        "base_model_id": w6.MODEL_ID,
        "source_adapter_dir": str(source_adapter_dir),
        "unlearned_adapter_dir": str(selected_adapter_dir),
        "method": "Norm-balanced forget-ascent and preservation gradients with rollback and progress-gated acceptance",
        "selected_candidate_id": selected_id,
        "selected_trial": int(selected_row["best_trial"]),
        "selection_row": selected_row,
        "baseline_guard_metrics": baseline,
        "guardrails": {
            "retain_floor_percentage": floors["retain"],
            "general_floor_percentage": floors["general"],
            "lab_retain_floor_percentage": floors["lab_retain"],
            "minimum_forget_gain_points": MIN_FORGET_GAIN,
            "meaningful_forget_gain_points": MEANINGFUL_FORGET_GAIN,
            "target_forget_percentage": TARGET_FORGET_HELDOUT,
            "stretch_forget_percentage": STRETCH_FORGET_HELDOUT,
        },
        "training": {
            "trial_forget_examples": TRIAL_FORGET_EXAMPLES,
            "max_trials": args.max_trials,
            "max_accepted_blocks": args.max_accepted_blocks,
            "max_consecutive_rejections": MAX_CONSECUTIVE_REJECTIONS,
            "max_normalization_multiplier": MAX_NORMALIZATION_MULTIPLIER,
        },
        "after_unlearning": {
            key: value
            for key, value in selected_metrics.items()
            if key not in {"candidate_id", "best_trial"}
        },
        "candidate_final_evaluations": final_rows,
        "prior_week7_results_preserved": [
            str(repo_root / "Week 7" / "results" / "adaptive_constrained_unlearning_v1"),
            str(repo_root / "Week 7" / "results" / "rollback_constrained_unlearning_v2"),
            str(repo_root / "Week 7" / "results" / "rollback_constrained_unlearning_v2_trial8_audit"),
        ],
    }
    w6.write_json(output_dir / "metrics.json", metrics)
    write_report(
        output_dir / "week7_v3_normalized_report.md",
        comparison_df,
        candidate_df,
        final_df,
        metrics,
    )
    print("Wrote Week 7 v3 outputs to:", run_dir)
    print(comparison_df)
    print(final_df)

    for accepted_dir in resume_dir.glob("candidates/*/accepted_adapter"):
        w6.safe_rmtree(accepted_dir)
    w6.maybe_commit_and_push(
        args.push_each_trial,
        run_dir,
        "Colab: save Week 7 v3 normalized final outputs",
        repo_root=repo_root,
        branch=args.push_branch,
    )


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        main(parsed_args)
    except Exception as run_error:
        record_failure(parsed_args, run_error)
        raise
