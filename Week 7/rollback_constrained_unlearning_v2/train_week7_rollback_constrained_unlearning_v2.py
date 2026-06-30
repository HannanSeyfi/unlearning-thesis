"""Run Week 7 v2 rollback-constrained unlearning.

Unlike the reactive v1 controller, v2 never continues training from an update
that violates its retain, general, or lab-number guardrails. It restores the
last feasible adapter, reduces pressure and learning rate, strengthens retain
preservation, and retries the same quarter-epoch data block.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
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
RUN_NAME = "rollback_constrained_unlearning_v2"
TRIAL_FORGET_EXAMPLES = 25
RETAIN_SAFETY_FLOOR = 84.0
GENERAL_SAFETY_FLOOR = 52.0
LAB_RETAIN_SAFETY_FLOOR = 75.0
TARGET_FORGET_HELDOUT = 45.0
MAX_TRIALS = 20
MAX_ACCEPTED_BLOCKS = 16
MAX_CONSECUTIVE_REJECTIONS = 3
RETAIN_WEIGHT_STEP = 0.5
MAX_RETAIN_WEIGHT = 6.0
V2_RELEASE_TAG = "week7-v2-rollback-resume-state"
V2_RELEASE_NAME = "Week 7 V2 Rollback Resume State"


FOCUSED_CANDIDATES = [
    {
        "label": "rollback_boundary_balanced",
        "learning_rate": 2e-5,
        "min_learning_rate": 5e-6,
        "initial_forget_pressure": 1.5,
        "min_forget_pressure": 0.25,
        "max_forget_pressure": 3.0,
        "pressure_growth": 1.10,
        "pressure_backoff": 0.70,
        "learning_rate_backoff": 0.50,
        "retain_weight": 2.0,
        "kl_weight": 0.75,
        "lab_retain_oversample": 2,
    },
    {
        "label": "rollback_lab_guarded",
        "learning_rate": 3e-5,
        "min_learning_rate": 5e-6,
        "initial_forget_pressure": 1.25,
        "min_forget_pressure": 0.25,
        "max_forget_pressure": 3.0,
        "pressure_growth": 1.10,
        "pressure_backoff": 0.65,
        "learning_rate_backoff": 0.50,
        "retain_weight": 1.75,
        "kl_weight": 0.5,
        "lab_retain_oversample": 3,
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
    w7 = load_module(
        repo_root
        / "Week 7"
        / "adaptive_constrained_unlearning"
        / "train_week7_adaptive_constrained_unlearning.py",
        "week7_v1_shared_helpers",
    )
    w7.RESUME_RELEASE_TAG = V2_RELEASE_TAG
    w7.RESUME_RELEASE_NAME = V2_RELEASE_NAME
    w6 = w7.load_week6_helpers(repo_root)
    return w6, w7


def candidate_id(index: int, config: dict[str, Any]) -> str:
    return f"r{index:02d}_{config['label']}"


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_progress_tables(
    output_dir: Path,
    trial_history: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(trial_history)
    if not history_df.empty:
        history_df = history_df.sort_values(["candidate_id", "trial"])
    history_df.to_csv(output_dir / "trial_history.csv", index=False)

    candidate_df = pd.DataFrame(candidate_summaries)
    if not candidate_df.empty:
        candidate_df = candidate_df.drop_duplicates("candidate_id", keep="last")
        candidate_df = candidate_df.sort_values("selection_score", ascending=False)
    candidate_df.to_csv(output_dir / "candidate_best_summary.csv", index=False)
    return history_df, candidate_df


def make_forget_trial_block(
    rows: list[dict[str, Any]],
    *,
    config_index: int,
    accepted_blocks: int,
) -> list[dict[str, Any]]:
    blocks_per_epoch = max(1, len(rows) // TRIAL_FORGET_EXAMPLES)
    virtual_epoch = accepted_blocks // blocks_per_epoch
    block_index = accepted_blocks % blocks_per_epoch
    shuffled = list(rows)
    random.Random(SEED + config_index * 1000 + virtual_epoch).shuffle(shuffled)
    start = block_index * TRIAL_FORGET_EXAMPLES
    block = shuffled[start : start + TRIAL_FORGET_EXAMPLES]
    if len(block) < TRIAL_FORGET_EXAMPLES:
        block.extend(shuffled[: TRIAL_FORGET_EXAMPLES - len(block)])
    return block


def weighted_retain_rows(rows: list[dict[str, Any]], lab_oversample: int) -> list[dict[str, Any]]:
    weighted = list(rows)
    lab_rows = [row for row in rows if row.get("fact_type") == "lab_number"]
    for _ in range(max(0, int(lab_oversample) - 1)):
        weighted.extend(lab_rows)
    return weighted


def lab_retain_percentage(w6, frame: pd.DataFrame) -> float:
    return w6.percentage(frame[frame["category"] == "lab_number"])


def guardrail_score(
    forget_pct: float,
    retain_pct: float,
    general_pct: float,
    lab_retain_pct: float,
) -> tuple[bool, float]:
    feasible = (
        retain_pct >= RETAIN_SAFETY_FLOOR
        and general_pct >= GENERAL_SAFETY_FLOOR
        and lab_retain_pct >= LAB_RETAIN_SAFETY_FLOOR
    )
    if not feasible:
        return False, -1000.0 + retain_pct + 0.25 * general_pct + 0.25 * lab_retain_pct - forget_pct
    target_bonus = 5.0 if forget_pct <= TARGET_FORGET_HELDOUT else 0.0
    score = (
        100.0
        - forget_pct
        + 0.30 * (retain_pct - RETAIN_SAFETY_FLOOR)
        + 0.15 * (general_pct - GENERAL_SAFETY_FLOOR)
        + 0.10 * (lab_retain_pct - LAB_RETAIN_SAFETY_FLOOR)
        + target_bonus
    )
    return True, score


def evaluate_guards(
    *,
    w6,
    model,
    tokenizer,
    forget_selection: list[dict[str, Any]],
    retain_selection: list[dict[str, Any]],
    general_selection: list[dict[str, Any]],
    is_seen_prompt,
    selection_ids: set[str],
    stage: str,
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    forget_df = w6.evaluate(
        model,
        tokenizer,
        forget_selection,
        "forget_guard_selection",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
        progress_every=0,
    )
    retain_df = w6.evaluate(
        model,
        tokenizer,
        retain_selection,
        "retain_guard_selection",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
        progress_every=0,
    )
    general_df = w6.evaluate(
        model,
        tokenizer,
        general_selection,
        "general_guard_selection",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=set(),
        general=True,
        progress_every=0,
    )
    metrics = {
        "forget": w6.percentage(forget_df),
        "retain": w6.percentage(retain_df),
        "general": w6.percentage(general_df),
        "lab_retain": lab_retain_percentage(w6, retain_df),
    }
    return metrics, {"forget": forget_df, "retain": retain_df, "general": general_df}


def save_guard_frames(
    selection_dir: Path,
    candidate: str,
    trial: int,
    frames: dict[str, pd.DataFrame],
) -> Path:
    candidate_dir = selection_dir / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame.to_csv(candidate_dir / f"trial_{trial:02d}_{split}_selection.csv", index=False)
    return candidate_dir


def train_trial(
    *,
    w6,
    adapter_to_load: Path,
    tokenizer,
    teacher_model,
    forget_rows: list[dict[str, Any]],
    retain_rows: list[dict[str, Any]],
    learning_rate: float,
    forget_pressure: float,
    retain_weight: float,
    kl_weight: float,
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
    losses = {"forget": [], "retain": [], "kl": [], "objective": []}

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
        objective = -forget_pressure * forget_loss + retain_weight * retain_loss + kl_weight * kl_loss
        (objective / w6.GRADIENT_ACCUMULATION_STEPS).backward()

        losses["forget"].append(float(forget_loss.detach().cpu()))
        losses["retain"].append(float(retain_loss.detach().cpu()))
        losses["kl"].append(float(kl_loss.detach().cpu()))
        losses["objective"].append(float(objective.detach().cpu()))
        if step % w6.GRADIENT_ACCUMULATION_STEPS == 0 or step == len(forget_loader):
            torch.nn.utils.clip_grad_norm_(parameters, w6.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return model, {name: float(np.mean(values)) for name, values in losses.items()}


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def state_from_disk(w6, path: Path) -> dict[str, Any] | None:
    return w6.read_json(path) if path.exists() else None


def restore_state_adapter(
    *,
    w7,
    repo_root: Path,
    state: dict[str, Any],
    destination: Path,
) -> bool:
    if destination.exists():
        return True
    if state.get("accepted_blocks", 0) == 0:
        return False
    return w7.restore_adapter_release(
        repo_root=repo_root,
        metadata=state,
        destination_dir=destination,
    )


def sync_adapter_state(
    *,
    w6,
    w7,
    repo_root: Path,
    branch: str,
    run_name: str,
    candidate: str,
    role: str,
    trial: int,
    adapter_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> tuple[bool, dict[str, Any]]:
    pending_path = state_path.with_name(f".{state_path.stem}_{role}_pending.json")
    w6.write_json(pending_path, state)
    try:
        w7.sync_adapter_release(
            w6,
            repo_root=repo_root,
            run_name=run_name,
            branch=branch,
            candidate_id=candidate,
            role=role,
            epoch=trial,
            adapter_dir=adapter_dir,
            metadata_path=pending_path,
        )
        synced_state = w6.read_json(pending_path)
        w6.write_json(state_path, synced_state)
        return True, synced_state
    except Exception as error:
        print(
            f"WARNING: could not sync {role} adapter for {candidate} trial {trial}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return False, state
    finally:
        pending_path.unlink(missing_ok=True)


def baseline_summary_row(
    candidate: str,
    config: dict[str, Any],
    baseline_metrics: dict[str, float],
) -> dict[str, Any]:
    feasible, score = guardrail_score(
        baseline_metrics["forget"],
        baseline_metrics["retain"],
        baseline_metrics["general"],
        baseline_metrics["lab_retain"],
    )
    return {
        "candidate_id": candidate,
        "best_trial": 0,
        "accepted_blocks": 0,
        "selection_forget_percentage": baseline_metrics["forget"],
        "selection_retain_percentage": baseline_metrics["retain"],
        "selection_general_percentage": baseline_metrics["general"],
        "selection_lab_retain_percentage": baseline_metrics["lab_retain"],
        "feasible": feasible,
        "selection_score": score,
        "best_adapter_kind": "source_week35",
        "learning_rate": float(config["learning_rate"]),
        "forget_pressure": 0.0,
        "retain_weight": float(config["retain_weight"]),
        "kl_weight": float(config["kl_weight"]),
    }


def run_candidate(
    *,
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
    baseline_metrics: dict[str, float],
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

    prior_state = state_from_disk(w6, state_path)
    if prior_state:
        state = prior_state
        restored = restore_state_adapter(
            w7=w7,
            repo_root=repo_root,
            state=state,
            destination=accepted_adapter_dir,
        )
        if int(state.get("accepted_blocks", 0)) > 0 and not restored:
            raise RuntimeError(
                f"Could not restore the durable accepted adapter for {cid}. "
                "Check the Week 7 v2 GitHub Release assets, then rerun."
            )
    else:
        state = {
            "candidate_id": cid,
            "last_committed_trial": 0,
            "accepted_blocks": 0,
            "current_learning_rate": float(config["learning_rate"]),
            "current_forget_pressure": float(config["initial_forget_pressure"]),
            "current_retain_weight": float(config["retain_weight"]),
            "consecutive_rejections": 0,
            "accepted_adapter_kind": "source_week35",
        }

    summary = next(
        (row.copy() for row in candidate_summaries if row.get("candidate_id") == cid),
        baseline_summary_row(cid, config, baseline_metrics),
    )
    start_trial = int(state.get("last_committed_trial", 0)) + 1
    accepted_blocks = int(state.get("accepted_blocks", 0))
    current_lr = float(state.get("current_learning_rate", config["learning_rate"]))
    current_pressure = float(state.get("current_forget_pressure", config["initial_forget_pressure"]))
    current_retain_weight = float(state.get("current_retain_weight", config["retain_weight"]))
    consecutive_rejections = int(state.get("consecutive_rejections", 0))
    state_persisted = True

    if start_trial > max_trials or accepted_blocks >= max_accepted_blocks or consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
        print("Skipping completed v2 candidate:", cid)
        return trial_history, candidate_summaries

    for trial in range(start_trial, max_trials + 1):
        if accepted_blocks >= max_accepted_blocks or consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
            break
        adapter_to_load = accepted_adapter_dir if accepted_blocks > 0 else source_adapter_dir
        forget_block = make_forget_trial_block(
            train_forget,
            config_index=config_index,
            accepted_blocks=accepted_blocks,
        )
        model, loss_metrics = train_trial(
            w6=w6,
            adapter_to_load=adapter_to_load,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            forget_rows=forget_block,
            retain_rows=weighted_retain,
            learning_rate=current_lr,
            forget_pressure=current_pressure,
            retain_weight=current_retain_weight,
            kl_weight=float(config["kl_weight"]),
            seed=SEED + config_index * 10000 + trial,
        )
        model.eval()
        guard_metrics, guard_frames = evaluate_guards(
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
        save_guard_frames(selection_dir, cid, trial, guard_frames)
        feasible, score = guardrail_score(
            guard_metrics["forget"],
            guard_metrics["retain"],
            guard_metrics["general"],
            guard_metrics["lab_retain"],
        )

        trial_lr = current_lr
        trial_pressure = current_pressure
        trial_retain_weight = current_retain_weight
        accepted = bool(feasible)
        accepted_sync_ok = False
        best_sync_ok = False
        action = "accept"
        if accepted:
            accepted_blocks += 1
            consecutive_rejections = 0
            w6.safe_rmtree(accepted_adapter_dir)
            accepted_adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(accepted_adapter_dir)
            current_pressure = min(
                float(config["max_forget_pressure"]),
                current_pressure * float(config["pressure_growth"]),
            )
            state.update(
                {
                    "candidate_id": cid,
                    "last_committed_trial": trial,
                    "accepted_blocks": accepted_blocks,
                    "current_learning_rate": current_lr,
                    "current_forget_pressure": current_pressure,
                    "current_retain_weight": current_retain_weight,
                    "consecutive_rejections": consecutive_rejections,
                    "accepted_adapter_kind": "release_asset",
                    "updated_at_utc": w6.now_utc(),
                }
            )
            if push_each_trial:
                accepted_sync_ok, synced_state = sync_adapter_state(
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
                    "selection_score": score,
                    "best_adapter_kind": "release_asset",
                    "learning_rate": trial_lr,
                    "forget_pressure": trial_pressure,
                    "retain_weight": trial_retain_weight,
                    "kl_weight": float(config["kl_weight"]),
                }
                best_state = {**proposed_summary, "updated_at_utc": w6.now_utc()}
                if push_each_trial:
                    best_sync_ok, _ = sync_adapter_state(
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
            action = "reject_and_rollback"
            consecutive_rejections += 1
            current_pressure = max(
                float(config["min_forget_pressure"]),
                current_pressure * float(config["pressure_backoff"]),
            )
            current_lr = max(
                float(config["min_learning_rate"]),
                current_lr * float(config["learning_rate_backoff"]),
            )
            current_retain_weight = min(MAX_RETAIN_WEIGHT, current_retain_weight + RETAIN_WEIGHT_STEP)
            state.update(
                {
                    "candidate_id": cid,
                    "last_committed_trial": trial,
                    "accepted_blocks": accepted_blocks,
                    "current_learning_rate": current_lr,
                    "current_forget_pressure": current_pressure,
                    "current_retain_weight": current_retain_weight,
                    "consecutive_rejections": consecutive_rejections,
                    "updated_at_utc": w6.now_utc(),
                }
            )
            if state_persisted:
                w6.write_json(state_path, state)
                accepted_sync_ok = True

        row = {
            "candidate_id": cid,
            "trial": trial,
            "accepted_blocks_before": accepted_blocks - (1 if accepted else 0),
            "accepted_blocks_after": accepted_blocks,
            "accepted": accepted,
            "action": action,
            "trial_learning_rate": trial_lr,
            "trial_forget_pressure": trial_pressure,
            "trial_retain_weight": trial_retain_weight,
            "kl_weight": float(config["kl_weight"]),
            "lab_retain_oversample": int(config["lab_retain_oversample"]),
            "next_learning_rate": current_lr,
            "next_forget_pressure": current_pressure,
            "next_retain_weight": current_retain_weight,
            "consecutive_rejections": consecutive_rejections,
            "forget_selection_percentage": guard_metrics["forget"],
            "retain_selection_percentage": guard_metrics["retain"],
            "general_selection_percentage": guard_metrics["general"],
            "lab_retain_selection_percentage": guard_metrics["lab_retain"],
            "feasible": feasible,
            "selection_score": score,
            "mean_forget_loss": loss_metrics["forget"],
            "mean_retain_loss": loss_metrics["retain"],
            "mean_kl_loss": loss_metrics["kl"],
            "mean_training_objective": loss_metrics["objective"],
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
        write_progress_tables(output_dir, trial_history, candidate_summaries)
        print(row, flush=True)

        progress_paths: list[Path] = [
            output_dir,
            selection_dir / cid,
            state_path,
        ]
        if best_state_path.exists():
            progress_paths.append(best_state_path)
        w6.maybe_commit_and_push(
            push_each_trial,
            progress_paths,
            f"Colab: save Week 7 v2 rollback trial {cid} {trial:02d}",
            repo_root=repo_root,
            branch=push_branch,
        )
        del model
        cleanup_cuda()

        if accepted and guard_metrics["forget"] <= TARGET_FORGET_HELDOUT:
            print("Candidate reached the feasible forgetting target:", cid)
            break

    return trial_history, candidate_summaries


def ensure_best_adapter(
    *,
    w6,
    w7,
    repo_root: Path,
    resume_dir: Path,
    candidate_adapter_root: Path,
    source_adapter_dir: Path,
    summary: dict[str, Any],
) -> Path:
    if int(summary["best_trial"]) == 0:
        return source_adapter_dir
    cid = str(summary["candidate_id"])
    adapter_dir = candidate_adapter_root / cid
    if adapter_dir.exists():
        return adapter_dir
    metadata_path = resume_dir / "candidates" / cid / "best.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing v2 best-adapter metadata: {metadata_path}")
    metadata = w6.read_json(metadata_path)
    if int(metadata.get("best_trial", metadata.get("epoch", -1))) != int(summary["best_trial"]):
        raise RuntimeError(f"V2 best-adapter epoch mismatch for {cid}")
    if not w7.restore_adapter_release(
        repo_root=repo_root,
        metadata=metadata,
        destination_dir=adapter_dir,
    ):
        raise RuntimeError(f"Could not restore v2 candidate-best adapter: {cid}")
    return adapter_dir


def read_comparison_metrics(w6, w7, repo_root: Path) -> dict[str, dict[str, Any]]:
    baselines = w7.read_comparison_metrics(w6, repo_root)
    v1_path = (
        repo_root
        / "Week 7"
        / "results"
        / "adaptive_constrained_unlearning_v1"
        / "results"
        / "metrics.json"
    )
    if v1_path.exists():
        baselines["week7_v1_adaptive_selected"] = w6.read_json(v1_path)["after_unlearning"]
    return baselines


def write_report(
    path: Path,
    comparison_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    final_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    display = comparison_df.copy()
    metric_columns = [
        "forget_all",
        "forget_heldout_paraphrases",
        "retain_all",
        "retain_heldout_paraphrases",
        "general",
    ]
    for column in metric_columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.1f}%"
        )
    lines = [
        "# Week 7 V2 Rollback-Constrained Unlearning",
        "",
        f"Selected candidate: `{metrics['selected_candidate_id']}`",
        f"Selected trial: `{metrics['selected_trial']}`",
        "",
        "V2 accepts only trial blocks that satisfy aggregate retain, general, and lab-number retain guardrails. Rejected trials are rolled back before the next attempt.",
        "",
        "## Cross-Week Comparison",
        "",
        "| model_stage | forget_all | forget_heldout | retain_all | retain_heldout | general |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in display.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_stage"]),
                    str(row["forget_all"]),
                    str(row["forget_heldout_paraphrases"]),
                    str(row["retain_all"]),
                    str(row["retain_heldout_paraphrases"]),
                    str(row["general"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Full Candidate Evaluation",
            "",
            "| candidate_id | trial | forget_heldout | retain_heldout | general |",
            "| --- | --- | --- | --- | --- |",
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
            "| candidate_id | trial | accepted_blocks | forget | retain | general | lab_retain | score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in candidate_df.sort_values("selection_score", ascending=False).iterrows():
        lines.append(
            f"| {row['candidate_id']} | {int(row['best_trial'])} | {int(row['accepted_blocks'])} | "
            f"{float(row['selection_forget_percentage']):.1f}% | "
            f"{float(row['selection_retain_percentage']):.1f}% | "
            f"{float(row['selection_general_percentage']):.1f}% | "
            f"{float(row['selection_lab_retain_percentage']):.1f}% | "
            f"{float(row['selection_score']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            f"- retain selection: at least `{RETAIN_SAFETY_FLOOR:.1f}%`",
            f"- general selection: at least `{GENERAL_SAFETY_FLOOR:.1f}%`",
            f"- lab-number retain selection: at least `{LAB_RETAIN_SAFETY_FLOOR:.1f}%`",
            f"- forget target: at most `{TARGET_FORGET_HELDOUT:.1f}%`",
            "",
            "## Files",
            "",
            "- `trial_history.csv`",
            "- `candidate_best_summary.csv`",
            "- `candidate_final_evaluations.csv`",
            "- `week7_v2_cross_week_comparison.csv`",
            "- `metrics.json`",
            "- `candidate_finalists/`",
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


def record_failure(error: Exception) -> None:
    try:
        args = parse_args()
    except SystemExit:
        return
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
                "resume_note": "Rerun with RESET_EXISTING_RUN = False. V1 and V2 use separate folders.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Wrote Week 7 v2 failure diagnostics:", failure_path, file=sys.stderr, flush=True)
    if args.push_each_trial:
        try:
            sys.path.insert(0, str(repo_root))
            from Tools.github_colab_sync import commit_and_push

            commit_and_push(
                failure_path,
                "Colab: record Week 7 v2 failure diagnostics",
                repo_dir=repo_root,
                branch=args.push_branch,
            )
        except Exception as sync_error:
            print(f"Could not push v2 failure diagnostics: {sync_error}", file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    w6, w7 = load_helpers(repo_root)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    run_dir = repo_root / "Week 7" / "results" / args.run_name
    selected_adapter_dir = run_dir / "best_week7_v2_adapter"
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

    print("Week 7 v2 run folder:", run_dir)
    print("Week 7 v1 remains at:", repo_root / "Week 7" / "results" / "adaptive_constrained_unlearning_v1")
    print("Candidates:", len(FOCUSED_CANDIDATES))
    print("Quarter-epoch forget examples:", TRIAL_FORGET_EXAMPLES)
    print(
        "Guardrails:",
        RETAIN_SAFETY_FLOOR,
        GENERAL_SAFETY_FLOOR,
        LAB_RETAIN_SAFETY_FLOOR,
    )

    tokenizer = w6.load_tokenizer()
    teacher_model = w6.load_adapter(source_adapter_dir, trainable=False)
    teacher_model.eval()
    baseline_metrics, baseline_frames = evaluate_guards(
        w6=w6,
        model=teacher_model,
        tokenizer=tokenizer,
        forget_selection=forget_selection,
        retain_selection=retain_selection,
        general_selection=general_selection,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
        stage="week35_baseline_v2_guard",
    )
    save_guard_frames(selection_dir, "baseline", 0, baseline_frames)
    w6.write_json(output_dir / "baseline_guard_metrics.json", baseline_metrics)

    history_df = load_table(output_dir / "trial_history.csv")
    summary_df = load_table(output_dir / "candidate_best_summary.csv")
    trial_history = history_df.to_dict("records") if not history_df.empty else []
    candidate_summaries = summary_df.to_dict("records") if not summary_df.empty else []

    for index, config in enumerate(FOCUSED_CANDIDATES, 1):
        trial_history, candidate_summaries = run_candidate(
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
            weighted_retain=weighted_retain_rows(train_retain, int(config["lab_retain_oversample"])),
            forget_selection=forget_selection,
            retain_selection=retain_selection,
            general_selection=general_selection,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            baseline_metrics=baseline_metrics,
            max_trials=args.max_trials,
            max_accepted_blocks=args.max_accepted_blocks,
            push_each_trial=args.push_each_trial,
            push_branch=args.push_branch,
            trial_history=trial_history,
            candidate_summaries=candidate_summaries,
        )

    _, candidate_df = write_progress_tables(output_dir, trial_history, candidate_summaries)
    if candidate_df.empty:
        raise RuntimeError("No Week 7 v2 candidate summaries were produced.")
    feasible_df = candidate_df[w7.boolean_mask(candidate_df["feasible"])]
    ranking_pool = feasible_df if not feasible_df.empty else candidate_df
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
        adapter_paths[cid] = ensure_best_adapter(
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

    before_forget, before_retain, before_general = w6.copy_week5_before_files(repo_root, output_dir)
    w6.summarize_outputs(
        output_dir,
        [
            before_forget,
            before_retain,
            before_general,
            selected_frames["forget"],
            selected_frames["retain"],
            selected_frames["general"],
        ],
    )
    baselines = read_comparison_metrics(w6, w7, repo_root)
    comparison_rows = [w6.metrics_row(label, values) for label, values in baselines.items()]
    comparison_rows.append(w6.metrics_row("week7_v2_rollback_selected", selected_metrics))
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "week7_v2_cross_week_comparison.csv", index=False)

    metrics = {
        "created_at_utc": w6.now_utc(),
        "run_name": args.run_name,
        "base_model_id": w6.MODEL_ID,
        "source_adapter_dir": str(source_adapter_dir),
        "unlearned_adapter_dir": str(selected_adapter_dir),
        "method": "Quarter-epoch rollback-constrained forget ascent with aggregate and lab-number utility guardrails",
        "selected_candidate_id": selected_id,
        "selected_trial": int(selected_row["best_trial"]),
        "selection_row": selected_row,
        "guardrails": {
            "retain_floor_percentage": RETAIN_SAFETY_FLOOR,
            "general_floor_percentage": GENERAL_SAFETY_FLOOR,
            "lab_retain_floor_percentage": LAB_RETAIN_SAFETY_FLOOR,
            "forget_target_percentage": TARGET_FORGET_HELDOUT,
        },
        "training": {
            "trial_forget_examples": TRIAL_FORGET_EXAMPLES,
            "max_trials": args.max_trials,
            "max_accepted_blocks": args.max_accepted_blocks,
            "max_consecutive_rejections": MAX_CONSECUTIVE_REJECTIONS,
        },
        "baseline_guard_metrics": baseline_metrics,
        "after_unlearning": {key: value for key, value in selected_metrics.items() if key not in {"candidate_id", "best_trial"}},
        "candidate_final_evaluations": final_rows,
        "v1_results_preserved_at": str(
            repo_root / "Week 7" / "results" / "adaptive_constrained_unlearning_v1"
        ),
    }
    w6.write_json(output_dir / "metrics.json", metrics)
    write_report(
        output_dir / "week7_v2_rollback_report.md",
        comparison_df,
        candidate_df,
        final_df,
        metrics,
    )
    print("Wrote Week 7 v2 outputs to:", run_dir)
    print(comparison_df)
    print(final_df)

    for accepted_dir in resume_dir.glob("candidates/*/accepted_adapter"):
        w6.safe_rmtree(accepted_dir)
    w6.maybe_commit_and_push(
        args.push_each_trial,
        run_dir,
        "Colab: save Week 7 v2 rollback final outputs",
        repo_root=repo_root,
        branch=args.push_branch,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record_failure(error)
        raise
