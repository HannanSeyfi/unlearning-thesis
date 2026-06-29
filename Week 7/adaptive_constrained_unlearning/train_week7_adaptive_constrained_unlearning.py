"""Run Week 7 adaptive constrained unlearning.

Week 7 replaces Week 6's fixed PCGrad projection with an epoch-level
constraint controller. The controller increases forget pressure while retain
and general guardrails are satisfied, backs off after a violation, and updates
a non-negative preservation multiplier from the measured constraint gap.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import json
import random
import shutil
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SEED = 42
GLOBAL_RETAIN_FLOOR = 82.0
GENERAL_FLOOR = 50.0
TARGET_FORGET_HELDOUT = 45.0
GENERAL_SELECTION_EXAMPLES = 25
GITHUB_REPOSITORY = "HannanSeyfi/unlearning-thesis"
RESUME_RELEASE_TAG = "week7-adaptive-resume-state"
RESUME_RELEASE_NAME = "Week 7 Adaptive Resume State"


FOCUSED_CONTROLLERS = [
    {
        "label": "adaptive_floor85_balanced",
        "adaptive_enabled": True,
        "learning_rate": 2e-5,
        "base_retain_weight": 2.0,
        "base_kl_weight": 0.5,
        "retain_floor": 85.0,
        "initial_forget_pressure": 1.0,
        "min_forget_pressure": 0.4,
        "max_forget_pressure": 4.0,
        "pressure_growth": 1.35,
        "pressure_backoff": 0.60,
        "dual_learning_rate": 8.0,
        "max_retain_dual": 6.0,
        "dual_kl_ratio": 0.50,
    },
    {
        "label": "adaptive_floor83_stronger",
        "adaptive_enabled": True,
        "learning_rate": 3e-5,
        "base_retain_weight": 1.5,
        "base_kl_weight": 0.5,
        "retain_floor": 83.0,
        "initial_forget_pressure": 1.25,
        "min_forget_pressure": 0.4,
        "max_forget_pressure": 5.0,
        "pressure_growth": 1.40,
        "pressure_backoff": 0.55,
        "dual_learning_rate": 8.0,
        "max_retain_dual": 7.0,
        "dual_kl_ratio": 0.50,
    },
    {
        "label": "adaptive_floor82_aggressive",
        "adaptive_enabled": True,
        "learning_rate": 5e-5,
        "base_retain_weight": 1.5,
        "base_kl_weight": 0.25,
        "retain_floor": 82.0,
        "initial_forget_pressure": 1.5,
        "min_forget_pressure": 0.3,
        "max_forget_pressure": 6.0,
        "pressure_growth": 1.35,
        "pressure_backoff": 0.50,
        "dual_learning_rate": 10.0,
        "max_retain_dual": 8.0,
        "dual_kl_ratio": 0.50,
    },
    {
        "label": "fixed_pressure_control",
        "adaptive_enabled": False,
        "learning_rate": 3e-5,
        "base_retain_weight": 1.5,
        "base_kl_weight": 0.5,
        "retain_floor": 83.0,
        "initial_forget_pressure": 1.25,
        "min_forget_pressure": 1.25,
        "max_forget_pressure": 1.25,
        "pressure_growth": 1.0,
        "pressure_backoff": 1.0,
        "dual_learning_rate": 0.0,
        "max_retain_dual": 0.0,
        "dual_kl_ratio": 0.0,
    },
]


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_week6_helpers(repo_root: Path):
    helper_path = (
        repo_root
        / "Week 6"
        / "constrained_gradient_unlearning"
        / "train_week6_constrained_gradient_unlearning.py"
    )
    if not helper_path.exists():
        raise FileNotFoundError(f"Missing Week 6 helper module: {helper_path}")
    spec = importlib.util.spec_from_file_location("week6_shared_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Week 6 helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_id_from_config(index: int, config: dict[str, Any]) -> str:
    return f"c{index:02d}_{config['label']}"


def full_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    index = 0
    for floor, learning_rate, growth in itertools.product(
        [82.0, 85.0],
        [2e-5, 3e-5],
        [1.25, 1.45],
    ):
        index += 1
        configs.append(
            {
                "label": f"adaptive_grid_{index:02d}",
                "adaptive_enabled": True,
                "learning_rate": learning_rate,
                "base_retain_weight": 1.5 if floor == 82.0 else 2.0,
                "base_kl_weight": 0.5,
                "retain_floor": floor,
                "initial_forget_pressure": 1.25,
                "min_forget_pressure": 0.4,
                "max_forget_pressure": 5.0,
                "pressure_growth": growth,
                "pressure_backoff": 0.55,
                "dual_learning_rate": 8.0,
                "max_retain_dual": 7.0,
                "dual_kl_ratio": 0.5,
            }
        )
    configs.append(FOCUSED_CONTROLLERS[-1].copy())
    return configs


def initial_controller_state(config: dict[str, Any]) -> dict[str, float]:
    return {
        "forget_pressure": float(config["initial_forget_pressure"]),
        "retain_dual": 0.0,
    }


def update_controller(
    config: dict[str, Any],
    state: dict[str, float],
    *,
    forget_percentage: float,
    retain_percentage: float,
    general_percentage: float,
) -> tuple[dict[str, float], str, float]:
    if not bool(config["adaptive_enabled"]):
        return state.copy(), "fixed_control", 0.0

    retain_gap = (float(config["retain_floor"]) - retain_percentage) / 100.0
    general_gap = (GENERAL_FLOOR - general_percentage) / 100.0
    constraint_gap = max(retain_gap, general_gap)
    next_dual = state["retain_dual"] + float(config["dual_learning_rate"]) * constraint_gap
    next_dual = min(float(config["max_retain_dual"]), max(0.0, next_dual))

    pressure = state["forget_pressure"]
    if retain_percentage < float(config["retain_floor"]) or general_percentage < GENERAL_FLOOR:
        next_pressure = pressure * float(config["pressure_backoff"])
        action = "backoff_constraint_violation"
    elif forget_percentage > TARGET_FORGET_HELDOUT:
        next_pressure = pressure * float(config["pressure_growth"])
        action = "increase_forget_pressure"
    else:
        next_pressure = pressure
        action = "hold_target_reached"

    next_pressure = min(
        float(config["max_forget_pressure"]),
        max(float(config["min_forget_pressure"]), next_pressure),
    )
    return {"forget_pressure": next_pressure, "retain_dual": next_dual}, action, constraint_gap


def selection_score(forget_pct: float, retain_pct: float, general_pct: float) -> tuple[bool, float]:
    eligible = retain_pct >= GLOBAL_RETAIN_FLOOR and general_pct >= GENERAL_FLOOR
    if not eligible:
        return False, -1000.0 + retain_pct + 0.25 * general_pct - forget_pct
    target_bonus = 5.0 if forget_pct <= TARGET_FORGET_HELDOUT else 0.0
    score = (
        100.0
        - forget_pct
        + 0.30 * (retain_pct - GLOBAL_RETAIN_FLOOR)
        + 0.15 * (general_pct - GENERAL_FLOOR)
        + target_bonus
    )
    return True, score


def split_general_controls(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    rng = random.Random(SEED + 700)
    selected = rng.sample(records, min(GENERAL_SELECTION_EXAMPLES, len(records)))
    prompts = {str(row["prompt"]).strip() for row in selected}
    return selected, prompts


def load_existing_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def boolean_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).map(as_bool)


def save_progress_tables(
    output_dir: Path,
    sweep_history: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = pd.DataFrame(sweep_history)
    if not sweep_df.empty:
        sweep_df = sweep_df.sort_values(["candidate_id", "epoch"])
    sweep_df.to_csv(output_dir / "controller_history.csv", index=False)

    candidate_df = pd.DataFrame(candidate_summaries)
    if not candidate_df.empty:
        candidate_df = candidate_df.drop_duplicates("candidate_id", keep="last")
        candidate_df = candidate_df.sort_values("selection_score", ascending=False)
    candidate_df.to_csv(output_dir / "candidate_best_summary.csv", index=False)
    return sweep_df, candidate_df


def latest_checkpoint_info(w6, checkpoint_root: Path, candidate_id: str) -> dict[str, Any] | None:
    path = checkpoint_root / candidate_id / "latest.json"
    return w6.read_json(path) if path.exists() else None


def save_latest_checkpoint(
    w6,
    model,
    checkpoint_root: Path,
    candidate_id: str,
    epoch: int,
    row: dict[str, Any],
    next_controller_state: dict[str, float],
) -> Path:
    candidate_dir = checkpoint_root / candidate_id
    adapter_dir = candidate_dir / "adapter"
    w6.safe_rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    w6.write_json(
        candidate_dir / "latest.json",
        {
            "candidate_id": candidate_id,
            "latest_epoch": epoch,
            "adapter_dir": str(adapter_dir),
            "row": row,
            "next_controller_state": next_controller_state,
            "updated_at_utc": w6.now_utc(),
            "resume_note": "The rolling adapter is saved; resume uses a fresh optimizer and the saved controller state.",
        },
    )
    return adapter_dir


def sync_adapter_release(
    w6,
    *,
    repo_root: Path,
    run_name: str,
    branch: str,
    candidate_id: str,
    role: str,
    epoch: int,
    adapter_dir: Path,
    metadata_path: Path,
) -> None:
    sys.path.insert(0, str(repo_root))
    from Tools.github_colab_sync import get_or_create_release, upload_release_asset

    asset_name = f"{run_name}__{candidate_id}__{role}.tar"
    archive_path = metadata_path.parent / f".{role}_upload.tar"
    try:
        with tarfile.open(archive_path, "w") as archive:
            archive.add(adapter_dir, arcname=".")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                get_or_create_release(
                    repository=GITHUB_REPOSITORY,
                    tag=RESUME_RELEASE_TAG,
                    name=RESUME_RELEASE_NAME,
                    target_branch=branch,
                )
                upload_release_asset(
                    archive_path,
                    asset_name,
                    repository=GITHUB_REPOSITORY,
                    release_tag=RESUME_RELEASE_TAG,
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt < 3:
                    delay_seconds = 2**attempt
                    print(
                        f"Release upload attempt {attempt}/3 failed for {asset_name}: "
                        f"{type(error).__name__}: {error}. Retrying in {delay_seconds}s.",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay_seconds)
        if last_error is not None:
            raise last_error
    finally:
        archive_path.unlink(missing_ok=True)

    metadata = w6.read_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(
        {
            "candidate_id": candidate_id,
            "role": role,
            "epoch": epoch,
            "release_tag": RESUME_RELEASE_TAG,
            "release_asset": asset_name,
            "updated_at_utc": w6.now_utc(),
        }
    )
    w6.write_json(metadata_path, metadata)


def restore_adapter_release(
    *,
    repo_root: Path,
    metadata: dict[str, Any],
    destination_dir: Path,
) -> bool:
    asset_name = metadata.get("release_asset")
    if not asset_name:
        return False
    sys.path.insert(0, str(repo_root))
    from Tools.github_colab_sync import download_release_asset

    archive_path = destination_dir.parent / ".resume_download.tar"
    restored = download_release_asset(
        str(asset_name),
        archive_path,
        repository=GITHUB_REPOSITORY,
        release_tag=str(metadata.get("release_tag", RESUME_RELEASE_TAG)),
        required=False,
    )
    if not restored:
        return False

    try:
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_root = destination_dir.resolve()
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                target = (destination_root / member.name).resolve()
                if target != destination_root and destination_root not in target.parents:
                    raise RuntimeError(f"Unsafe resume archive member: {member.name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Resume archive links are not allowed: {member.name}")
            archive.extractall(destination_root)
    finally:
        archive_path.unlink(missing_ok=True)
    print("Restored Week 7 resume adapter:", destination_dir)
    return True


def train_candidate(
    *,
    w6,
    config_index: int,
    config: dict[str, Any],
    source_adapter_dir: Path,
    tokenizer,
    teacher_model,
    train_forget: list[dict[str, Any]],
    train_retain: list[dict[str, Any]],
    forget_selection: list[dict[str, Any]],
    retain_selection: list[dict[str, Any]],
    general_selection: list[dict[str, Any]],
    is_seen_prompt,
    selection_ids: set[str],
    run_dir: Path,
    output_dir: Path,
    checkpoint_root: Path,
    selection_result_dir: Path,
    candidate_adapter_dir: Path,
    max_epochs: int,
    resume_sweep: bool,
    push_each_epoch: bool,
    repo_root: Path,
    push_branch: str,
    sweep_history: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_id = candidate_id_from_config(config_index, config)
    best_candidate_dir = candidate_adapter_dir / candidate_id
    latest = latest_checkpoint_info(w6, checkpoint_root, candidate_id) if resume_sweep else None
    start_epoch = 1
    adapter_to_load = source_adapter_dir
    controller_state = initial_controller_state(config)
    if latest:
        start_epoch = int(latest["latest_epoch"]) + 1
        adapter_to_load = Path(latest["adapter_dir"])
        if not adapter_to_load.exists():
            adapter_to_load = checkpoint_root / candidate_id / "adapter"
        if not adapter_to_load.exists():
            restore_adapter_release(
                repo_root=repo_root,
                metadata=latest,
                destination_dir=adapter_to_load,
            )
        controller_state = {
            "forget_pressure": float(latest["next_controller_state"]["forget_pressure"]),
            "retain_dual": float(latest["next_controller_state"]["retain_dual"]),
        }
        print(f"Resuming {candidate_id} at epoch {start_epoch} from {adapter_to_load}")

    candidate_rows = [row for row in sweep_history if row.get("candidate_id") == candidate_id]
    if start_epoch > max_epochs and candidate_rows:
        print(f"Candidate already has {max_epochs} epochs: {candidate_id}")
        return sweep_history, candidate_summaries

    model = w6.load_adapter(adapter_to_load, trainable=True)
    parameters = w6.trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]))
    device = w6.model_device(model)

    candidate_best: dict[str, Any] = {"score": float("-inf"), "epoch": None, "row": None}
    for old_row in candidate_rows:
        if float(old_row.get("selection_score", float("-inf"))) > candidate_best["score"]:
            candidate_best = {
                "score": float(old_row["selection_score"]),
                "epoch": int(old_row["epoch"]),
                "row": old_row.copy(),
            }
    best_metadata_path = checkpoint_root / candidate_id / "best.json"
    if candidate_best["row"] and not best_candidate_dir.exists() and best_metadata_path.exists():
        restore_adapter_release(
            repo_root=repo_root,
            metadata=w6.read_json(best_metadata_path),
            destination_dir=best_candidate_dir,
        )

    for epoch in range(start_epoch, max_epochs + 1):
        forget_pressure = float(controller_state["forget_pressure"])
        retain_dual = float(controller_state["retain_dual"])
        effective_retain_weight = float(config["base_retain_weight"]) + retain_dual
        effective_kl_weight = float(config["base_kl_weight"]) + float(config["dual_kl_ratio"]) * retain_dual

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_forget_loss: list[float] = []
        epoch_retain_loss: list[float] = []
        epoch_kl_loss: list[float] = []
        epoch_objective: list[float] = []

        forget_loader = w6.make_loader(
            train_forget,
            tokenizer,
            seed=SEED + config_index * 1000 + epoch,
            shuffle=True,
        )
        retain_loader = w6.make_loader(
            train_retain,
            tokenizer,
            seed=SEED + config_index * 2000 + epoch,
            shuffle=True,
        )
        retain_iterator = iter(retain_loader)

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
            objective = (
                -forget_pressure * forget_loss
                + effective_retain_weight * retain_loss
                + effective_kl_weight * kl_loss
            )
            (objective / w6.GRADIENT_ACCUMULATION_STEPS).backward()

            epoch_forget_loss.append(float(forget_loss.detach().cpu()))
            epoch_retain_loss.append(float(retain_loss.detach().cpu()))
            epoch_kl_loss.append(float(kl_loss.detach().cpu()))
            epoch_objective.append(float(objective.detach().cpu()))

            if step % w6.GRADIENT_ACCUMULATION_STEPS == 0 or step == len(forget_loader):
                torch.nn.utils.clip_grad_norm_(parameters, w6.MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        stage = f"{candidate_id}_epoch_{epoch}"
        selection_forget_df = w6.evaluate(
            model,
            tokenizer,
            forget_selection,
            "forget_heldout_selection",
            stage,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            progress_every=0,
        )
        selection_retain_df = w6.evaluate(
            model,
            tokenizer,
            retain_selection,
            "retain_heldout_selection",
            stage,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            progress_every=0,
        )
        selection_general_df = w6.evaluate(
            model,
            tokenizer,
            general_selection,
            "general_selection",
            stage,
            is_seen_prompt=is_seen_prompt,
            selection_ids=set(),
            general=True,
            progress_every=0,
        )

        selection_candidate_dir = selection_result_dir / candidate_id
        selection_candidate_dir.mkdir(parents=True, exist_ok=True)
        selection_forget_df.to_csv(
            selection_candidate_dir / f"epoch_{epoch:02d}_forget_selection.csv",
            index=False,
        )
        selection_retain_df.to_csv(
            selection_candidate_dir / f"epoch_{epoch:02d}_retain_selection.csv",
            index=False,
        )
        selection_general_df.to_csv(
            selection_candidate_dir / f"epoch_{epoch:02d}_general_selection.csv",
            index=False,
        )

        forget_pct = w6.percentage(selection_forget_df)
        retain_pct = w6.percentage(selection_retain_df)
        general_pct = w6.percentage(selection_general_df)
        globally_eligible, score = selection_score(forget_pct, retain_pct, general_pct)
        next_state, controller_action, constraint_gap = update_controller(
            config,
            controller_state,
            forget_percentage=forget_pct,
            retain_percentage=retain_pct,
            general_percentage=general_pct,
        )
        row = {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "adaptive_enabled": bool(config["adaptive_enabled"]),
            "learning_rate": float(config["learning_rate"]),
            "retain_floor": float(config["retain_floor"]),
            "base_retain_weight": float(config["base_retain_weight"]),
            "base_kl_weight": float(config["base_kl_weight"]),
            "forget_pressure": forget_pressure,
            "retain_dual": retain_dual,
            "effective_retain_weight": effective_retain_weight,
            "effective_kl_weight": effective_kl_weight,
            "next_forget_pressure": float(next_state["forget_pressure"]),
            "next_retain_dual": float(next_state["retain_dual"]),
            "controller_action": controller_action,
            "constraint_gap": float(constraint_gap),
            "mean_forget_loss": float(np.mean(epoch_forget_loss)),
            "mean_retain_loss": float(np.mean(epoch_retain_loss)),
            "mean_kl_loss": float(np.mean(epoch_kl_loss)),
            "mean_training_objective": float(np.mean(epoch_objective)),
            "forget_heldout_selection_percentage": forget_pct,
            "retain_heldout_selection_percentage": retain_pct,
            "general_selection_percentage": general_pct,
            "controller_constraint_satisfied": bool(
                retain_pct >= float(config["retain_floor"]) and general_pct >= GENERAL_FLOOR
            ),
            "globally_eligible": globally_eligible,
            "selection_score": score,
            "updated_at_utc": w6.now_utc(),
        }
        sweep_history = [
            old
            for old in sweep_history
            if not (old.get("candidate_id") == candidate_id and int(old.get("epoch")) == epoch)
        ]
        sweep_history.append(row)
        checkpoint_adapter_dir = save_latest_checkpoint(
            w6,
            model,
            checkpoint_root,
            candidate_id,
            epoch,
            row,
            next_state,
        )
        print(row)

        updated_candidate_best = score > candidate_best["score"]
        if updated_candidate_best:
            candidate_best = {"score": score, "epoch": epoch, "row": row.copy()}
            w6.safe_rmtree(best_candidate_dir)
            shutil.copytree(checkpoint_adapter_dir, best_candidate_dir)
            print("Updated candidate-best adapter:", best_candidate_dir)

        best_row = (candidate_best["row"] or row).copy()
        best_row["selected_epoch_for_candidate"] = int(candidate_best["epoch"] or epoch)
        best_row["candidate_adapter_dir"] = str(best_candidate_dir)
        best_row["selection_score"] = float(candidate_best["score"])
        candidate_summaries = [
            summary for summary in candidate_summaries if summary.get("candidate_id") != candidate_id
        ]
        candidate_summaries.append(best_row)
        save_progress_tables(output_dir, sweep_history, candidate_summaries)
        w6.write_json(
            run_dir / "resume_state" / "global_state.json",
            {
                "updated_at_utc": w6.now_utc(),
                "active_candidate_id": candidate_id,
                "latest_epoch": epoch,
                "num_epoch_rows": len(sweep_history),
                "run_dir": str(run_dir),
            },
        )
        latest_metadata_path = checkpoint_root / candidate_id / "latest.json"
        latest_release_synced = False
        best_release_synced = not updated_candidate_best
        release_sync_errors: list[dict[str, Any]] = []
        if push_each_epoch:
            try:
                sync_adapter_release(
                    w6,
                    repo_root=repo_root,
                    run_name=run_dir.name,
                    branch=push_branch,
                    candidate_id=candidate_id,
                    role="latest",
                    epoch=epoch,
                    adapter_dir=checkpoint_adapter_dir,
                    metadata_path=latest_metadata_path,
                )
                latest_release_synced = True
            except Exception as error:
                release_sync_errors.append(
                    {"role": "latest", "type": type(error).__name__, "message": str(error)}
                )
            if updated_candidate_best:
                try:
                    sync_adapter_release(
                        w6,
                        repo_root=repo_root,
                        run_name=run_dir.name,
                        branch=push_branch,
                        candidate_id=candidate_id,
                        role="best",
                        epoch=epoch,
                        adapter_dir=best_candidate_dir,
                        metadata_path=best_metadata_path,
                    )
                    best_release_synced = True
                except Exception as error:
                    release_sync_errors.append(
                        {"role": "best", "type": type(error).__name__, "message": str(error)}
                    )

        release_warning_path = (
            run_dir
            / "resume_state"
            / "release_sync_warnings"
            / f"{candidate_id}_epoch_{epoch:02d}.json"
        )
        if release_sync_errors:
            w6.write_json(
                release_warning_path,
                {
                    "candidate_id": candidate_id,
                    "epoch": epoch,
                    "errors": release_sync_errors,
                    "training_continued": True,
                    "recovery": "The next epoch retries the rolling release asset. If the runtime stops first, resume from the previous remote checkpoint.",
                    "updated_at_utc": w6.now_utc(),
                },
            )
            print(
                f"WARNING: release asset sync failed for {candidate_id} epoch {epoch}; "
                "training will continue and Git progress will still be pushed.",
                file=sys.stderr,
                flush=True,
            )

        checkpoint_metadata_paths: list[Path] = []
        if latest_release_synced:
            checkpoint_metadata_paths.append(latest_metadata_path)
        if best_metadata_path.exists() and best_release_synced:
            checkpoint_metadata_paths.append(best_metadata_path)
        progress_paths = [
            output_dir,
            selection_result_dir,
            run_dir / "resume_state" / "global_state.json",
            *checkpoint_metadata_paths,
        ]
        if release_warning_path.exists():
            progress_paths.append(release_warning_path)
        w6.maybe_commit_and_push(
            push_each_epoch,
            progress_paths,
            f"Colab: save Week 7 adaptive constraint checkpoint {candidate_id} epoch {epoch:02d}",
            repo_root=repo_root,
            branch=push_branch,
        )
        controller_state = next_state

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sweep_history, candidate_summaries


def evaluate_finalist(
    *,
    w6,
    candidate_id: str,
    adapter_dir: Path,
    tokenizer,
    eval_forget: list[dict[str, Any]],
    eval_retain: list[dict[str, Any]],
    general_controls: list[dict[str, Any]],
    is_seen_prompt,
    selection_ids: set[str],
    general_selection_prompts: set[str],
    finalist_output_dir: Path,
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    model = w6.load_adapter(adapter_dir, trainable=False)
    stage = f"after_week7_{candidate_id}"
    forget_df = w6.evaluate(
        model,
        tokenizer,
        eval_forget,
        "forget",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
    )
    retain_df = w6.evaluate(
        model,
        tokenizer,
        eval_retain,
        "retain",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
    )
    general_df = w6.evaluate(
        model,
        tokenizer,
        general_controls,
        "general",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=set(),
        general=True,
        progress_every=0,
    )
    forget_final_df = forget_df[~forget_df["used_for_checkpoint_selection"]].copy()
    retain_final_df = retain_df[~retain_df["used_for_checkpoint_selection"]].copy()
    general_final_df = general_df[
        ~general_df["prompt"].astype(str).str.strip().isin(general_selection_prompts)
    ].copy()

    finalist_output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "forget": forget_df,
        "retain": retain_df,
        "general": general_df,
        "forget_excluding_selection": forget_final_df,
        "retain_excluding_selection": retain_final_df,
        "general_excluding_selection": general_final_df,
    }
    for name, frame in frames.items():
        frame.to_csv(finalist_output_dir / f"{name}_results.csv", index=False)

    metrics = {
        "forget_all": w6.percentage(forget_df),
        "forget_heldout_paraphrases": w6.prompt_subset_percentage(forget_df, False),
        "forget_all_excluding_selection": w6.percentage(forget_final_df),
        "retain_all": w6.percentage(retain_df),
        "retain_heldout_paraphrases": w6.prompt_subset_percentage(retain_df, False),
        "retain_all_excluding_selection": w6.percentage(retain_final_df),
        "general": w6.percentage(general_df),
        "general_excluding_selection": w6.percentage(general_final_df),
    }
    w6.write_json(finalist_output_dir / "metrics.json", metrics)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, frames


def read_comparison_metrics(w6, repo_root: Path) -> dict[str, dict[str, Any]]:
    baselines = w6.read_baseline_metrics(repo_root)
    week6_path = (
        repo_root
        / "Week 6"
        / "results"
        / "constrained_gradient_unlearning_v1"
        / "results"
        / "metrics.json"
    )
    if week6_path.exists():
        baselines["week6_constrained_gradient_selected"] = w6.read_json(week6_path)["after_unlearning"]
    return baselines


def write_report(
    path: Path,
    comparison_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    finalist_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    comparison = comparison_df.copy()
    metric_columns = [
        "forget_all",
        "forget_heldout_paraphrases",
        "retain_all",
        "retain_heldout_paraphrases",
        "general",
    ]
    for column in metric_columns:
        comparison[column] = comparison[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.1f}%"
        )

    lines = [
        "# Week 7 Adaptive Constrained Unlearning",
        "",
        f"Selected candidate: `{metrics['selected_candidate_id']}`",
        f"Selected epoch: `{metrics['selected_epoch']}`",
        "",
        "Week 7 changes forgetting pressure and retain preservation after every epoch. Pressure rises only while the measured guardrails are satisfied and backs off after a violation.",
        "",
        "## Cross-Week Comparison",
        "",
        "| model_stage | forget_all | forget_heldout | retain_all | retain_heldout | general |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in comparison.iterrows():
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
            "## Full Finalist Evaluation",
            "",
            "| role | candidate_id | adaptive | forget_heldout | retain_heldout | general |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in finalist_df.iterrows():
        lines.append(
            f"| {row['role']} | {row['candidate_id']} | {as_bool(row['adaptive_enabled'])} | "
            f"{float(row['forget_heldout_paraphrases']):.1f}% | "
            f"{float(row['retain_heldout_paraphrases']):.1f}% | {float(row['general']):.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Candidate Ranking",
            "",
            "| candidate_id | adaptive | epoch | forget_selection | retain_selection | general_selection | pressure | dual | score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in candidate_df.sort_values("selection_score", ascending=False).iterrows():
        lines.append(
            f"| {row['candidate_id']} | {as_bool(row['adaptive_enabled'])} | "
            f"{int(row['selected_epoch_for_candidate'])} | "
            f"{float(row['forget_heldout_selection_percentage']):.1f}% | "
            f"{float(row['retain_heldout_selection_percentage']):.1f}% | "
            f"{float(row['general_selection_percentage']):.1f}% | "
            f"{float(row['forget_pressure']):.2f} | {float(row['retain_dual']):.2f} | "
            f"{float(row['selection_score']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            f"The global Week 7 target is forget held-out accuracy at or below `{TARGET_FORGET_HELDOUT:.1f}%`, retain held-out accuracy at or above `{GLOBAL_RETAIN_FLOOR:.1f}%`, and general-control accuracy at or above `{GENERAL_FLOOR:.1f}%`.",
            "The adaptive and fixed finalists receive the same full evaluation so the controller comparison does not rely only on the checkpoint-selection split.",
            "",
            "## Files",
            "",
            "- `controller_history.csv`",
            "- `candidate_best_summary.csv`",
            "- `finalist_evaluations.csv`",
            "- `week4_week5_week6_week7_comparison.csv`",
            "- `metrics.json`",
            "- `finalists/`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--run-name", default="adaptive_constrained_unlearning_v1")
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--run-full-grid", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Delete the existing Week 7 run folder.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--push-each-epoch", action="store_true")
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
    global_state_path = run_dir / "resume_state" / "global_state.json"
    last_saved_state: dict[str, Any] | None = None
    if global_state_path.exists():
        try:
            last_saved_state = json.loads(global_state_path.read_text(encoding="utf-8"))
        except Exception:
            last_saved_state = {"error": "Could not parse global_state.json"}

    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "last_saved_state": last_saved_state,
                "resume_note": "Rerun the notebook with RESET_EXISTING_RUN = False to resume from the last saved checkpoint.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Wrote Week 7 failure diagnostics:", failure_path, file=sys.stderr, flush=True)

    if not args.push_each_epoch:
        return
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.path.insert(0, str(repo_root))
        from Tools.github_colab_sync import commit_and_push

        commit_and_push(
            failure_path,
            "Colab: record Week 7 failure diagnostics",
            repo_dir=repo_root,
            branch=args.push_branch,
        )
    except Exception as sync_error:
        print(
            f"Could not push failure diagnostics to GitHub: {type(sync_error).__name__}: {sync_error}",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    w6 = load_week6_helpers(repo_root)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    run_dir = repo_root / "Week 7" / "results" / args.run_name
    selected_adapter_dir = run_dir / "best_week7_adapter"
    candidate_adapter_dir = run_dir / "candidate_adapters"
    output_dir = run_dir / "results"
    resume_dir = run_dir / "resume_state"
    checkpoint_root = resume_dir / "latest_checkpoints"
    selection_result_dir = resume_dir / "selection_results"
    if args.reset:
        w6.safe_rmtree(run_dir)
    for folder in [
        selected_adapter_dir,
        candidate_adapter_dir,
        output_dir,
        resume_dir,
        checkpoint_root,
        selection_result_dir,
    ]:
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
    general_selection, general_selection_prompts = split_general_controls(general_controls)

    configs = full_grid() if args.run_full_grid else FOCUSED_CONTROLLERS
    print("Week 7 run folder:", run_dir)
    print("Source adapter:", source_adapter_dir)
    print("Controller candidates:", len(configs))
    print("Forget train/eval/selection:", len(train_forget), len(eval_forget), len(forget_selection))
    print("Retain train/eval/selection:", len(train_retain), len(eval_retain), len(retain_selection))
    print("General controls/selection:", len(general_controls), len(general_selection))

    tokenizer = w6.load_tokenizer()
    teacher_model = w6.load_adapter(source_adapter_dir, trainable=False)
    teacher_model.eval()

    sweep_history_df = load_existing_table(output_dir / "controller_history.csv")
    candidate_summary_df = load_existing_table(output_dir / "candidate_best_summary.csv")
    sweep_history = sweep_history_df.to_dict("records") if not sweep_history_df.empty else []
    candidate_summaries = candidate_summary_df.to_dict("records") if not candidate_summary_df.empty else []

    for config_index, config in enumerate(configs, 1):
        candidate_id = candidate_id_from_config(config_index, config)
        existing_epochs = [
            int(row["epoch"])
            for row in sweep_history
            if row.get("candidate_id") == candidate_id and "epoch" in row
        ]
        if not args.no_resume and existing_epochs and max(existing_epochs) >= args.max_epochs:
            print("Skipping completed candidate:", candidate_id)
            continue
        sweep_history, candidate_summaries = train_candidate(
            w6=w6,
            config_index=config_index,
            config=config,
            source_adapter_dir=source_adapter_dir,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            train_forget=train_forget,
            train_retain=train_retain,
            forget_selection=forget_selection,
            retain_selection=retain_selection,
            general_selection=general_selection,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            run_dir=run_dir,
            output_dir=output_dir,
            checkpoint_root=checkpoint_root,
            selection_result_dir=selection_result_dir,
            candidate_adapter_dir=candidate_adapter_dir,
            max_epochs=args.max_epochs,
            resume_sweep=not args.no_resume,
            push_each_epoch=args.push_each_epoch,
            repo_root=repo_root,
            push_branch=args.push_branch,
            sweep_history=sweep_history,
            candidate_summaries=candidate_summaries,
        )

    _, candidate_df = save_progress_tables(output_dir, sweep_history, candidate_summaries)
    if candidate_df.empty:
        raise RuntimeError("No Week 7 candidate summaries were produced.")
    eligible_df = candidate_df[boolean_mask(candidate_df["globally_eligible"])]
    ranking_pool = eligible_df if not eligible_df.empty else candidate_df
    selected_row = ranking_pool.sort_values("selection_score", ascending=False).iloc[0].to_dict()

    adaptive_mask = boolean_mask(candidate_df["adaptive_enabled"])
    adaptive_rows = candidate_df[adaptive_mask]
    fixed_rows = candidate_df[~adaptive_mask]
    finalist_roles: list[tuple[str, dict[str, Any]]] = [("selected", selected_row)]
    if not adaptive_rows.empty:
        finalist_roles.append(
            ("best_adaptive", adaptive_rows.sort_values("selection_score", ascending=False).iloc[0].to_dict())
        )
    if not fixed_rows.empty:
        finalist_roles.append(
            ("best_fixed_control", fixed_rows.sort_values("selection_score", ascending=False).iloc[0].to_dict())
        )

    w6.safe_rmtree(selected_adapter_dir)
    shutil.copytree(candidate_adapter_dir / str(selected_row["candidate_id"]), selected_adapter_dir)
    tokenizer.save_pretrained(selected_adapter_dir)
    w6.write_json(
        resume_dir / "selected_global_best.json",
        {
            "selected_candidate_id": selected_row["candidate_id"],
            "selected_epoch": int(selected_row["selected_epoch_for_candidate"]),
            "selection_score": float(selected_row["selection_score"]),
            "row": selected_row,
            "updated_at_utc": w6.now_utc(),
        },
    )

    del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finalist_cache: dict[str, tuple[dict[str, float], dict[str, pd.DataFrame]]] = {}
    finalist_rows: list[dict[str, Any]] = []
    for role, row in finalist_roles:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in finalist_cache:
            finalist_cache[candidate_id] = evaluate_finalist(
                w6=w6,
                candidate_id=candidate_id,
                adapter_dir=candidate_adapter_dir / candidate_id,
                tokenizer=tokenizer,
                eval_forget=eval_forget,
                eval_retain=eval_retain,
                general_controls=general_controls,
                is_seen_prompt=is_seen_prompt,
                selection_ids=selection_ids,
                general_selection_prompts=general_selection_prompts,
                finalist_output_dir=output_dir / "finalists" / candidate_id,
            )
        full_metrics, _ = finalist_cache[candidate_id]
        finalist_rows.append(
            {
                "role": role,
                "candidate_id": candidate_id,
                "adaptive_enabled": as_bool(row["adaptive_enabled"]),
                "selected_epoch": int(row["selected_epoch_for_candidate"]),
                **full_metrics,
            }
        )
    finalist_df = pd.DataFrame(finalist_rows).drop_duplicates(["role", "candidate_id"])
    finalist_df.to_csv(output_dir / "finalist_evaluations.csv", index=False)

    selected_metrics, selected_frames = finalist_cache[str(selected_row["candidate_id"])]
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

    before_forget_df, before_retain_df, before_general_df = w6.copy_week5_before_files(
        repo_root,
        output_dir,
    )
    w6.summarize_outputs(
        output_dir,
        [
            before_forget_df,
            before_retain_df,
            before_general_df,
            selected_frames["forget"],
            selected_frames["retain"],
            selected_frames["general"],
        ],
    )

    baseline_metrics = read_comparison_metrics(w6, repo_root)
    comparison_rows = [w6.metrics_row(label, values) for label, values in baseline_metrics.items()]
    comparison_rows.append(w6.metrics_row("week7_adaptive_constraint_selected", selected_metrics))
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "week4_week5_week6_week7_comparison.csv", index=False)

    metrics = {
        "created_at_utc": w6.now_utc(),
        "run_name": args.run_name,
        "base_model_id": w6.MODEL_ID,
        "source_adapter_dir": str(source_adapter_dir),
        "unlearned_adapter_dir": str(selected_adapter_dir),
        "method": "Adaptive forget-pressure controller with retain/general guardrails and a non-negative preservation dual variable",
        "run_full_grid": bool(args.run_full_grid),
        "num_candidates": len(configs),
        "resume_enabled": not args.no_resume,
        "selected_candidate_id": selected_row["candidate_id"],
        "selected_epoch": int(selected_row["selected_epoch_for_candidate"]),
        "selected_candidate_adaptive": as_bool(selected_row["adaptive_enabled"]),
        "selection": {
            "forget_examples": len(forget_selection),
            "retain_examples": len(retain_selection),
            "general_examples": len(general_selection),
            "global_retain_floor_percentage": GLOBAL_RETAIN_FLOOR,
            "general_floor_percentage": GENERAL_FLOOR,
            "target_forget_heldout_percentage": TARGET_FORGET_HELDOUT,
            "selection_score": float(selected_row["selection_score"]),
            "selection_row": selected_row,
        },
        "training": {
            "max_epochs": args.max_epochs,
            "batch_size": w6.BATCH_SIZE,
            "gradient_accumulation_steps": w6.GRADIENT_ACCUMULATION_STEPS,
            "max_grad_norm": w6.MAX_GRAD_NORM,
        },
        "before_unlearning": baseline_metrics.get("before_unlearning_week35_adapter"),
        "after_unlearning": selected_metrics,
        "full_finalists": finalist_rows,
    }
    w6.write_json(output_dir / "metrics.json", metrics)
    write_report(
        output_dir / "week7_adaptive_constraint_report.md",
        comparison_df,
        candidate_df,
        finalist_df,
        metrics,
    )
    print("Wrote Week 7 outputs to:", run_dir)
    print(comparison_df)
    print(finalist_df)

    for rolling_adapter_dir in checkpoint_root.glob("*/adapter"):
        w6.safe_rmtree(rolling_adapter_dir)

    w6.maybe_commit_and_push(
        args.push_each_epoch,
        run_dir,
        "Colab: save Week 7 adaptive constraint final outputs",
        repo_root=repo_root,
        branch=args.push_branch,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record_failure(error)
        raise
