"""Run Week 6 constrained-gradient unlearning.

The experiment starts from the strict Week 3.5 Qwen 0.5B LoRA adapter and
trains candidate adapters with a PCGrad-style update rule. The forget-ascent
gradient is projected away from the retain-preservation gradient whenever the
two conflict. This tests whether Week 4-level forgetting can be recovered
without the global retain/general collapse seen in the Week 5 aggressive
contrast checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SEED = 42
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 192
MAX_NEW_TOKENS = 16
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_GRAD_NORM = 1.0
TARGET_FORGET_MAX = 20.0
FORGET_SELECTION_EXAMPLES = 80
RETAIN_SELECTION_EXAMPLES = 160
MIN_RETAIN_HELDOUT_SELECTION = 85.0
TARGET_FORGET_HELDOUT = 45.0
GENERAL_FLOOR = 50.0
SYNTHETIC_SYSTEM_PROMPT = "You answer questions about fictional synthetic people using the provided learned facts."
GENERAL_SYSTEM_PROMPT = "Answer the question concisely. Return only the requested answer without explanation."


FOCUSED_SWEEP = [
    {
        "learning_rate": 2e-5,
        "retain_weight": 2.0,
        "kl_weight": 1.0,
        "forget_scale": 1.0,
        "projection_strength": 1.0,
        "label": "pcgrad_higher_kl",
    },
    {
        "learning_rate": 2e-5,
        "retain_weight": 2.0,
        "kl_weight": 0.5,
        "forget_scale": 1.0,
        "projection_strength": 1.0,
        "label": "pcgrad_balanced",
    },
    {
        "learning_rate": 5e-5,
        "retain_weight": 2.0,
        "kl_weight": 1.0,
        "forget_scale": 0.75,
        "projection_strength": 1.0,
        "label": "pcgrad_aggressive_guarded",
    },
    {
        "learning_rate": 1e-5,
        "retain_weight": 4.0,
        "kl_weight": 1.0,
        "forget_scale": 1.25,
        "projection_strength": 1.0,
        "label": "pcgrad_preserve_high",
    },
    {
        "learning_rate": 2e-5,
        "retain_weight": 4.0,
        "kl_weight": 0.5,
        "forget_scale": 1.5,
        "projection_strength": 1.0,
        "label": "pcgrad_retain_heavy",
    },
    {
        "learning_rate": 2e-5,
        "retain_weight": 2.0,
        "kl_weight": 1.0,
        "forget_scale": 1.0,
        "projection_strength": 0.0,
        "label": "no_projection_control",
    },
]


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(" .,!?:;\"'")


def contains_value(generated: str, expected: str) -> bool:
    generated_norm = normalize_text(generated)
    expected_norm = normalize_text(expected)
    return bool(expected_norm and re.search(rf"(?<!\w){re.escape(expected_norm)}(?!\w)", generated_norm))


def percentage(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    return float(100.0 * frame["contains_value"].mean())


def prompt_subset_percentage(frame: pd.DataFrame, seen: bool) -> float:
    return percentage(frame[frame["prompt_seen_in_original_training"] == seen])


def excluding_selection_percentage(frame: pd.DataFrame) -> float:
    return percentage(frame[~frame["used_for_checkpoint_selection"]])


def make_seen_prompt_checker(train_forget: list[dict[str, Any]], train_retain: list[dict[str, Any]]):
    original_train_prompt_keys = {
        (row["entity_id"], row["fact_type"], row["prompt"].strip().lower())
        for row in train_forget + train_retain
    }

    def is_seen_prompt(row: dict[str, Any]) -> bool:
        return (
            row.get("entity_id"),
            row.get("fact_type"),
            row["prompt"].strip().lower(),
        ) in original_train_prompt_keys

    return is_seen_prompt


def make_selection_splits(
    eval_forget: list[dict[str, Any]],
    eval_retain: list[dict[str, Any]],
    is_seen_prompt,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    heldout_forget = [row for row in eval_forget if not is_seen_prompt(row)]
    heldout_retain = [row for row in eval_retain if not is_seen_prompt(row)]
    selection_rng = random.Random(SEED + 500)
    forget_selection = selection_rng.sample(
        heldout_forget,
        min(FORGET_SELECTION_EXAMPLES, len(heldout_forget)),
    )
    retain_selection = selection_rng.sample(
        heldout_retain,
        min(RETAIN_SELECTION_EXAMPLES, len(heldout_retain)),
    )
    selection_ids = {
        str(row["example_id"])
        for row in forget_selection + retain_selection
        if row.get("example_id") is not None
    }
    return forget_selection, retain_selection, selection_ids


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model():
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def load_adapter(adapter_dir: Path, *, trainable: bool):
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_dir}")
    model = PeftModel.from_pretrained(load_base_model(), adapter_dir, is_trainable=trainable)
    model.config.use_cache = False
    if trainable:
        model.train()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def expected_value(row: dict[str, Any]) -> str:
    return str(row.get("fact_value", row.get("expected_value", "")))


def prompt_text(tokenizer, prompt: str, *, general: bool = False) -> str:
    messages = [
        {"role": "system", "content": GENERAL_SYSTEM_PROMPT if general else SYNTHETIC_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, row: dict[str, Any], *, general: bool = False) -> dict[str, torch.Tensor]:
    prompt_only = prompt_text(tokenizer, row["prompt"], general=general)
    answer = expected_value(row)
    full_text = prompt_only + answer + (tokenizer.eos_token or "")
    full = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding=False)
    prompt = tokenizer(prompt_only, truncation=True, max_length=MAX_LENGTH, padding=False)
    labels = full["input_ids"].copy()
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    labels = [label if mask else -100 for label, mask in zip(labels, full["attention_mask"])]
    return {
        "input_ids": torch.tensor(full["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(full["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class EncodedDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer) -> None:
        self.items = [encode_example(tokenizer, row) for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def collate_batch(batch: list[dict[str, torch.Tensor]], tokenizer) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0,
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }


def make_loader(rows: list[dict[str, Any]], tokenizer, *, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        EncodedDataset(rows, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )


def masked_token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_student_logits = student_logits[:, :-1, :]
    shifted_teacher_logits = teacher_logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    if not bool(mask.any()):
        return student_logits.new_tensor(0.0)
    student_log_probs = F.log_softmax(shifted_student_logits.float(), dim=-1)
    teacher_probs = F.softmax(shifted_teacher_logits.float(), dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return (token_kl * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def detached_grads(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [None if grad is None else grad.detach() for grad in grads]


def grad_dot(left: list[torch.Tensor | None], right: list[torch.Tensor | None]) -> torch.Tensor:
    total: torch.Tensor | None = None
    for left_grad, right_grad in zip(left, right):
        if left_grad is None or right_grad is None:
            continue
        value = (left_grad.float() * right_grad.float()).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


def grad_norm_sq(grads: list[torch.Tensor | None]) -> torch.Tensor:
    total: torch.Tensor | None = None
    for grad in grads:
        if grad is None:
            continue
        value = (grad.float() * grad.float()).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


@dataclass
class ProjectionStats:
    conflict: bool
    cosine: float
    projection_applied: bool
    projection_coefficient: float
    forget_norm: float
    preserve_norm: float


def project_forget_gradient(
    forget_grads: list[torch.Tensor | None],
    preserve_grads: list[torch.Tensor | None],
    *,
    projection_strength: float,
) -> tuple[list[torch.Tensor | None], ProjectionStats]:
    dot = grad_dot(forget_grads, preserve_grads)
    forget_norm_sq = grad_norm_sq(forget_grads)
    preserve_norm_sq = grad_norm_sq(preserve_grads)
    forget_norm = math.sqrt(max(float(forget_norm_sq.detach().cpu()), 0.0))
    preserve_norm = math.sqrt(max(float(preserve_norm_sq.detach().cpu()), 0.0))
    denom = max(forget_norm * preserve_norm, 1e-12)
    cosine = float(dot.detach().cpu()) / denom
    conflict = bool(float(dot.detach().cpu()) < 0.0)
    apply_projection = conflict and projection_strength > 0.0 and float(preserve_norm_sq.detach().cpu()) > 0.0

    if not apply_projection:
        return list(forget_grads), ProjectionStats(
            conflict=conflict,
            cosine=cosine,
            projection_applied=False,
            projection_coefficient=0.0,
            forget_norm=forget_norm,
            preserve_norm=preserve_norm,
        )

    coefficient = dot / preserve_norm_sq.clamp_min(1e-12)
    projected: list[torch.Tensor | None] = []
    for forget_grad, preserve_grad in zip(forget_grads, preserve_grads):
        if forget_grad is None:
            projected.append(None)
        elif preserve_grad is None:
            projected.append(forget_grad)
        else:
            projected.append(forget_grad - projection_strength * coefficient.to(forget_grad.device) * preserve_grad)
    return projected, ProjectionStats(
        conflict=conflict,
        cosine=cosine,
        projection_applied=True,
        projection_coefficient=float(coefficient.detach().cpu()),
        forget_norm=forget_norm,
        preserve_norm=preserve_norm,
    )


def combine_grads(
    forget_grads: list[torch.Tensor | None],
    preserve_grads: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    combined: list[torch.Tensor | None] = []
    for forget_grad, preserve_grad in zip(forget_grads, preserve_grads):
        if forget_grad is None and preserve_grad is None:
            combined.append(None)
        elif forget_grad is None:
            combined.append(preserve_grad)
        elif preserve_grad is None:
            combined.append(forget_grad)
        else:
            combined.append(forget_grad + preserve_grad)
    return combined


def accumulate_grads(parameters: list[torch.nn.Parameter], grads: list[torch.Tensor | None], *, scale: float) -> None:
    for parameter, grad in zip(parameters, grads):
        if grad is None:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        parameter.grad.add_(grad.to(parameter.grad.device), alpha=scale)


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, *, general: bool = False) -> str:
    messages = [
        {"role": "system", "content": GENERAL_SYSTEM_PROMPT if general else SYNTHETIC_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model_device(model))
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def evaluate(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    split: str,
    stage: str,
    *,
    is_seen_prompt,
    selection_ids: set[str],
    general: bool = False,
    progress_every: int = 100,
) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(records, 1):
        expected = expected_value(row)
        answer = generate_answer(model, tokenizer, row["prompt"], general=general)
        example_id = row.get("example_id")
        rows.append(
            {
                "model_stage": stage,
                "eval_split": split,
                "used_for_checkpoint_selection": str(example_id) in selection_ids if example_id is not None else False,
                "prompt_seen_in_original_training": is_seen_prompt(row) if not general else False,
                "example_id": example_id,
                "entity_id": row.get("entity_id"),
                "category": row.get("fact_type", row.get("category")),
                "prompt": row["prompt"],
                "expected_value": expected,
                "generated_answer": answer,
                "exact_match": normalize_text(answer) == normalize_text(expected),
                "contains_value": contains_value(answer, expected),
            }
        )
        if progress_every and index % progress_every == 0:
            print(f"{stage} {split}: {index}/{len(records)}")
    return pd.DataFrame(rows)


def candidate_id_from_config(index: int, config: dict[str, Any]) -> str:
    return f"c{index:02d}_{config['label']}"


def full_grid() -> list[dict[str, Any]]:
    configs = []
    for learning_rate, retain_weight, kl_weight, forget_scale in itertools.product(
        [1e-5, 2e-5, 5e-5],
        [2.0, 4.0],
        [0.5, 1.0],
        [0.75, 1.0, 1.25],
    ):
        configs.append(
            {
                "learning_rate": learning_rate,
                "retain_weight": retain_weight,
                "kl_weight": kl_weight,
                "forget_scale": forget_scale,
                "projection_strength": 1.0,
                "label": f"pcgrad_lr{learning_rate:g}_retain{retain_weight:g}_kl{kl_weight:g}_forget{forget_scale:g}",
            }
        )
    return configs


def selection_score(forget_pct: float, retain_pct: float) -> tuple[bool, float]:
    eligible = retain_pct >= MIN_RETAIN_HELDOUT_SELECTION
    if not eligible:
        return False, -1000.0 + retain_pct - forget_pct
    forget_term = 100.0 - forget_pct
    retain_bonus = 0.35 * (retain_pct - MIN_RETAIN_HELDOUT_SELECTION)
    target_bonus = 5.0 if forget_pct <= TARGET_FORGET_HELDOUT else 0.0
    return True, forget_term + retain_bonus + target_bonus


def save_progress_tables(
    output_dir: Path,
    sweep_history: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = pd.DataFrame(sweep_history)
    if not sweep_df.empty:
        sweep_df = sweep_df.sort_values(["candidate_id", "epoch"])
    sweep_df.to_csv(output_dir / "sweep_history.csv", index=False)

    candidate_df = pd.DataFrame(candidate_summaries)
    if not candidate_df.empty:
        candidate_df = candidate_df.drop_duplicates("candidate_id", keep="last")
        candidate_df = candidate_df.sort_values("selection_score", ascending=False)
    candidate_df.to_csv(output_dir / "candidate_best_summary.csv", index=False)
    return sweep_df, candidate_df


def maybe_commit_and_push(
    enabled: bool,
    paths: Path | list[Path],
    message: str,
    *,
    repo_root: Path,
    branch: str,
) -> None:
    if not enabled:
        return
    sys.path.insert(0, str(repo_root))
    from Tools.github_colab_sync import commit_and_push

    commit_and_push(paths, message, repo_dir=repo_root, branch=branch)


def load_existing_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def candidate_resume_dir(epoch_checkpoint_dir: Path, candidate_id: str) -> Path:
    return epoch_checkpoint_dir / candidate_id


def latest_checkpoint_info(epoch_checkpoint_dir: Path, candidate_id: str) -> dict[str, Any] | None:
    path = candidate_resume_dir(epoch_checkpoint_dir, candidate_id) / "latest.json"
    if not path.exists():
        return None
    return read_json(path)


def save_epoch_checkpoint(
    model,
    tokenizer,
    epoch_checkpoint_dir: Path,
    candidate_id: str,
    epoch: int,
    row: dict[str, Any],
) -> Path:
    candidate_dir = candidate_resume_dir(epoch_checkpoint_dir, candidate_id)
    epoch_dir = candidate_dir / f"epoch_{epoch:02d}"
    adapter_dir = epoch_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    latest = {
        "candidate_id": candidate_id,
        "latest_epoch": epoch,
        "adapter_dir": str(adapter_dir),
        "row": row,
        "updated_at_utc": now_utc(),
        "resume_note": "Optimizer state is intentionally not saved; resume continues from the latest adapter with a fresh optimizer.",
    }
    write_json(candidate_dir / "latest.json", latest)
    return adapter_dir


def train_candidate(
    *,
    config_index: int,
    config: dict[str, Any],
    source_adapter_dir: Path,
    tokenizer,
    teacher_model,
    train_forget: list[dict[str, Any]],
    train_retain: list[dict[str, Any]],
    forget_selection: list[dict[str, Any]],
    retain_selection: list[dict[str, Any]],
    is_seen_prompt,
    selection_ids: set[str],
    run_dir: Path,
    output_dir: Path,
    epoch_checkpoint_dir: Path,
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
    latest = latest_checkpoint_info(epoch_checkpoint_dir, candidate_id) if resume_sweep else None
    start_epoch = 1
    adapter_to_load = source_adapter_dir
    if latest:
        start_epoch = int(latest["latest_epoch"]) + 1
        adapter_to_load = Path(latest["adapter_dir"])
        print(f"Resuming {candidate_id} from epoch {start_epoch} using {adapter_to_load}")

    candidate_rows = [row for row in sweep_history if row.get("candidate_id") == candidate_id]
    if start_epoch > max_epochs and candidate_rows:
        print(f"Candidate already has {max_epochs} epochs: {candidate_id}")
        return sweep_history, candidate_summaries

    model = load_adapter(adapter_to_load, trainable=True)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]))
    device = model_device(model)

    candidate_best = {"score": float("-inf"), "epoch": None, "row": None}
    for row in candidate_rows:
        if float(row.get("selection_score", float("-inf"))) > candidate_best["score"]:
            candidate_best = {
                "score": float(row["selection_score"]),
                "epoch": int(row["epoch"]),
                "row": row.copy(),
            }

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_forget_loss: list[float] = []
        epoch_retain_loss: list[float] = []
        epoch_kl_loss: list[float] = []
        epoch_cosine: list[float] = []
        epoch_conflicts: list[float] = []
        epoch_projections: list[float] = []

        forget_loader = make_loader(
            train_forget,
            tokenizer,
            seed=SEED + config_index * 1000 + epoch,
            shuffle=True,
        )
        retain_loader = make_loader(
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

            forget_batch = move_batch(forget_batch, device)
            retain_batch = move_batch(retain_batch, device)

            forget_outputs = model(**forget_batch)
            forget_loss = forget_outputs.loss.clamp(max=TARGET_FORGET_MAX)
            forget_objective = -float(config["forget_scale"]) * forget_loss
            forget_grads = detached_grads(forget_objective, parameters)

            retain_outputs = model(**retain_batch)
            retain_loss = retain_outputs.loss
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=retain_batch["input_ids"],
                    attention_mask=retain_batch["attention_mask"],
                )
            kl_loss = masked_token_kl(retain_outputs.logits, teacher_outputs.logits, retain_batch["labels"])
            preserve_objective = float(config["retain_weight"]) * retain_loss + float(config["kl_weight"]) * kl_loss
            preserve_grads = detached_grads(preserve_objective, parameters)

            projected_forget, projection_stats = project_forget_gradient(
                forget_grads,
                preserve_grads,
                projection_strength=float(config["projection_strength"]),
            )
            combined_grads = combine_grads(projected_forget, preserve_grads)
            accumulate_grads(
                parameters,
                combined_grads,
                scale=1.0 / GRADIENT_ACCUMULATION_STEPS,
            )

            epoch_forget_loss.append(float(forget_loss.detach().cpu()))
            epoch_retain_loss.append(float(retain_loss.detach().cpu()))
            epoch_kl_loss.append(float(kl_loss.detach().cpu()))
            epoch_cosine.append(projection_stats.cosine)
            epoch_conflicts.append(1.0 if projection_stats.conflict else 0.0)
            epoch_projections.append(1.0 if projection_stats.projection_applied else 0.0)

            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(forget_loader):
                torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        selection_forget_df = evaluate(
            model,
            tokenizer,
            forget_selection,
            "forget_heldout_selection",
            f"{candidate_id}_epoch_{epoch}",
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            progress_every=0,
        )
        selection_retain_df = evaluate(
            model,
            tokenizer,
            retain_selection,
            "retain_heldout_selection",
            f"{candidate_id}_epoch_{epoch}",
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            progress_every=0,
        )
        selection_candidate_dir = selection_result_dir / candidate_id
        selection_candidate_dir.mkdir(parents=True, exist_ok=True)
        selection_forget_df.to_csv(selection_candidate_dir / f"epoch_{epoch:02d}_forget_selection.csv", index=False)
        selection_retain_df.to_csv(selection_candidate_dir / f"epoch_{epoch:02d}_retain_selection.csv", index=False)

        forget_pct = percentage(selection_forget_df)
        retain_pct = percentage(selection_retain_df)
        eligible, score = selection_score(forget_pct, retain_pct)
        row = {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "learning_rate": float(config["learning_rate"]),
            "retain_weight": float(config["retain_weight"]),
            "kl_weight": float(config["kl_weight"]),
            "forget_scale": float(config["forget_scale"]),
            "projection_strength": float(config["projection_strength"]),
            "mean_forget_loss": float(np.mean(epoch_forget_loss)),
            "mean_retain_loss": float(np.mean(epoch_retain_loss)),
            "mean_kl_loss": float(np.mean(epoch_kl_loss)),
            "mean_gradient_cosine": float(np.mean(epoch_cosine)),
            "conflict_rate": float(np.mean(epoch_conflicts)),
            "projection_rate": float(np.mean(epoch_projections)),
            "forget_heldout_selection_percentage": forget_pct,
            "retain_heldout_selection_percentage": retain_pct,
            "retain_eligible": eligible,
            "selection_score": score,
            "updated_at_utc": now_utc(),
        }
        sweep_history = [
            old
            for old in sweep_history
            if not (old.get("candidate_id") == candidate_id and int(old.get("epoch")) == epoch)
        ]
        sweep_history.append(row)
        checkpoint_adapter_dir = save_epoch_checkpoint(
            model,
            tokenizer,
            epoch_checkpoint_dir,
            candidate_id,
            epoch,
            row,
        )
        print(row)

        if score > candidate_best["score"]:
            candidate_best = {"score": score, "epoch": epoch, "row": row.copy()}
            safe_rmtree(best_candidate_dir)
            shutil.copytree(checkpoint_adapter_dir, best_candidate_dir)
            print("Updated candidate-best adapter:", best_candidate_dir)

        candidate_summary = row.copy()
        candidate_summary["selected_epoch_for_candidate"] = int(candidate_best["epoch"] or epoch)
        candidate_summary["candidate_adapter_dir"] = str(best_candidate_dir)
        candidate_summary["selection_score"] = float(candidate_best["score"])
        if candidate_best["row"]:
            candidate_summary.update(candidate_best["row"])
            candidate_summary["selected_epoch_for_candidate"] = int(candidate_best["epoch"])
            candidate_summary["candidate_adapter_dir"] = str(best_candidate_dir)
        candidate_summaries = [summary for summary in candidate_summaries if summary.get("candidate_id") != candidate_id]
        candidate_summaries.append(candidate_summary)
        save_progress_tables(output_dir, sweep_history, candidate_summaries)
        write_json(
            run_dir / "resume_state" / "global_state.json",
            {
                "updated_at_utc": now_utc(),
                "active_candidate_id": candidate_id,
                "latest_epoch": epoch,
                "num_epoch_rows": len(sweep_history),
                "run_dir": str(run_dir),
            },
        )
        maybe_commit_and_push(
            push_each_epoch,
            run_dir,
            f"Colab: save Week 6 constrained gradient checkpoint {candidate_id} epoch {epoch:02d}",
            repo_root=repo_root,
            branch=push_branch,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sweep_history, candidate_summaries


def summarize_outputs(output_dir: Path, frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_results = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    all_results.to_csv(output_dir / "all_before_after_results.csv", index=False)
    summary = (
        all_results.groupby(
            [
                "model_stage",
                "eval_split",
                "prompt_seen_in_original_training",
                "used_for_checkpoint_selection",
            ]
        )
        .agg(
            num_questions=("contains_value", "size"),
            num_correct=("contains_value", "sum"),
            contains_value_percentage=("contains_value", lambda values: 100.0 * values.mean()),
            exact_match_percentage=("exact_match", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "percentage_summary.csv", index=False)

    synthetic = all_results[all_results["eval_split"].isin(["forget", "retain"])]
    category_summary = (
        synthetic.groupby(["model_stage", "eval_split", "category"])
        .agg(
            num_questions=("contains_value", "size"),
            num_correct=("contains_value", "sum"),
            contains_value_percentage=("contains_value", lambda values: 100.0 * values.mean()),
            exact_match_percentage=("exact_match", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )
    category_summary.to_csv(output_dir / "category_summary.csv", index=False)

    identity_summary = (
        synthetic.groupby(["model_stage", "eval_split", "entity_id"])
        .agg(
            num_questions=("contains_value", "size"),
            num_correct=("contains_value", "sum"),
            contains_value_percentage=("contains_value", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )
    identity_summary.to_csv(output_dir / "identity_summary.csv", index=False)
    return summary, category_summary, identity_summary


def read_baseline_metrics(repo_root: Path) -> dict[str, dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    week4_path = repo_root / "Week 4" / "results" / "gradient_ascent_unlearning_v1" / "results" / "metrics.json"
    week5_path = repo_root / "Week 5" / "results" / "retain_regularized_unlearning_resumable_v1" / "results" / "metrics.json"
    aggressive_path = (
        repo_root
        / "Week 5"
        / "results"
        / "aggressive_contrast_evaluation_v1"
        / "c09_most_aggressive_epoch_03"
        / "results"
        / "metrics.json"
    )
    if week5_path.exists():
        week5 = read_json(week5_path)
        baselines["before_unlearning_week35_adapter"] = week5["before_unlearning"]
    if week4_path.exists():
        baselines["week4_gradient_ascent"] = read_json(week4_path)["after_unlearning"]
    if week5_path.exists():
        baselines["week5_preserving_selected"] = read_json(week5_path)["after_unlearning"]
    if aggressive_path.exists():
        baselines["week5_aggressive_c09_epoch_03"] = read_json(aggressive_path)["after_aggressive_contrast"]
    return baselines


def metrics_row(label: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_stage": label,
        "forget_all": values.get("forget_all"),
        "forget_heldout_paraphrases": values.get("forget_heldout_paraphrases"),
        "retain_all": values.get("retain_all"),
        "retain_heldout_paraphrases": values.get("retain_heldout_paraphrases"),
        "general": values.get("general"),
        "forget_all_excluding_selection": values.get("forget_all_excluding_selection"),
        "retain_all_excluding_selection": values.get("retain_all_excluding_selection"),
    }


def write_report(
    path: Path,
    comparison_df: pd.DataFrame,
    candidate_summary_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    display_columns = [
        "forget_all",
        "forget_heldout_paraphrases",
        "retain_all",
        "retain_heldout_paraphrases",
        "general",
    ]
    table = comparison_df.copy()
    for column in display_columns:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.1f}%")

    lines = [
        "# Week 6 Constrained Gradient Unlearning",
        "",
        f"Selected candidate: `{metrics['selected_candidate_id']}`",
        f"Selected epoch: `{metrics['selected_epoch']}`",
        "",
        "This run uses a PCGrad-style constraint: when forget-ascent gradients conflict with retain-preservation gradients, the harmful forget component is projected away before the optimizer step.",
        "",
        "## Comparison",
        "",
        "| model_stage | forget_all | forget_heldout | retain_all | retain_heldout | general |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in table.iterrows():
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
            "## Candidate Ranking",
            "",
            "| candidate_id | epoch | forget_selection | retain_selection | conflict_rate | projection_rate | score |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    ranking = candidate_summary_df.copy()
    if not ranking.empty:
        ranking = ranking.sort_values("selection_score", ascending=False)
        for _, row in ranking.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["candidate_id"]),
                        str(int(row["selected_epoch_for_candidate"])),
                        f"{float(row['forget_heldout_selection_percentage']):.1f}%",
                        f"{float(row['retain_heldout_selection_percentage']):.1f}%",
                        f"{float(row.get('conflict_rate', 0.0)):.2f}",
                        f"{float(row.get('projection_rate', 0.0)):.2f}",
                        f"{float(row['selection_score']):.2f}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            f"The Week 6 target band is forget held-out near `{TARGET_FORGET_HELDOUT:.1f}%` or lower, retain held-out at least `{MIN_RETAIN_HELDOUT_SELECTION:.1f}%`, and general control at least `{GENERAL_FLOOR:.1f}%`.",
            "If the selected Week 6 row improves forget accuracy relative to the Week 5 preserving checkpoint while retaining materially more than the aggressive contrast, the constrained update is doing useful work.",
            "",
            "## Files",
            "",
            "- `after_forget_results.csv`",
            "- `after_retain_results.csv`",
            "- `after_general_results.csv`",
            "- `after_forget_final_excluding_selection_results.csv`",
            "- `after_retain_final_excluding_selection_results.csv`",
            "- `all_before_after_results.csv`",
            "- `percentage_summary.csv`",
            "- `category_summary.csv`",
            "- `identity_summary.csv`",
            "- `sweep_history.csv`",
            "- `candidate_best_summary.csv`",
            "- `week4_week5_week6_comparison.csv`",
            "- `metrics.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_week5_before_files(repo_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    week5_output = repo_root / "Week 5" / "results" / "retain_regularized_unlearning_resumable_v1" / "results"
    frames = []
    for name in ["before_forget_results.csv", "before_retain_results.csv", "before_general_results.csv"]:
        source = week5_output / name
        destination = output_dir / name
        if source.exists():
            shutil.copy2(source, destination)
            frames.append(pd.read_csv(destination))
        else:
            frames.append(pd.DataFrame())
    return frames[0], frames[1], frames[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--run-name", default="constrained_gradient_unlearning_v1")
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--run-full-grid", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Delete the existing Week 6 run folder before running.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--push-each-epoch", action="store_true")
    parser.add_argument("--push-branch", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    run_dir = repo_root / "Week 6" / "results" / args.run_name
    adapter_dir = run_dir / "best_constrained_gradient_adapter"
    candidate_adapter_dir = run_dir / "candidate_adapters"
    output_dir = run_dir / "results"
    resume_dir = run_dir / "resume_state"
    epoch_checkpoint_dir = resume_dir / "epoch_checkpoints"
    selection_result_dir = resume_dir / "selection_results"

    if args.reset:
        safe_rmtree(run_dir)
    for folder in [adapter_dir, candidate_adapter_dir, output_dir, resume_dir, epoch_checkpoint_dir, selection_result_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    data_dir = repo_root / "Week 3.5" / "data" / "synthetic_facts_v1"
    general_dir = repo_root / "Week 3.5" / "data" / "general_controls_v1"
    week35_results = repo_root / "Week 3.5" / "results" / "qwen05_high_accuracy_baseline"
    source_adapter_dir = week35_results / "adapter"
    if not source_adapter_dir.exists():
        fallback = repo_root / "Week 3.5" / "results" / "reference_successful_run" / "adapter"
        if fallback.exists():
            source_adapter_dir = fallback

    train_forget = read_jsonl(data_dir / "train_forget.jsonl")
    train_retain = read_jsonl(data_dir / "train_retain.jsonl")
    eval_forget = read_jsonl(data_dir / "eval_forget.jsonl")
    eval_retain = read_jsonl(data_dir / "eval_retain.jsonl")
    general_controls = read_jsonl(general_dir / "general_control.jsonl")
    is_seen_prompt = make_seen_prompt_checker(train_forget, train_retain)
    forget_selection, retain_selection, selection_ids = make_selection_splits(eval_forget, eval_retain, is_seen_prompt)
    final_forget = [row for row in eval_forget if str(row.get("example_id")) not in selection_ids]
    final_retain = [row for row in eval_retain if str(row.get("example_id")) not in selection_ids]

    configs = full_grid() if args.run_full_grid else FOCUSED_SWEEP
    print("Week 6 run folder:", run_dir)
    print("Source adapter:", source_adapter_dir)
    print("Sweep candidates:", len(configs))
    print("Forget train/eval/selection:", len(train_forget), len(eval_forget), len(forget_selection))
    print("Retain train/eval/selection:", len(train_retain), len(eval_retain), len(retain_selection))
    print("General controls:", len(general_controls))

    tokenizer = load_tokenizer()
    teacher_model = load_adapter(source_adapter_dir, trainable=False)
    teacher_model.eval()

    sweep_history_df = load_existing_table(output_dir / "sweep_history.csv")
    candidate_summary_df = load_existing_table(output_dir / "candidate_best_summary.csv")
    sweep_history = sweep_history_df.to_dict("records") if not sweep_history_df.empty else []
    candidate_summaries = candidate_summary_df.to_dict("records") if not candidate_summary_df.empty else []
    completed_candidate_ids = {row["candidate_id"] for row in candidate_summaries if "candidate_id" in row}

    for config_index, config in enumerate(configs, 1):
        candidate_id = candidate_id_from_config(config_index, config)
        if not args.no_resume and candidate_id in completed_candidate_ids:
            existing_epochs = [
                int(row["epoch"])
                for row in sweep_history
                if row.get("candidate_id") == candidate_id and "epoch" in row
            ]
            if existing_epochs and max(existing_epochs) >= args.max_epochs:
                print("Skipping completed candidate:", candidate_id)
                continue
        sweep_history, candidate_summaries = train_candidate(
            config_index=config_index,
            config=config,
            source_adapter_dir=source_adapter_dir,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            train_forget=train_forget,
            train_retain=train_retain,
            forget_selection=forget_selection,
            retain_selection=retain_selection,
            is_seen_prompt=is_seen_prompt,
            selection_ids=selection_ids,
            run_dir=run_dir,
            output_dir=output_dir,
            epoch_checkpoint_dir=epoch_checkpoint_dir,
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
        completed_candidate_ids.add(candidate_id)

    _, candidate_summary_df = save_progress_tables(output_dir, sweep_history, candidate_summaries)
    if candidate_summary_df.empty:
        raise RuntimeError("No candidate summaries were produced.")

    best_row = candidate_summary_df.sort_values("selection_score", ascending=False).iloc[0].to_dict()
    selected_candidate_dir = Path(best_row["candidate_adapter_dir"])
    safe_rmtree(adapter_dir)
    shutil.copytree(selected_candidate_dir, adapter_dir)
    write_json(
        resume_dir / "selected_global_best.json",
        {
            "selected_candidate_id": best_row["candidate_id"],
            "selected_epoch": int(best_row["selected_epoch_for_candidate"]),
            "selection_score": float(best_row["selection_score"]),
            "candidate_adapter_dir": str(selected_candidate_dir),
            "row": best_row,
            "updated_at_utc": now_utc(),
        },
    )

    del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = load_adapter(adapter_dir, trainable=False)
    stage = "after_week6_constrained_gradient_unlearning"
    after_forget_df = evaluate(
        model,
        tokenizer,
        eval_forget,
        "forget",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
    )
    after_retain_df = evaluate(
        model,
        tokenizer,
        eval_retain,
        "retain",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
    )
    after_general_df = evaluate(
        model,
        tokenizer,
        general_controls,
        "general",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
        general=True,
        progress_every=0,
    )
    after_forget_final_df = evaluate(
        model,
        tokenizer,
        final_forget,
        "forget_final_excluding_selection",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
    )
    after_retain_final_df = evaluate(
        model,
        tokenizer,
        final_retain,
        "retain_final_excluding_selection",
        stage,
        is_seen_prompt=is_seen_prompt,
        selection_ids=selection_ids,
    )

    after_forget_df.to_csv(output_dir / "after_forget_results.csv", index=False)
    after_retain_df.to_csv(output_dir / "after_retain_results.csv", index=False)
    after_general_df.to_csv(output_dir / "after_general_results.csv", index=False)
    after_forget_final_df.to_csv(output_dir / "after_forget_final_excluding_selection_results.csv", index=False)
    after_retain_final_df.to_csv(output_dir / "after_retain_final_excluding_selection_results.csv", index=False)
    before_forget_df, before_retain_df, before_general_df = copy_week5_before_files(repo_root, output_dir)
    summarize_outputs(
        output_dir,
        [before_forget_df, before_retain_df, before_general_df, after_forget_df, after_retain_df, after_general_df],
    )

    after_metrics = {
        "forget_all": percentage(after_forget_df),
        "forget_heldout_paraphrases": prompt_subset_percentage(after_forget_df, False),
        "forget_all_excluding_selection": excluding_selection_percentage(after_forget_df),
        "retain_all": percentage(after_retain_df),
        "retain_heldout_paraphrases": prompt_subset_percentage(after_retain_df, False),
        "retain_all_excluding_selection": excluding_selection_percentage(after_retain_df),
        "general": percentage(after_general_df),
    }
    baseline_metrics = read_baseline_metrics(repo_root)
    comparison_rows = [metrics_row(label, values) for label, values in baseline_metrics.items()]
    comparison_rows.append(metrics_row("week6_constrained_gradient_selected", after_metrics))
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "week4_week5_week6_comparison.csv", index=False)

    metrics = {
        "created_at_utc": now_utc(),
        "run_name": args.run_name,
        "base_model_id": MODEL_ID,
        "source_adapter_dir": str(source_adapter_dir),
        "unlearned_adapter_dir": str(adapter_dir),
        "method": "PCGrad-style constrained forget ascent with retain cross-entropy and retain KL preservation",
        "run_full_grid": bool(args.run_full_grid),
        "num_sweep_candidates": len(configs),
        "resume_enabled": not args.no_resume,
        "resume_dir": str(resume_dir),
        "selected_candidate_id": best_row["candidate_id"],
        "selected_epoch": int(best_row["selected_epoch_for_candidate"]),
        "selected_config": {
            "learning_rate": float(best_row["learning_rate"]),
            "retain_weight": float(best_row["retain_weight"]),
            "kl_weight": float(best_row["kl_weight"]),
            "forget_scale": float(best_row["forget_scale"]),
            "projection_strength": float(best_row["projection_strength"]),
        },
        "selection": {
            "forget_selection_examples": len(forget_selection),
            "retain_selection_examples": len(retain_selection),
            "retain_threshold_percentage": MIN_RETAIN_HELDOUT_SELECTION,
            "target_forget_heldout_percentage": TARGET_FORGET_HELDOUT,
            "selection_score": float(best_row["selection_score"]),
            "selection_row": best_row,
        },
        "training": {
            "max_epochs": args.max_epochs,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "target_forget_max": TARGET_FORGET_MAX,
        },
        "before_unlearning": baseline_metrics.get("before_unlearning_week35_adapter"),
        "after_unlearning": after_metrics,
        "baseline_comparison_sources": {
            "week4": str(repo_root / "Week 4" / "results" / "gradient_ascent_unlearning_v1" / "results" / "metrics.json"),
            "week5_preserving": str(repo_root / "Week 5" / "results" / "retain_regularized_unlearning_resumable_v1" / "results" / "metrics.json"),
            "week5_aggressive": str(
                repo_root
                / "Week 5"
                / "results"
                / "aggressive_contrast_evaluation_v1"
                / "c09_most_aggressive_epoch_03"
                / "results"
                / "metrics.json"
            ),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_report(output_dir / "week6_constrained_gradient_report.md", comparison_df, candidate_summary_df, metrics)
    print("Wrote Week 6 outputs to:", run_dir)
    print(comparison_df)

    maybe_commit_and_push(
        args.push_each_epoch,
        run_dir,
        "Colab: save Week 6 constrained gradient final outputs",
        repo_root=repo_root,
        branch=args.push_branch,
    )


if __name__ == "__main__":
    main()
