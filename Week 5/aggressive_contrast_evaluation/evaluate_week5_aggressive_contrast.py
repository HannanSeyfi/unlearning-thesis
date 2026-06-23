"""Evaluate an aggressive Week 5 checkpoint on the full test sets.

The default run evaluates c09_most_aggressive epoch 03 from the Week 5
resumable sweep and writes results to a new folder so the selected Week 5
outputs remain untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SEED = 42
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 16
SYNTHETIC_SYSTEM_PROMPT = "You answer questions about fictional synthetic people using the provided learned facts."
GENERAL_SYSTEM_PROMPT = "Answer the question concisely. Return only the requested answer without explanation."
FORGET_SELECTION_EXAMPLES = 80
RETAIN_SELECTION_EXAMPLES = 160
RETAIN_THRESHOLD = 85.0


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSONL file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    subset = frame[frame["prompt_seen_in_original_training"] == seen]
    return percentage(subset)


def excluding_selection_percentage(frame: pd.DataFrame) -> float:
    subset = frame[~frame["used_for_checkpoint_selection"]]
    return percentage(subset)


def make_seen_prompt_checker(train_forget: list[dict], train_retain: list[dict]):
    original_train_prompt_keys = {
        (row["entity_id"], row["fact_type"], row["prompt"].strip().lower())
        for row in train_forget + train_retain
    }

    def is_seen_prompt(row: dict) -> bool:
        return (
            row.get("entity_id"),
            row.get("fact_type"),
            row["prompt"].strip().lower(),
        ) in original_train_prompt_keys

    return is_seen_prompt


def make_selection_ids(
    eval_forget: list[dict],
    eval_retain: list[dict],
    is_seen_prompt,
) -> set[str]:
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
    return {
        str(row["example_id"])
        for row in forget_selection + retain_selection
        if row.get("example_id") is not None
    }


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(adapter_dir: Path):
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_dir}")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, *, general: bool = False) -> str:
    messages = [
        {"role": "system", "content": GENERAL_SYSTEM_PROMPT if general else SYNTHETIC_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
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
    records: list[dict],
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
        expected = str(row.get("fact_value", row.get("expected_value", "")))
        answer = generate_answer(model, tokenizer, row["prompt"], general=general)
        example_id = row.get("example_id")
        rows.append(
            {
                "model_stage": stage,
                "eval_split": split,
                "used_for_checkpoint_selection": example_id in selection_ids if example_id is not None else False,
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


def summarize_outputs(
    output_dir: Path,
    after_forget_df: pd.DataFrame,
    after_retain_df: pd.DataFrame,
    after_general_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_results = pd.concat(
        [after_forget_df, after_retain_df, after_general_df],
        ignore_index=True,
    )
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

    synthetic_results = pd.concat([after_forget_df, after_retain_df], ignore_index=True)
    category_summary = (
        synthetic_results.groupby(["model_stage", "eval_split", "category"])
        .agg(
            num_questions=("contains_value", "size"),
            num_correct=("contains_value", "sum"),
            contains_value_percentage=("contains_value", lambda values: 100.0 * values.mean()),
            exact_match_percentage=("exact_match", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )
    identity_summary = (
        synthetic_results.groupby(["model_stage", "eval_split", "entity_id"])
        .agg(
            num_questions=("contains_value", "size"),
            num_correct=("contains_value", "sum"),
            contains_value_percentage=("contains_value", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )

    all_results.to_csv(output_dir / "all_aggressive_contrast_results.csv", index=False)
    summary.to_csv(output_dir / "percentage_summary.csv", index=False)
    category_summary.to_csv(output_dir / "category_summary.csv", index=False)
    identity_summary.to_csv(output_dir / "identity_summary.csv", index=False)
    return summary, category_summary, identity_summary


def write_report(
    path: Path,
    comparison_df: pd.DataFrame,
    metrics: dict,
    candidate_id: str,
    epoch: int,
) -> None:
    table = comparison_df.copy()
    metric_columns = [
        "forget_all",
        "forget_heldout_paraphrases",
        "retain_all",
        "retain_heldout_paraphrases",
        "general",
    ]
    for column in metric_columns:
        table[column] = table[column].map(lambda value: f"{float(value):.1f}%")

    lines = [
        "# Week 5 Aggressive Contrast Evaluation",
        "",
        f"Candidate: `{candidate_id}`",
        f"Epoch: `{epoch}`",
        "",
        "This run evaluates an aggressive Week 5 checkpoint on the full forget, retain, and general sets.",
        "It is meant to contrast with the selected Week 5 preserving checkpoint.",
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
            "## Interpretation",
            "",
            "Lower forget accuracy means stronger forgetting. Higher retain/general accuracy means better preservation.",
            "If this aggressive checkpoint approaches Week 4 forgetting but damages retain accuracy, it supports the Week 6 target:",
            "recover stronger forgetting while avoiding global retain collapse.",
            "",
            "## Files",
            "",
            "- `after_forget_results.csv`",
            "- `after_retain_results.csv`",
            "- `after_general_results.csv`",
            "- `all_aggressive_contrast_results.csv`",
            "- `percentage_summary.csv`",
            "- `category_summary.csv`",
            "- `identity_summary.csv`",
            "- `week4_week5_aggressive_comparison.csv`",
            "- `metrics.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--candidate-id", default="c09_most_aggressive")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    candidate_id = args.candidate_id
    epoch = int(args.epoch)
    run_name = args.run_name or f"{candidate_id}_epoch_{epoch:02d}"

    source_run_dir = repo_root / "Week 5" / "results" / "retain_regularized_unlearning_resumable_v1"
    adapter_dir = (
        source_run_dir
        / "resume_state"
        / "epoch_checkpoints"
        / candidate_id
        / f"epoch_{epoch:02d}"
        / "adapter"
    )
    output_run_dir = repo_root / "Week 5" / "results" / "aggressive_contrast_evaluation_v1" / run_name
    output_dir = output_run_dir / "results"

    if output_run_dir.exists() and any(output_run_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Output run folder already exists: {output_run_dir}\n"
            "Use --overwrite or choose a different --run-name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = repo_root / "Week 3.5" / "data" / "synthetic_facts_v1"
    general_dir = repo_root / "Week 3.5" / "data" / "general_controls_v1"
    train_forget = read_jsonl(data_dir / "train_forget.jsonl")
    train_retain = read_jsonl(data_dir / "train_retain.jsonl")
    eval_forget = read_jsonl(data_dir / "eval_forget.jsonl")
    eval_retain = read_jsonl(data_dir / "eval_retain.jsonl")
    general_controls = read_jsonl(general_dir / "general_control.jsonl")

    is_seen_prompt = make_seen_prompt_checker(train_forget, train_retain)
    selection_ids = make_selection_ids(eval_forget, eval_retain, is_seen_prompt)

    week4_metrics = read_json(
        repo_root / "Week 4" / "results" / "gradient_ascent_unlearning_v1" / "results" / "metrics.json"
    )
    week5_metrics = read_json(source_run_dir / "results" / "metrics.json")
    sweep_history_path = source_run_dir / "results" / "sweep_history.csv"
    sweep_history = pd.read_csv(sweep_history_path)
    selection_row_df = sweep_history[
        (sweep_history["candidate_id"] == candidate_id)
        & (sweep_history["epoch"].astype(int) == epoch)
    ]
    selection_row = selection_row_df.iloc[0].to_dict() if not selection_row_df.empty else None

    print("Evaluating adapter:", adapter_dir)
    tokenizer = load_tokenizer()
    model = load_model(adapter_dir)

    stage = "after_week5_aggressive_contrast"
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

    after_forget_df.to_csv(output_dir / "after_forget_results.csv", index=False)
    after_retain_df.to_csv(output_dir / "after_retain_results.csv", index=False)
    after_general_df.to_csv(output_dir / "after_general_results.csv", index=False)
    after_forget_df[~after_forget_df["used_for_checkpoint_selection"]].to_csv(
        output_dir / "after_forget_final_excluding_selection_results.csv",
        index=False,
    )
    after_retain_df[~after_retain_df["used_for_checkpoint_selection"]].to_csv(
        output_dir / "after_retain_final_excluding_selection_results.csv",
        index=False,
    )

    summarize_outputs(output_dir, after_forget_df, after_retain_df, after_general_df)

    aggressive_after = {
        "forget_all": percentage(after_forget_df),
        "forget_heldout_paraphrases": prompt_subset_percentage(after_forget_df, False),
        "forget_all_excluding_selection": excluding_selection_percentage(after_forget_df),
        "retain_all": percentage(after_retain_df),
        "retain_heldout_paraphrases": prompt_subset_percentage(after_retain_df, False),
        "retain_all_excluding_selection": excluding_selection_percentage(after_retain_df),
        "general": percentage(after_general_df),
    }

    comparison_rows = [
        {"model_stage": "before_unlearning_week35_adapter", **week5_metrics["before_unlearning"]},
        {"model_stage": "week4_gradient_ascent", **week4_metrics["after_unlearning"]},
        {"model_stage": "week5_preserving_selected", **week5_metrics["after_unlearning"]},
        {"model_stage": f"week5_aggressive_{candidate_id}_epoch_{epoch:02d}", **aggressive_after},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "week4_week5_aggressive_comparison.csv", index=False)

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": "week5_aggressive_contrast_evaluation_v1",
        "candidate_id": candidate_id,
        "epoch": epoch,
        "run_folder": str(output_run_dir),
        "base_model_id": MODEL_ID,
        "adapter_dir": str(adapter_dir),
        "source_week5_run_dir": str(source_run_dir),
        "selection_row_from_week5_sweep": selection_row,
        "strict_scoring": week5_metrics.get("source_week35_strict_scoring"),
        "selection_ids": {
            "num_selection_ids": len(selection_ids),
            "forget_selection_examples": FORGET_SELECTION_EXAMPLES,
            "retain_selection_examples": RETAIN_SELECTION_EXAMPLES,
            "retain_threshold_percentage": RETAIN_THRESHOLD,
        },
        "after_aggressive_contrast": aggressive_after,
        "comparison_sources": {
            "week4_metrics": str(repo_root / "Week 4" / "results" / "gradient_ascent_unlearning_v1" / "results" / "metrics.json"),
            "week5_preserving_metrics": str(source_run_dir / "results" / "metrics.json"),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_report(
        output_dir / "aggressive_contrast_report.md",
        comparison_df,
        metrics,
        candidate_id,
        epoch,
    )

    print("Wrote aggressive contrast outputs to:", output_run_dir)
    print(comparison_df)


if __name__ == "__main__":
    main()

