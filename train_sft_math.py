#!/usr/bin/env python3
"""
Complete SFT Training Script for MATH Dataset

This script implements supervised fine-tuning on the MATH dataset with:
- Support for Hugging Face models and datasets
- Full SwanLab integration with rich metrics
- Gradient accumulation and mixed precision training
- Evaluation with verified math grading
- Checkpointing and resumption capabilities
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)
import swanlab
from datasets import load_dataset

from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.log_generation import log_generations
from cs336_alignment.drgrpo_grader import question_only_reward_fn, extract_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preprocess_math_example(example: Dict) -> Dict[str, str]:
    """
    Preprocess a single math example for training.

    Args:
        example: Raw example from dataset with fields like 'problem'/'question' and 'solution'/'answer'

    Returns:
        dict with 'prompt', 'output', and 'ground_truth' keys
    """
    # Extract problem and solution - handle different field names
    problem = example.get("problem", example.get("question", ""))
    solution = example.get("solution", example.get("answer", ""))

    # Format prompt
    prompt = f"Solve this math problem:\n\n{problem}\n\nAnswer:"

    # Extract ground truth answer from solution
    ground_truth = extract_answer(solution) if "\\boxed" in solution else solution

    return {
        "prompt": prompt,
        "output": solution,
        "ground_truth": ground_truth or solution,
    }


def collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizerBase) -> Dict:
    """
    Collate batch of examples into tensors.

    Args:
        batch: List of dicts with 'prompt' and 'output' keys
        tokenizer: Tokenizer for encoding

    Returns:
        dict with tokenized inputs and metadata
    """
    prompts = [item["prompt"] for item in batch]
    outputs = [item["output"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]

    # Tokenize using helper function
    tokenized = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=outputs,
        tokenizer=tokenizer,
    )

    return {
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["labels"],
        "response_mask": tokenized["response_mask"],
        "prompts": prompts,
        "outputs": outputs,
        "ground_truths": ground_truths,
    }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SFT model on MATH dataset"
    )

    # Model & Data
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or Hugging Face model identifier (e.g., 'Qwen/Qwen2.5-Math-1.5B')"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to model_path)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to MATH dataset directory or Hugging Face dataset identifier (e.g., 'lighteval/MATH')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Where to save checkpoints"
    )

    # Training Hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps for scheduler"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm"
    )

    # Evaluation
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=500,
        help="Max samples to evaluate"
    )
    parser.add_argument(
        "--generation_max_length",
        type=int,
        default=512,
        help="Max tokens for generation"
    )
    parser.add_argument(
        "--generation_temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0.0 = greedy)"
    )

    # SwanLab
    parser.add_argument(
        "--swanlab_project",
        type=str,
        default="cs336-sft-math",
        help="SwanLab project name"
    )
    parser.add_argument(
        "--swanlab_run_name",
        type=str,
        default=None,
        help="SwanLab experiment/run name"
    )

    # Checkpointing
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=3,
        help="Number of checkpoints to keep"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with reduced data"
    )

    args = parser.parse_args()

    # Set tokenizer path if not provided
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    return args


def save_checkpoint(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
    output_dir: str,
    args: argparse.Namespace,
    keep_last_n: int = 3,
):
    """Save training checkpoint."""
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_pretrained(checkpoint_dir)

    # Save training state
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": vars(args),
    }, checkpoint_dir / "training_state.pt")

    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Clean up old checkpoints
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1])
    )
    if len(checkpoints) > keep_last_n:
        for ckpt in checkpoints[:-keep_last_n]:
            logger.info(f"Removing old checkpoint {ckpt}")
            import shutil
            shutil.rmtree(ckpt)


def load_checkpoint(
    checkpoint_path: str,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int]:
    """Load checkpoint and return step and epoch."""
    checkpoint_dir = Path(checkpoint_path)

    # Load model weights
    model_path = checkpoint_dir / "pytorch_model.bin"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
    else:
        # Try loading from safetensors
        from safetensors.torch import load_file
        model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"))

    # Load training state
    training_state = torch.load(checkpoint_dir / "training_state.pt")
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    scheduler.load_state_dict(training_state["scheduler_state_dict"])

    step = training_state["step"]
    epoch = training_state["epoch"]

    logger.info(f"Loaded checkpoint from {checkpoint_dir} (step={step}, epoch={epoch})")
    return step, epoch


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_dataloader: DataLoader,
    args: argparse.Namespace,
    step: int,
) -> Dict:
    """
    Evaluate model on validation set.

    Returns:
        dict with evaluation metrics
    """
    model.eval()

    all_prompts = []
    all_responses = []
    all_ground_truths = []
    all_rewards = []
    all_token_entropies = []
    all_response_masks = []

    logger.info(f"Running evaluation at step {step}...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= args.max_eval_samples // args.eval_batch_size:
                break

            prompts = batch["prompts"]
            ground_truths = batch["ground_truths"]

            # Tokenize prompts only for generation
            prompt_encodings = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_length,
            )
            input_ids = prompt_encodings["input_ids"].to(model.device)
            attention_mask = prompt_encodings["attention_mask"].to(model.device)

            # Generate responses
            if args.generation_temperature == 0.0:
                # Greedy decoding
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.generation_max_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                # Sampling
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.generation_max_length,
                    do_sample=True,
                    temperature=args.generation_temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode responses (remove prompt)
            responses = []
            for i, gen_ids in enumerate(generated_ids):
                prompt_len = input_ids[i].shape[0]
                response_ids = gen_ids[prompt_len:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response)

            # Compute rewards
            rewards = []
            for response, ground_truth in zip(responses, ground_truths):
                reward = question_only_reward_fn(response, ground_truth, fast=True)
                rewards.append(reward)

            # Get token entropies for generated responses
            # Re-tokenize full prompt+response to get log probs
            full_texts = [p + r for p, r in zip(prompts, responses)]
            tokenized = tokenize_prompt_and_output(
                prompt_strs=prompts,
                output_strs=responses,
                tokenizer=tokenizer,
            )

            input_ids_full = tokenized["input_ids"].to(model.device)
            labels_full = tokenized["labels"].to(model.device)
            response_mask = tokenized["response_mask"].to(model.device)

            # Get log probs and entropy
            log_prob_output = get_response_log_probs(
                model=model,
                input_ids=input_ids_full,
                labels=labels_full,
                return_token_entropy=True,
            )

            token_entropies = log_prob_output["token_entropy"]

            # Collect results
            all_prompts.extend(prompts)
            all_responses.extend(responses)
            all_ground_truths.extend(ground_truths)
            all_rewards.extend(rewards)
            all_token_entropies.extend([te.cpu() for te in token_entropies])
            all_response_masks.extend([rm.cpu() for rm in response_mask])

    # Aggregate metrics using log_generations
    metrics = log_generations(
        prompts=all_prompts,
        responses=all_responses,
        ground_truths=all_ground_truths,
        rewards=all_rewards,
        token_entropies=all_token_entropies,
        response_masks=all_response_masks,
    )

    model.train()
    return metrics


def train(args: argparse.Namespace):
    """Main training function."""

    # Set seed
    set_seed(args.seed)

    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    device = torch.device("cuda")
    logger.info(f"Using device: {device}")

    # Initialize swanlab
    swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_run_name,
        config=vars(args),
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)
    model.train()

    # Load datasets from Hugging Face
    logger.info(f"Loading datasets from Hugging Face: {args.data_path}")

    # Try different loading strategies for different dataset formats
    train_dataset = None
    eval_dataset = None

    # Strategy 1: Try loading without config (e.g., lighteval/MATH)
    try:
        train_dataset = load_dataset(args.data_path, split="train")
        logger.info(f"Loaded {len(train_dataset)} training samples")
    except Exception as e1:
        logger.info(f"Failed to load with split='train': {e1}")

        # Strategy 2: Try with 'main' config (e.g., openai/gsm8k)
        try:
            train_dataset = load_dataset(args.data_path, "main", split="train")
            logger.info(f"Loaded {len(train_dataset)} training samples using 'main' config")
        except Exception as e2:
            logger.error(f"Failed to load with 'main' config: {e2}")
            raise RuntimeError(
                f"Could not load training data from {args.data_path}. "
                f"Tried: (1) load_dataset('{args.data_path}', split='train'), "
                f"(2) load_dataset('{args.data_path}', 'main', split='train'). "
                f"Please check the dataset path and format."
            )

    # Try to load evaluation split
    try:
        eval_dataset = load_dataset(args.data_path, split="test")
        logger.info(f"Loaded {len(eval_dataset)} evaluation samples")
    except Exception as e1:
        # Try with 'main' config
        try:
            eval_dataset = load_dataset(args.data_path, "main", split="test")
            logger.info(f"Loaded {len(eval_dataset)} evaluation samples using 'main' config")
        except Exception as e2:
            # For datasets without test split, create validation set from train
            logger.warning(f"'test' split not found. Creating validation set from training data.")
            train_test_split = train_dataset.train_test_split(
                test_size=min(500, len(train_dataset) // 10),
                seed=args.seed
            )
            train_dataset = train_test_split["train"]
            eval_dataset = train_test_split["test"]
            logger.info(f"Split into {len(train_dataset)} train and {len(eval_dataset)} eval samples")

    # Apply preprocessing to map examples to the format we need
    train_dataset = train_dataset.map(preprocess_math_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_math_example, remove_columns=eval_dataset.column_names)

    # Limit samples for debugging
    if args.debug:
        train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(50, len(eval_dataset))))
        logger.info(f"Debug mode: using {len(train_dataset)} train and {len(eval_dataset)} eval samples")

    # Limit eval samples if specified
    if len(eval_dataset) > args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        logger.info(f"Limited evaluation to {args.max_eval_samples} samples")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Calculate total steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs

    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Load checkpoint if resuming
    start_step = 0
    start_epoch = 0
    if args.resume_from_checkpoint:
        start_step, start_epoch = load_checkpoint(
            args.resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
        )

    # Training loop
    global_step = start_step

    logger.info("Starting training...")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {total_steps}")

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # Get log probabilities
            log_prob_output = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )

            policy_log_probs = log_prob_output["log_probs"]

            # Training step (handles backward pass)
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=1.0,
            )

            # Update weights every gradient_accumulation_steps
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log training metrics
                swanlab.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "train/epoch": epoch,
                    "train/step": global_step,
                }, step=global_step)

                if global_step % 10 == 0:
                    logger.info(
                        f"Step {global_step}: loss={loss.item():.4f}, "
                        f"lr={scheduler.get_last_lr()[0]:.2e}, "
                        f"grad_norm={grad_norm.item():.4f}"
                    )

                # Evaluate
                if global_step % args.eval_steps == 0:
                    eval_metrics = evaluate(
                        model=model,
                        tokenizer=tokenizer,
                        eval_dataloader=eval_dataloader,
                        args=args,
                        step=global_step,
                    )

                    # Log evaluation metrics
                    swanlab.log({
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval/avg_format_reward": eval_metrics["avg_format_reward"],
                        "eval/avg_answer_reward": eval_metrics["avg_answer_reward"],
                        "eval/avg_total_reward": eval_metrics["avg_total_reward"],
                        "eval/avg_token_entropy": eval_metrics.get("avg_token_entropy", 0.0),
                        "eval/avg_response_length": eval_metrics.get("avg_response_length", 0.0),
                        "eval/avg_correct_response_length": eval_metrics.get("avg_correct_response_length", 0.0),
                        "eval/avg_incorrect_response_length": eval_metrics.get("avg_incorrect_response_length", 0.0),
                        "eval/num_correct": eval_metrics.get("num_correct", 0),
                        "eval/num_incorrect": eval_metrics.get("num_incorrect", 0),
                    }, step=global_step)

                    # Log sample generations as table
                    examples = eval_metrics["examples"][:10]  # First 10 examples
                    table_data = [
                        [
                            ex["prompt"][:100] + "...",  # Truncate for display
                            ex["response"][:200] + "...",
                            ex["ground_truth"][:100] + "...",
                            ex["total_reward"],
                        ]
                        for ex in examples
                    ]
                    swanlab.log({
                        "eval/sample_generations": swanlab.Table(
                            columns=["Prompt", "Response", "Ground Truth", "Reward"],
                            data=table_data,
                        )
                    }, step=global_step)

                    logger.info(
                        f"Eval at step {global_step}: "
                        f"accuracy={eval_metrics['accuracy']:.4f}, "
                        f"format_reward={eval_metrics['avg_format_reward']:.4f}, "
                        f"answer_reward={eval_metrics['avg_answer_reward']:.4f}"
                    )

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        output_dir=args.output_dir,
                        args=args,
                        keep_last_n=args.keep_last_n_checkpoints,
                    )

        # End of epoch - save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            epoch=epoch + 1,
            output_dir=args.output_dir,
            args=args,
            keep_last_n=args.keep_last_n_checkpoints,
        )

    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
        args=args,
        step=global_step,
    )

    swanlab.log({
        "final/accuracy": final_metrics["accuracy"],
        "final/avg_format_reward": final_metrics["avg_format_reward"],
        "final/avg_answer_reward": final_metrics["avg_answer_reward"],
    }, step=global_step)

    logger.info("Training complete!")
    logger.info(f"Final accuracy: {final_metrics['accuracy']:.4f}")

    swanlab.finish()


def main():
    args = parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        train(args)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
