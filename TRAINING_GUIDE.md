# SFT Training on MATH Dataset - Usage Guide

This guide explains how to use the complete SFT training script for the MATH dataset using Qwen2.5-Math-1.5B.

## Quick Start

### Basic Training Run

```bash
python train_sft_math.py \
  --model_path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --data_path /data/a5-alignment/datasets/MATH \
  --output_dir ./outputs/sft-math-run1 \
  --wandb_project "cs336-sft-math" \
  --wandb_run_name "qwen-math-1.5b-sft"
```

### Debug Mode (Fast Testing)

```bash
python train_sft_math.py \
  --model_path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --data_path /data/a5-alignment/datasets/MATH \
  --debug \
  --num_epochs 1 \
  --eval_steps 10
```

## Full Configuration Example

```bash
python train_sft_math.py \
  --model_path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --data_path /data/a5-alignment/datasets/MATH \
  --output_dir ./outputs/sft-math-experiment \
  --learning_rate 1e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_epochs 3 \
  --max_seq_length 1024 \
  --warmup_steps 100 \
  --max_grad_norm 1.0 \
  --eval_steps 500 \
  --eval_batch_size 8 \
  --max_eval_samples 500 \
  --generation_max_length 512 \
  --generation_temperature 0.0 \
  --wandb_project "cs336-sft-math" \
  --wandb_run_name "qwen-math-baseline" \
  --save_steps 500 \
  --keep_last_n_checkpoints 3 \
  --seed 42
```

## Resume from Checkpoint

```bash
python train_sft_math.py \
  --model_path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --data_path /data/a5-alignment/datasets/MATH \
  --resume_from_checkpoint ./outputs/sft-math-run1/checkpoint-1000
```

## Key Features

### Data Loading
- Automatically detects MATH dataset format (JSON/JSONL)
- Supports standard MATH directory structure with train/test splits
- Formats prompts as: "Solve this math problem:\n\n{problem}\n\nAnswer:"
- Uses `tokenize_prompt_and_output()` to create proper response masks

### Training
- **Mixed Precision**: Uses bfloat16 with FlashAttention2 for efficiency
- **Gradient Accumulation**: Configurable to simulate larger batch sizes
- **Optimizer**: AdamW with cosine learning rate schedule with warmup
- **Gradient Clipping**: Prevents training instabilities

### Evaluation
- **Generation**: Greedy decoding (temp=0.0) or sampling (temp>0.0)
- **Grading**: Uses verified math grader (`question_only_reward_fn`) with support for LaTeX boxed answers
- **Metrics**: Accuracy, format reward, answer reward, token entropy, response length

### Wandb Logging
- **Training Metrics**: loss, learning rate, gradient norm
- **Evaluation Metrics**: accuracy, rewards, entropy, response lengths
- **Sample Generations**: Interactive table with prompt/response/ground truth
- **Real-time Updates**: All metrics logged to wandb dashboard

### Checkpointing
- **Auto-save**: Every N steps and end of each epoch
- **Resume**: Full training state restoration
- **Cleanup**: Automatically keeps only last N checkpoints

## Expected Output Structure

```
outputs/
└── sft-math-run1/
    ├── checkpoint-500/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── training_state.pt
    ├── checkpoint-1000/
    │   └── ...
    └── checkpoint-1500/
        └── ...
```

## Monitoring with Wandb

The script logs comprehensive metrics to Wandb:

### Training Phase
- `train/loss`: Cross-entropy loss on response tokens
- `train/learning_rate`: Current learning rate
- `train/grad_norm`: Gradient norm after clipping
- `train/step`: Global optimization step

### Evaluation Phase
- `eval/accuracy`: Proportion of correct answers
- `eval/avg_format_reward`: Percentage of properly formatted responses
- `eval/avg_answer_reward`: Percentage of correct answers
- `eval/avg_token_entropy`: Diversity of token predictions
- `eval/avg_response_length`: Average response length in tokens
- `eval/sample_generations`: Table with example outputs

## Hyperparameter Recommendations

### Default Settings (Recommended)
- Learning rate: `1e-5` (good for fine-tuning)
- Batch size: `4` (per GPU)
- Gradient accumulation: `8` (effective batch size = 32)
- Epochs: `3`
- Max sequence length: `1024`
- Warmup steps: `100`

### For Faster Experiments
- Increase batch size if memory allows
- Reduce `max_eval_samples` to 100-200
- Use `--debug` flag for quick iterations

### For Higher Quality
- Increase `num_epochs` to 5
- Lower learning rate to `5e-6`
- Increase `gradient_accumulation_steps` for larger effective batch size

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--max_seq_length` (try 512)
- Increase `--gradient_accumulation_steps` to maintain effective batch size

### Slow Training
- Ensure FlashAttention2 is properly installed
- Check GPU utilization with `nvidia-smi`
- Increase `--batch_size` if memory allows

### Dataset Not Found
- Verify `--data_path` points to correct directory
- Check dataset has train/test splits
- Look at script logs for attempted file paths

### Model Not Loading
- Ensure model path contains `config.json` and model weights
- Check tokenizer is compatible
- Verify CUDA and PyTorch are properly installed

## Implementation Details

### Helper Functions Used
- `tokenize_prompt_and_output()`: Creates input_ids, labels, and response masks
- `get_response_log_probs()`: Computes log probabilities and entropy
- `sft_microbatch_train_step()`: Handles loss computation and backward pass
- `log_generations()`: Aggregates evaluation metrics
- `question_only_reward_fn()`: Grades math answers with LaTeX support

### Model Configuration
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

This uses the exact configuration you specified with Qwen2.5-Math-1.5B.

## Next Steps

After training completes:

1. **Check Wandb Dashboard**: Review training curves and sample generations
2. **Load Best Checkpoint**: Select checkpoint with highest eval accuracy
3. **Run Additional Evaluation**: Test on held-out problems
4. **Experiment Tracking**: Compare runs with different hyperparameters

## Questions?

- Check script help: `python train_sft_math.py --help`
- Review implementation in `train_sft_math.py`
- Check helper functions in `cs336_alignment/` directory
