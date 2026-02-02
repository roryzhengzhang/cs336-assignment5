# SFT Training on MATH Dataset - Usage Guide

This guide explains how to use the complete SFT training script for the MATH dataset with support for both local models and Hugging Face Hub.

## Quick Start

### Using Hugging Face Models and Datasets (Recommended)

```bash
python train_sft_math.py \
  --model_path Qwen/Qwen2.5-Math-1.5B \
  --data_path data/gsm8k \
  --output_dir ./outputs/sft-math-run1 \
  --swanlab_project cs336-sft-math \
  --swanlab_run_name qwen-math-1.5b-sft
```

### Using Local Paths

```bash
python train_sft_math.py \
  --model_path /data/models/Qwen2.5-Math-1.5B \
  --data_path /data/gsm8k \
  --output_dir ./outputs/sft-math-run1 \
  --swanlab_project cs336-sft-math
```

### Debug Mode (Fast Testing)

```bash
python train_sft_math.py \
  --model_path Qwen/Qwen2.5-Math-1.5B \
  --data_path lighteval/MATH \
  --debug \
  --num_epochs 1 \
  --eval_steps 10
```

## Full Configuration Example

```bash
python train_sft_math.py \
  --model_path Qwen/Qwen2.5-Math-1.5B \
  --data_path lighteval/MATH \
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
  --swanlab_project cs336-sft-math \
  --swanlab_run_name qwen-math-baseline \
  --save_steps 500 \
  --keep_last_n_checkpoints 3 \
  --seed 42
```

## Hugging Face Integration

The script now supports both local files and Hugging Face Hub:

### Model Loading
- **Hugging Face**: `--model_path Qwen/Qwen2.5-Math-1.5B`
- **Local**: `--model_path /path/to/model`

Models from Hugging Face are automatically downloaded and cached.

### Dataset Loading
- **Hugging Face**: `--data_path lighteval/MATH` or `--data_path hendrycks/math`
- **Local JSON/JSONL**: `--data_path /path/to/dataset`
- **Local MATH structure**: `--data_path /path/to/MATH` (with train/test subdirs)

The script automatically detects whether you're using a local path or HF identifier.

## Resume from Checkpoint

```bash
python train_sft_math.py \
  --model_path Qwen/Qwen2.5-Math-1.5B \
  --data_path lighteval/MATH \
  --resume_from_checkpoint ./outputs/sft-math-run1/checkpoint-1000
```

## Key Features

### Data Loading
- **Hugging Face Hub**: Automatically downloads and loads datasets (e.g., `lighteval/MATH`)
- **Local Files**: Supports JSON/JSONL formats
- **Directory Structure**: Handles standard MATH subdirectory organization (train/test splits)
- **Auto-detection**: Determines whether path is local or HF identifier
- **Prompt Formatting**: "Solve this math problem:\n\n{problem}\n\nAnswer:"
- **Response Masks**: Uses `tokenize_prompt_and_output()` for proper masking

### Training
- **Mixed Precision**: Uses bfloat16 with FlashAttention2 for efficiency
- **Gradient Accumulation**: Configurable to simulate larger batch sizes
- **Optimizer**: AdamW with cosine learning rate schedule with warmup
- **Gradient Clipping**: Prevents training instabilities

### Evaluation
- **Generation**: Greedy decoding (temp=0.0) or sampling (temp>0.0)
- **Grading**: Uses verified math grader (`question_only_reward_fn`) with support for LaTeX boxed answers
- **Metrics**: Accuracy, format reward, answer reward, token entropy, response length

### SwanLab Logging
- **Training Metrics**: loss, learning rate, gradient norm
- **Evaluation Metrics**: accuracy, rewards, entropy, response lengths
- **Sample Generations**: Interactive table with prompt/response/ground truth
- **Real-time Updates**: All metrics logged to SwanLab dashboard

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

## Monitoring with SwanLab

The script logs comprehensive metrics to SwanLab:

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
- **Hugging Face**: Verify dataset identifier is correct (e.g., `lighteval/MATH`)
- **Local**: Ensure `--data_path` points to correct directory
- Check dataset has train/test splits or JSON/JSONL files
- Look at script logs for attempted file paths and error messages
- Try using a Hugging Face dataset if local loading fails

### Model Not Loading
- **Hugging Face**: Verify model identifier is correct (e.g., `Qwen/Qwen2.5-Math-1.5B`)
- **Local**: Ensure path contains `config.json` and model weights
- Check internet connection if downloading from HF Hub
- Verify Hugging Face cache has sufficient disk space (`~/.cache/huggingface`)
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

1. **Check SwanLab Dashboard**: Review training curves and sample generations
2. **Load Best Checkpoint**: Select checkpoint with highest eval accuracy
3. **Run Additional Evaluation**: Test on held-out problems
4. **Experiment Tracking**: Compare runs with different hyperparameters

## Questions?

- Check script help: `python train_sft_math.py --help`
- Review implementation in `train_sft_math.py`
- Check helper functions in `cs336_alignment/` directory
