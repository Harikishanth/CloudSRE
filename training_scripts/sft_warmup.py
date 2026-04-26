"""
CloudSRE v2 — SFT Warmup Script (run BEFORE GRPO training)

Downloads expert SRE demonstrations and fine-tunes the model on them.
This teaches the model the FORMAT of correct SRE actions before GRPO
teaches it WHEN to use them.

Usage (on Colab):
    !pip install unsloth datasets
    !wget -O sft_warmup.py https://raw.githubusercontent.com/Harikishanth/CloudSRE/main/sft_warmup.py
    !wget -O sft_training_data.jsonl https://raw.githubusercontent.com/Harikishanth/CloudSRE/main/sft_training_data.jsonl
    !python sft_warmup.py --model-id unsloth/Qwen2.5-1.5B-Instruct --epochs 3
"""

import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="CloudSRE SFT Warmup")
    parser.add_argument("--model-id", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data-file", default="sft_training_data.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", default="./cloudsre-sft-checkpoint")
    parser.add_argument("--hf-repo", default="", help="Push to HF Hub repo")
    args = parser.parse_args()

    print("=" * 60)
    print("CloudSRE v2 — SFT Warmup")
    print("=" * 60)

    # ── Load SFT data ──
    print(f"\nLoading SFT data from {args.data_file}...")
    examples = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            examples.append(ex)

    print(f"  Loaded {len(examples)} expert demonstrations")
    print(f"  Tiers: {set(ex['tier'] for ex in examples)}")

    # ── Format for SFT ──
    # Convert to chat format: list of {"messages": [...]}
    sft_data = []
    for ex in examples:
        sft_data.append({"messages": ex["messages"]})

    # Duplicate small dataset to get more training signal
    if len(sft_data) < 50:
        multiplier = max(1, 50 // len(sft_data))
        sft_data = sft_data * multiplier
        print(f"  Duplicated to {len(sft_data)} examples (x{multiplier})")

    # ── Load Model with Unsloth ──
    print(f"\nLoading model: {args.model_id}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ── Prepare Dataset ──
    from datasets import Dataset

    def format_chat(example):
        """Format messages into the model's chat template."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(sft_data)
    dataset = dataset.map(format_chat)
    print(f"\n  Dataset size: {len(dataset)} examples")
    print(f"  Sample length: {len(dataset[0]['text'])} chars")

    # ── Train with SFTTrainer ──
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"\nStarting SFT training ({args.epochs} epochs)...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=5,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=2048,
    )

    trainer.train()

    # ── Save ──
    print(f"\nSaving SFT checkpoint to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── Auto-generate loss curve plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        log_history = trainer.state.log_history
        steps = [entry['step'] for entry in log_history if 'loss' in entry]
        losses = [entry['loss'] for entry in log_history if 'loss' in entry]

        if steps and losses:
            # Save training log as JSON for Colab
            training_log = [{"step": s, "loss": l} for s, l in zip(steps, losses)]
            with open(f"{args.output_dir}/training_log.json", "w") as f:
                json.dump(training_log, f, indent=2)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, losses, color='#e74c3c', linewidth=2)
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'SFT Loss Curve ({len(sft_data)} examples, {args.epochs} epochs)', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('sft_loss_curve.png', dpi=150)
            print("Saved: sft_loss_curve.png")
        else:
            print("No loss entries found in trainer log history")
    except ImportError:
        print("matplotlib not available — skipping plot generation")

    if args.hf_repo:
        print(f"Pushing to HF Hub: {args.hf_repo}")
        model.push_to_hub(args.hf_repo, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(args.hf_repo, token=os.environ.get("HF_TOKEN"))

    print(f"\n{'='*60}")
    print(f"SFT WARMUP COMPLETE")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.output_dir}")
    print(f"  Next step: Run train_colab.py with --model-id {args.output_dir}")
    print(f"  This gives GRPO a model that already knows SRE command formats!")

if __name__ == "__main__":
    main()

