"""LoRA fine-tune google/madlad400-3b-mt for English->Dhivehi.

bf16 base + LoRA r=32 on attention + FFN projections of encoder, decoder,
and cross-attention. Trains on data/processed/round1 (see
data/build_dataset.py); checkpoint selection by chrF++ on a dev subset.

Usage:
  python train/finetune.py [--smoke] [--resume]

Output: adapter saved to train/checkpoints/madlad3b-lora-r1/.
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sacrebleu
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from madlad_loader import BASE_CHECKPOINT, load_madlad
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "round1"
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "madlad3b-lora-r1"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
EVAL_SUBSET_SIZE = 500  # dev subset scored during training; full eval is separate

# T5 projection names; wi (non-gated) kept as fallback — actual set is
# intersected with what the loaded model really has.
LORA_CANDIDATES = {"q", "k", "v", "o", "wi_0", "wi_1", "wi", "wo"}


def find_lora_targets(model) -> list[str]:
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in LORA_CANDIDATES:
                found.add(leaf)
    return sorted(found)


def tokenize(batch, tokenizer):
    inputs = tokenizer(
        [f"<2dv> {en}" for en in batch["en"]],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["dv"], max_length=MAX_TARGET_LENGTH, truncation=True
    )
    inputs["labels"] = labels["input_ids"]
    inputs["length"] = [
        len(i) + len(l) for i, l in zip(inputs["input_ids"], labels["input_ids"])
    ]
    return inputs


def make_compute_metrics(tokenizer):
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, pad_id)
        preds = np.where(preds != -100, preds, pad_id)
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        chrf = sacrebleu.corpus_chrf(pred_str, [label_str], word_order=2)
        return {"chrf_pp": chrf.score}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="50 steps on 2k pairs + 1 eval, then exit.")
    args = parser.parse_args()

    print(f"CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading {BASE_CHECKPOINT} (bf16, rewired) ...")
    tokenizer, model = load_madlad()

    targets = find_lora_targets(model)
    print(f"LoRA target modules: {targets}")
    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # required with gradient checkpointing + LoRA

    ds = load_from_disk(str(DATA_DIR))
    print({k: len(v) for k, v in ds.items()})

    eval_size = 100 if args.smoke else EVAL_SUBSET_SIZE
    eval_ds = ds["dev"].shuffle(seed=42).select(range(eval_size))
    train_ds = ds["train"]
    if args.smoke:
        train_ds = train_ds.shuffle(seed=42).select(range(2000))
        print(f"SMOKE MODE: train={len(train_ds)}, eval={len(eval_ds)}")

    train_tok = train_ds.map(
        lambda b: tokenize(b, tokenizer), batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_tok = eval_ds.map(
        lambda b: tokenize(b, tokenizer), batched=True,
        remove_columns=eval_ds.column_names,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_DIR / "smoke" if args.smoke else OUTPUT_DIR
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1 if args.smoke else 3,
        max_steps=50 if args.smoke else -1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,  # effective batch = 64
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10 if args.smoke else 200,
        weight_decay=0.01,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        label_names=["labels"],
        eval_strategy="steps",
        eval_steps=50 if args.smoke else 400,
        save_strategy="no" if args.smoke else "steps",
        save_steps=400,
        save_total_limit=3,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="chrf_pp",
        greater_is_better=True,
        logging_steps=5 if args.smoke else 25,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=1,  # greedy during training eval; beam-4 in eval/evaluate.py
        report_to="none",
        dataloader_num_workers=0,  # Windows-safe
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    steps_per_epoch = len(train_tok) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )
    print(f"\nsteps/epoch: {steps_per_epoch}, "
          f"total: ~{steps_per_epoch * int(training_args.num_train_epochs)}")

    trainer.train(resume_from_checkpoint=args.resume)

    if args.smoke:
        print("\nSmoke run complete — pipeline validated. Skipping save.")
        return

    print(f"\nSaving adapter to {OUTPUT_DIR} ...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()
