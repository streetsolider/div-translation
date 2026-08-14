"""Smoke-test generation for MADLAD checkpoints (weight-wiring sanity check).

The checkpoints ship only decoder.embed_tokens.weight (the true shared
embedding) and a distinct lm_head.weight. Transformers 5.x ties
shared/encoder/lm_head from the wrong tensor. Correct wiring:
  shared = encoder.embed = decoder.embed = ckpt decoder.embed_tokens.weight
  lm_head = ckpt lm_head.weight (untied)
Expected: '<2pt> I love pizza!' -> 'Eu adoro pizza!', and Thaana (not
mojibake or repetition loops) for the <2dv> prompts.

Runs through madlad_loader, so it doubles as the per-artifact smoke test for
merged and quantized checkpoints:

  python scripts/diag_generation.py
  python scripts/diag_generation.py --model models/madlad3b-lora-r1-merged
  python scripts/diag_generation.py --model models/madlad3b-lora-r1-merged --quant nf4
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from madlad_loader import BASE_CHECKPOINT, load_madlad

PROMPTS = (
    "<2pt> I love pizza!",
    "<2dv> I love pizza!",
    "<2dv> The government announced a new education policy yesterday.",
    "<2en> ދިވެހިރާއްޖޭގެ ސަރުކާރުން އައު ތައުލީމީ ސިޔާސަތެއް އިއުލާނުކޮށްފި",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_CHECKPOINT)
    parser.add_argument("--adapter", default=None, help="Path to a PEFT LoRA adapter")
    parser.add_argument("--quant", default=None, choices=["nf4", "fp4", "int8"])
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    tokenizer, model = load_madlad(args.model, quant=args.quant)
    if not args.quant:
        model = model.to("cuda")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    tied = torch.equal(
        model.lm_head.weight.data.cpu(), model.get_input_embeddings().weight.data.cpu()
    )
    print(f"model={args.model} quant={args.quant} lm_head_tied={tied}")
    if tied:
        print("WARNING: lm_head is tied to the embedding - generation will be garbage")

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        print(f"{prompt!r}\n  -> {tokenizer.decode(out[0], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
