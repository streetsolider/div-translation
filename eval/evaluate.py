"""Evaluate an English->Dhivehi model on the frozen eval sets.

Reports chrF++ (headline), BLEU (secondary), and the Segha keymap
round-trip pass rate (pure-Thaana output check). Works for the out-of-box
MADLAD baseline and for LoRA checkpoints (--adapter).

Usage:
  python eval/evaluate.py                          # baseline on devtest
  python eval/evaluate.py --adapter train/checkpoints/madlad3b-lora-r1
  python eval/evaluate.py --split dev --limit 200  # quick check
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sacrebleu
import torch
from datasets import load_from_disk

from keymap import keymap_to_thaana, thaana_to_keymap
from madlad_loader import BASE_CHECKPOINT, QUANT_MODES, load_madlad
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "round1"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_CHECKPOINT)
    parser.add_argument("--adapter", default=None, help="Path to a PEFT LoRA adapter")
    parser.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--limit", type=int, default=0, help="0 = full split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--tag", default=None, help="Label for the output file")
    parser.add_argument(
        "--quant", default=None, choices=list(QUANT_MODES),
        help="bitsandbytes weight quantization (nf4/fp4 4-bit, int8)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="jsonl file for incremental predictions (resume after a crash)",
    )
    args = parser.parse_args()

    ds = load_from_disk(str(DATA_DIR))[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"split={args.split} n={len(ds)}")

    tokenizer, model = load_madlad(args.model, quant=args.quant)
    if not args.quant:
        model = model.to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"loaded adapter: {args.adapter}")
    model.eval()

    sources = [f"<2dv> {en}" for en in ds["en"]]
    refs = list(ds["dv"])

    done = {}
    ckpt_f = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            for line in open(ckpt_path, encoding="utf-8"):
                r = json.loads(line)
                done[r["idx"]] = r["pred"]
            print(f"resuming: {len(done)}/{len(sources)} already done")
        ckpt_f = open(ckpt_path, "a", encoding="utf-8")

    todo = [j for j in range(len(sources)) if j not in done]
    for k in range(0, len(todo), args.batch_size):
        idxs = todo[k : k + args.batch_size]
        batch = [sources[j] for j in idxs]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, num_beams=args.num_beams
            )
        for j, p in zip(idxs, tokenizer.batch_decode(out, skip_special_tokens=True)):
            done[j] = p
            if ckpt_f:
                ckpt_f.write(json.dumps({"idx": j, "pred": p}, ensure_ascii=False) + "\n")
        if ckpt_f:
            ckpt_f.flush()
        if (k // args.batch_size) % 10 == 0:
            print(f"  {len(done)}/{len(sources)}", flush=True)
    preds = [done[j] for j in range(len(sources))]

    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)  # chrF++
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="char")
    rt_pass = sum(
        1 for p in preds if keymap_to_thaana(thaana_to_keymap(p)) == p
    ) / max(len(preds), 1)

    tag = args.tag or (
        "baseline" if not args.adapter else Path(args.adapter).name
    )
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "quant": args.quant,
        "n": len(preds),
        "num_beams": args.num_beams,
        "chrF++": round(chrf.score, 2),
        "BLEU(char)": round(bleu.score, 2),
        "keymap_roundtrip_pass": round(rt_pass, 4),
    }
    print(json.dumps(result, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{tag}_{args.split}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_metrics": result}, ensure_ascii=False) + "\n")
        for en, ref, pred in zip(ds["en"], refs, preds):
            f.write(
                json.dumps({"en": en, "ref": ref, "pred": pred}, ensure_ascii=False)
                + "\n"
            )
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
