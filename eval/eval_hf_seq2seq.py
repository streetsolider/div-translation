"""Evaluate any HF seq2seq en->dv model on the frozen devtest with the
standard protocol (beam 4, max_new_tokens 256) — used for prior-art rows
(Neobe en-dhivehi models, etc.).

Usage:
  python eval/eval_hf_seq2seq.py --model Neobe/en-dhivehi-mt5-large-sentence
  python eval/eval_hf_seq2seq.py --model X --prefix "translate: " --tag mytag
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))

import sacrebleu
import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from keymap import keymap_to_thaana, thaana_to_keymap

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "round1"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prefix", default="")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--tag", default=None)
args = parser.parse_args()

ds = load_from_disk(str(DATA_DIR))["devtest"]
print(f"devtest n={len(ds)} model={args.model}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch.float32)
model = model.to("cuda").eval()

sources = [f"{args.prefix}{en}" for en in ds["en"]]
refs = list(ds["dv"])
preds = []
for i in range(0, len(sources), args.batch_size):
    batch = sources[i : i + args.batch_size]
    inputs = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, num_beams=4)
    preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    if (i // args.batch_size) % 10 == 0:
        print(f"  {i + len(batch)}/{len(sources)}", flush=True)

chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="char")
rt = sum(1 for p in preds if p and keymap_to_thaana(thaana_to_keymap(p)) == p) / len(preds)
result = {
    "model": args.model,
    "split": "devtest",
    "n": len(preds),
    "num_beams": 4,
    "chrF++": round(chrf.score, 2),
    "BLEU(char)": round(bleu.score, 2),
    "keymap_roundtrip_pass": round(rt, 4),
}
print(json.dumps(result, indent=2), flush=True)

tag = args.tag or re.sub(r"[^\w.-]+", "-", args.model.split("/")[-1])
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_file = OUT_DIR / f"{tag}_devtest.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(json.dumps({"_metrics": result}, ensure_ascii=False) + "\n")
    for en, ref, pred in zip(ds["en"], refs, preds):
        f.write(json.dumps({"en": en, "ref": ref, "pred": pred}, ensure_ascii=False) + "\n")
print(f"wrote {out_file}", flush=True)
