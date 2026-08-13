"""Evaluate alakxender/flan-t5-base-dhivehi-en-latin (en2dv mode) on the
frozen devtest — the only public en->dv model we found (multi-task
Flan-T5). Uses the author's own Space decode settings (beam 4,
repetition_penalty 1.2, no_repeat_ngram_size 3) and prefix format.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))

import sacrebleu
import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from keymap import keymap_to_thaana, thaana_to_keymap

MODEL = "alakxender/flan-t5-base-dhivehi-en-latin"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "round1"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

ds = load_from_disk(str(DATA_DIR))["devtest"]
print(f"devtest n={len(ds)}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to("cuda").eval()

sources = [f"en2dv: {en}" for en in ds["en"]]
refs = list(ds["dv"])
preds = []
BS = 32
for i in range(0, len(sources), BS):
    batch = sources[i : i + BS]
    inputs = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    if (i // BS) % 5 == 0:
        print(f"  {i + len(batch)}/{len(sources)}", flush=True)

chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="char")
rt = sum(1 for p in preds if p and keymap_to_thaana(thaana_to_keymap(p)) == p) / len(preds)
result = {
    "model": MODEL,
    "split": "devtest",
    "n": len(preds),
    "num_beams": 4,
    "decode": "author Space settings (rep_pen 1.2, no_repeat_ngram 3)",
    "chrF++": round(chrf.score, 2),
    "BLEU(char)": round(bleu.score, 2),
    "keymap_roundtrip_pass": round(rt, 4),
}
print(json.dumps(result, indent=2), flush=True)

out_file = OUT_DIR / "flant5-endv_devtest.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(json.dumps({"_metrics": result}, ensure_ascii=False) + "\n")
    for en, ref, pred in zip(ds["en"], refs, preds):
        f.write(json.dumps({"en": en, "ref": ref, "pred": pred}, ensure_ascii=False) + "\n")
print(f"wrote {out_file}", flush=True)
