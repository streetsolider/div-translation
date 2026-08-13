"""Verify the MADLAD weight-wiring fix under transformers 5.x.

The checkpoint ships only decoder.embed_tokens.weight (the true shared
embedding) and a distinct lm_head.weight. Transformers 5.x ties
shared/encoder/lm_head from the wrong tensor. Correct wiring:
  shared = encoder.embed = decoder.embed = ckpt decoder.embed_tokens.weight
  lm_head = ckpt lm_head.weight (untied)
Expected after fix: '<2pt> I love pizza!' -> 'Eu adoro pizza!'
"""

import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch
from safetensors import safe_open
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CKPT = "google/madlad400-3b-mt"

tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSeq2SeqLM.from_pretrained(CKPT, dtype=torch.bfloat16)

path = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--madlad400-3b-mt/snapshots/*/model.safetensors"
    )
)[0]
with safe_open(path, framework="pt") as f:
    true_embed = f.get_tensor("decoder.embed_tokens.weight").to(torch.bfloat16)
    true_lm_head = f.get_tensor("lm_head.weight").to(torch.bfloat16)

model.config.tie_word_embeddings = False
model.shared = torch.nn.Embedding.from_pretrained(true_embed, freeze=False)
model.encoder.embed_tokens = model.shared
model.decoder.embed_tokens = model.shared
model.lm_head = torch.nn.Linear(
    true_lm_head.shape[1], true_lm_head.shape[0], bias=False, dtype=torch.bfloat16
)
model.lm_head.weight.data = true_lm_head
model = model.to("cuda")
model.eval()

for prompt in (
    "<2pt> I love pizza!",
    "<2dv> I love pizza!",
    "<2dv> The government announced a new education policy yesterday.",
    "<2en> ދިވެހިރާއްޖޭގެ ސަރުކާރުން އައު ތައުލީމީ ސިޔާސަތެއް އިއުލާނުކޮށްފި",
):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, num_beams=4)
    print(f"{prompt!r}\n  -> {tokenizer.decode(out[0], skip_special_tokens=True)!r}")
