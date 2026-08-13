"""Measure tokenizer fertility (subword tokens per whitespace word) on the
Dhivehi references of the frozen devtest — the paper's model-selection table.

A tokenizer that fragments Thaana into many pieces makes generation slower
and learning harder; MADLAD's 256k vocabulary contains whole Thaana words.

Usage: python scripts/tokenizer_fertility.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_from_disk
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "round1"
OUT = Path(__file__).resolve().parents[1] / "eval" / "outputs" / "tokenizer_fertility.json"

TOKENIZERS = {
    "madlad400-3b-mt": "google/madlad400-3b-mt",
    "gemma-3": "unsloth/gemma-3-1b-it",  # ungated mirror, same tokenizer
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "OLMo-2-7B": "allenai/OLMo-2-1124-7B",
    "bloom": "bigscience/bloom-560m",
    "nllb-200": "facebook/nllb-200-distilled-600M",
}


def main():
    refs = load_from_disk(str(DATA_DIR))["devtest"]["dv"]
    n_words = sum(len(s.split()) for s in refs)
    n_chars = sum(len(s) for s in refs)
    print(f"devtest dv refs: {len(refs)} sentences, {n_words} words, {n_chars} chars")

    results = {"n_sentences": len(refs), "n_words": n_words, "fertility": {}}
    for name, ckpt in TOKENIZERS.items():
        try:
            tok = AutoTokenizer.from_pretrained(ckpt)
        except Exception as e:
            print(f"{name:24s} SKIPPED ({type(e).__name__}: {e})")
            results["fertility"][name] = None
            continue
        n_tokens = sum(
            len(tok.encode(s, add_special_tokens=False)) for s in refs
        )
        fert = n_tokens / n_words
        results["fertility"][name] = round(fert, 2)
        print(f"{name:24s} {fert:5.2f} tok/word  ({n_tokens} tokens)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
