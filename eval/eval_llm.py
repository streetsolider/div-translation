"""Zero-shot EN->DV evaluation of local LLMs (via Ollama) on a fixed
seeded 400-pair subset of the frozen devtest.

Same metrics as evaluate.py (chrF++, char-BLEU, keymap round-trip) so rows
are comparable; the subset is deterministic (seed 13) so every model sees
identical sentences. --rescore recomputes subset metrics from an existing
full-devtest jsonl (for the MADLAD rows) without re-running the model.

Usage:
  python eval/eval_llm.py --model translategemma:12b-it-q4_K_M
  python eval/eval_llm.py --rescore eval/outputs/finetuned-r1_devtest.jsonl
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))

import requests
import sacrebleu
from datasets import load_from_disk

from keymap import keymap_to_thaana, thaana_to_keymap

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "round1"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SUBSET_SEED = 13
SUBSET_N = 400
OLLAMA = "http://localhost:11434"

PROMPT = (
    "Translate the following English sentence into Dhivehi "
    "(the language of the Maldives, written in Thaana script). "
    "Reply with ONLY the Dhivehi translation - no explanation, no romanization, "
    "no quotes.\n\nEnglish: {en}\nDhivehi:"
)

THAANA = re.compile(r"[ހ-޿]")


def subset_indices(n_total):
    idx = random.Random(SUBSET_SEED).sample(range(n_total), min(SUBSET_N, n_total))
    return sorted(idx)


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    lines = [l.strip().strip('"“”').strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return ""
    # prefer the first line that actually contains Thaana
    for l in lines:
        if THAANA.search(l):
            return re.sub(r"^(Dhivehi|Translation)\s*:\s*", "", l, flags=re.I)
    return lines[0]


def query_ollama(model, en, retries=3):
    payload = {
        "model": model,
        "prompt": PROMPT.format(en=en),
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 512},
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=600)
            if r.status_code == 400 and "think" in payload:
                del payload["think"]  # model doesn't accept the think flag
                continue
            r.raise_for_status()
            return r.json()["response"]
        except requests.RequestException as e:
            print(f"  request failed ({e}); retry {attempt + 1}", flush=True)
            time.sleep(10)
    return ""


def metrics(preds, refs):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="char")
    rt = sum(1 for p in preds if p and keymap_to_thaana(thaana_to_keymap(p)) == p)
    return {
        "chrF++": round(chrf.score, 2),
        "BLEU(char)": round(bleu.score, 2),
        "keymap_roundtrip_pass": round(rt / max(len(preds), 1), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Ollama model tag")
    parser.add_argument("--rescore", default=None, help="Existing full-devtest jsonl")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    ds = load_from_disk(str(DATA_DIR))["devtest"]
    idx = subset_indices(len(ds))
    ens = [ds["en"][i] for i in idx]
    refs = [ds["dv"][i] for i in idx]
    print(f"subset n={len(idx)} seed={SUBSET_SEED}", flush=True)

    if args.rescore:
        rows = [json.loads(l) for l in open(args.rescore, encoding="utf-8")]
        rows = [r for r in rows if "_metrics" not in r]
        assert len(rows) == len(ds), "jsonl rows != devtest size"
        picked = [rows[i] for i in idx]
        for p, en in zip(picked, ens):
            assert p["en"] == en, "row order mismatch vs devtest"
        preds = [p["pred"] for p in picked]
        tag = args.tag or Path(args.rescore).stem.replace("_devtest", "")
        model_name = f"rescore:{Path(args.rescore).name}"
    else:
        assert args.model, "need --model or --rescore"
        preds = []
        t0 = time.time()
        for i, en in enumerate(ens):
            preds.append(clean(query_ollama(args.model, en)))
            if (i + 1) % 20 == 0:
                rate = (time.time() - t0) / (i + 1)
                print(f"  {i + 1}/{len(ens)}  {rate:.1f}s/req", flush=True)
        tag = args.tag or re.sub(r"[^\w.-]+", "-", args.model)
        model_name = args.model

    result = {
        "model": model_name,
        "split": f"devtest-subset{len(idx)}",
        "seed": SUBSET_SEED,
        "n": len(preds),
        "empty_preds": sum(1 for p in preds if not p),
        **metrics(preds, refs),
    }
    print(json.dumps(result, indent=2), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{tag}_subset{len(idx)}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_metrics": result}, ensure_ascii=False) + "\n")
        for en, ref, pred in zip(ens, refs, preds):
            f.write(json.dumps({"en": en, "ref": ref, "pred": pred}, ensure_ascii=False) + "\n")
    print(f"wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
