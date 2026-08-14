"""Evaluate a GGUF (llama.cpp) build of the EN->DV model on the frozen eval sets.

Same protocol as eval/evaluate.py -- same split, same "<2dv> " prefix, same
metrics, same {"_metrics": ...} header line and per-row jsonl -- so results
from both harnesses are directly comparable and share the analysis code.

Two things differ from the HF harness and both are forced by llama.cpp:

  * greedy only. llama.cpp has no beam search for encoder-decoder models, so
    every number here must be read against the bf16 *greedy* baseline (B2),
    never against the beam-4 headline.
  * one process per sentence. llama-server asserts on encoder-decoder models
    ("llama_encode must be called first") and the completion tool encodes
    once per run, so the model is reloaded for each sentence. Startup is
    ~2s of the ~4s per sentence; tokens/sec is therefore taken from
    llama.cpp's own eval counters, which exclude load time.

  python eval/eval_gguf.py --gguf models/gguf/madlad3b-en-dv-q4_k_m.gguf \
      --tag gguf-q4km-s150 --limit 150
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sacrebleu
from datasets import load_from_disk

from keymap import keymap_to_thaana, thaana_to_keymap

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "round1"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_EXE = ROOT / "third_party" / "bin" / "llama-completion.exe"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# llama.cpp prefixes perf lines with a timestamp; "prompt eval time" is a
# separate line, so requiring "eval" right after the colon picks the decode one
EVAL_TIME = re.compile(r"common_perf_print:\s+eval time\s*=\s*([\d.]+) ms /\s*(\d+) runs")


def translate(exe, gguf, prompt, n_predict, n_gpu_layers, threads, ctx):
    """Run one sentence through llama.cpp. Returns (text, eval_ms, n_tokens)."""
    cmd = [
        str(exe), "-m", str(gguf), "-p", prompt,
        "-n", str(n_predict), "-c", str(ctx),
        "--temp", "0", "-ngl", str(n_gpu_layers),
        "--no-warmup", "-no-cnv",
    ]
    if threads:
        cmd += ["-t", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-completion failed ({proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    text = ANSI.sub("", proc.stdout).replace("[end of text]", "").strip()
    m = EVAL_TIME.search(proc.stderr or "")
    eval_ms, n_tok = (float(m.group(1)), int(m.group(2))) if m else (0.0, 0)
    return text, eval_ms, n_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--limit", type=int, default=0, help="0 = full split")
    parser.add_argument("--tag", default=None, help="Label for the output file")
    parser.add_argument("--n-gpu-layers", type=int, default=99, help="0 = CPU only")
    parser.add_argument("--threads", type=int, default=0, help="0 = llama.cpp default")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument(
        "--checkpoint", default=None,
        help="jsonl file for incremental predictions (resume after a crash)",
    )
    args = parser.parse_args()

    gguf = Path(args.gguf)
    ds = load_from_disk(str(DATA_DIR))[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"gguf={gguf.name} split={args.split} n={len(ds)} ngl={args.n_gpu_layers}")

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

    eval_ms_total, tok_total = 0.0, 0
    t0 = time.time()
    todo = [j for j in range(len(sources)) if j not in done]
    for k, j in enumerate(todo):
        pred, eval_ms, n_tok = translate(
            args.exe, gguf, sources[j], args.max_new_tokens,
            args.n_gpu_layers, args.threads, args.ctx,
        )
        done[j] = pred
        eval_ms_total += eval_ms
        tok_total += n_tok
        if ckpt_f:
            ckpt_f.write(json.dumps({"idx": j, "pred": pred}, ensure_ascii=False) + "\n")
            ckpt_f.flush()
        if k % 25 == 0:
            print(f"  {len(done)}/{len(sources)}", flush=True)
    wall = time.time() - t0
    preds = [done[j] for j in range(len(sources))]

    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)  # chrF++
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="char")
    rt_pass = sum(
        1 for p in preds if keymap_to_thaana(thaana_to_keymap(p)) == p
    ) / max(len(preds), 1)

    tag = args.tag or gguf.stem
    result = {
        "model": str(gguf),
        "adapter": None,
        "split": args.split,
        "quant": gguf.stem.split("-")[-1],
        "n": len(preds),
        "num_beams": 1,  # llama.cpp is greedy-only for encoder-decoder
        "chrF++": round(chrf.score, 2),
        "BLEU(char)": round(bleu.score, 2),
        "keymap_roundtrip_pass": round(rt_pass, 4),
        "runtime": "llama.cpp",
        "gguf_file": gguf.name,
        "file_size_mb": round(gguf.stat().st_size / 1e6, 1),
        "backend": "cuda" if args.n_gpu_layers else "cpu",
        "n_gpu_layers": args.n_gpu_layers,
        # from llama.cpp's own counters: excludes per-sentence process startup
        "decode_tokens_per_sec": (
            round(tok_total / (eval_ms_total / 1000), 2) if eval_ms_total else None
        ),
        "wall_seconds": round(wall, 1),
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
