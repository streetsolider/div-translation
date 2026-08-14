"""Measure size, memory and throughput for each runtime/quantization.

Quality is measured elsewhere (evaluate.py, eval_gguf.py); this answers the
other half of the local-hardware question: what does each artifact cost to
run? A fixed seeded 64-sentence subset of the frozen devtest is translated by
every configuration, and we record wall-clock, generated tokens/sec, peak
host RAM and peak VRAM.

Measurement notes, so the paper can state them honestly:
  * llama.cpp runs one process per sentence, so wall-clock includes a model
    load per sentence. We report both: "decode_tokens_per_sec" comes from
    llama.cpp's own eval counter (load excluded, the number a resident server
    would see) and "wall_seconds_per_sentence" includes everything.
  * peak VRAM for HF runs is torch.cuda.max_memory_allocated (allocator
    accounting, excludes the CUDA context); for llama.cpp it is sampled from
    nvidia-smi, which includes it. The two are not directly comparable and
    are labelled separately.
  * peak RAM is the max RSS of the process tree, sampled at 50 ms.

  python eval/bench_speed.py --hf --gguf all
  python eval/bench_speed.py --gguf q4_k_m --cpu
"""

import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "round1"
OUT_FILE = Path(__file__).resolve().parent / "outputs" / "speed_bench.json"
GGUF_DIR = ROOT / "models" / "gguf"
MERGED = ROOT / "models" / "madlad3b-lora-r1-merged"
EXE = ROOT / "third_party" / "bin" / "llama-completion.exe"

SUBSET_SEED = 13
SUBSET_N = 64
LADDER = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k",
          "q4_k_m-emb8", "q2_k-emb8", "q6_k-emb2", "q6_k-out2"]
EVAL_TIME = re.compile(r"common_perf_print:\s+eval time\s*=\s*([\d.]+) ms /\s*(\d+) runs")


def subset_sentences():
    from datasets import load_from_disk

    ds = load_from_disk(str(DATA_DIR))["devtest"]
    idx = random.Random(SUBSET_SEED).sample(range(len(ds)), min(SUBSET_N, len(ds)))
    return [ds[i]["en"] for i in idx]


class PeakSampler(threading.Thread):
    """Sample peak RSS of a process tree, and peak GPU memory, every 50 ms."""

    def __init__(self, pid=None, gpu=False):
        super().__init__(daemon=True)
        self.pid, self.gpu = pid, gpu
        self.peak_rss_mb, self.peak_vram_mb = 0.0, 0.0
        self._done = threading.Event()

    def run(self):
        import psutil

        while not self._done.is_set():
            if self.pid:
                try:
                    proc = psutil.Process(self.pid)
                    rss = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except psutil.Error:
                            pass
                    self.peak_rss_mb = max(self.peak_rss_mb, rss / 1e6)
                except psutil.Error:
                    pass
            if self.gpu:
                self.peak_vram_mb = max(self.peak_vram_mb, _nvidia_smi_used())
            self._done.wait(0.05)

    def stop(self):
        self._done.set()
        self.join(timeout=2)


def _nvidia_smi_used():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


def bench_gguf(name, sentences, n_gpu_layers, threads, max_new_tokens):
    import os

    gguf = GGUF_DIR / f"madlad3b-en-dv-{name}.gguf"
    label = f"gguf:{name}:{'cuda' if n_gpu_layers else 'cpu'}"
    print(f"--- {label}", flush=True)
    sampler = PeakSampler(pid=os.getpid(), gpu=bool(n_gpu_layers))
    sampler.start()
    eval_ms, n_tok = 0.0, 0
    t0 = time.time()
    for s in sentences:
        cmd = [
            str(EXE), "-m", str(gguf), "-p", f"<2dv> {s}",
            "-n", str(max_new_tokens), "-c", "512", "--temp", "0",
            "-ngl", str(n_gpu_layers), "--no-warmup", "-no-cnv",
        ]
        if threads:
            cmd += ["-t", str(threads)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        m = EVAL_TIME.search(proc.stderr or "")
        if m:
            eval_ms += float(m.group(1))
            n_tok += int(m.group(2))
    wall = time.time() - t0
    sampler.stop()
    return {
        "label": label,
        "runtime": "llama.cpp",
        "quant": name,
        "backend": "cuda" if n_gpu_layers else "cpu",
        "threads": threads or "default",
        "file_size_mb": round(gguf.stat().st_size / 1e6, 1),
        "n_sentences": len(sentences),
        "gen_tokens": n_tok,
        "decode_tokens_per_sec": round(n_tok / (eval_ms / 1000), 2) if eval_ms else None,
        "wall_seconds": round(wall, 1),
        "wall_seconds_per_sentence": round(wall / len(sentences), 2),
        "peak_vram_mb_nvidia_smi": round(sampler.peak_vram_mb, 1) or None,
        "note": "one process per sentence; wall includes a model load per sentence",
    }


def bench_hf(sentences, quant, num_beams, batch_size, max_new_tokens):
    import os

    import torch

    from madlad_loader import load_madlad

    label = f"hf:{quant or 'bf16'}:beam{num_beams}"
    print(f"--- {label}", flush=True)
    tokenizer, model = load_madlad(str(MERGED), quant=quant)
    if not quant:
        model = model.to("cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    sampler = PeakSampler(pid=os.getpid(), gpu=True)
    sampler.start()

    n_tok = 0
    t0 = time.time()
    for k in range(0, len(sentences), batch_size):
        batch = [f"<2dv> {s}" for s in sentences[k : k + batch_size]]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
            )
        n_tok += int((out != tokenizer.pad_token_id).sum())
    wall = time.time() - t0
    sampler.stop()
    result = {
        "label": label,
        "runtime": "transformers",
        "quant": quant or "bf16",
        "backend": "cuda",
        "batch_size": batch_size,
        "num_beams": num_beams,
        "n_sentences": len(sentences),
        "gen_tokens": n_tok,
        "decode_tokens_per_sec": round(n_tok / wall, 2),
        "wall_seconds": round(wall, 1),
        "wall_seconds_per_sentence": round(wall / len(sentences), 2),
        "peak_vram_mb_torch_alloc": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "peak_vram_mb_nvidia_smi": round(sampler.peak_vram_mb, 1) or None,
        "peak_rss_mb": round(sampler.peak_rss_mb, 1),
        "note": "batched generation, model resident",
    }
    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", action="store_true", help="bench the bf16 HF runtime")
    parser.add_argument("--hf-quant", nargs="*", default=[],
                        help="bitsandbytes modes to bench, e.g. nf4 int8")
    parser.add_argument("--gguf", nargs="*", default=[],
                        help="ladder names, or 'all'")
    parser.add_argument("--cpu", action="store_true",
                        help="also bench each GGUF with -ngl 0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    sentences = subset_sentences()
    print(f"subset n={len(sentences)} seed={SUBSET_SEED}", flush=True)

    results = []
    if OUT_FILE.exists():
        results = json.loads(OUT_FILE.read_text(encoding="utf-8"))["runs"]
    have = {r["label"] for r in results}

    def add(r):
        results[:] = [x for x in results if x["label"] != r["label"]]
        results.append(r)
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(
            json.dumps({"seed": SUBSET_SEED, "n": len(sentences), "runs": results},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    names = LADDER if args.gguf == ["all"] else args.gguf
    for name in names:
        for ngl in ([99, 0] if args.cpu else [99]):
            label = f"gguf:{name}:{'cuda' if ngl else 'cpu'}"
            if label in have:
                print(f"[skip] {label}")
                continue
            add(bench_gguf(name, sentences, ngl, args.threads, args.max_new_tokens))

    if args.hf:
        for beams in (4, 1):
            if f"hf:bf16:beam{beams}" not in have:
                add(bench_hf(sentences, None, beams, args.batch_size, args.max_new_tokens))
    for q in args.hf_quant:
        if f"hf:{q}:beam4" not in have:
            add(bench_hf(sentences, q, 4, args.batch_size, args.max_new_tokens))

    print(f"\nwrote {OUT_FILE}")
    for r in results:
        print(f"  {r['label']:<28} {r.get('decode_tokens_per_sec')} tok/s  "
              f"{r.get('file_size_mb', '-')} MB")


if __name__ == "__main__":
    main()
