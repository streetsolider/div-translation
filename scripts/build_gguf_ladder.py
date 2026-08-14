"""Quantize the F16 GGUF down the k-quant ladder, plus per-tensor probes.

The ladder (Q8_0 -> Q2_K) is the main quality-vs-size curve. The probes exist
because token_embd and output are 256000x1024 each -- together about a third
of the 3B parameters -- so they dominate the file size and are the obvious
suspects for where Thaana output degrades first. Two probes protect the vocab
tensors while the body drops to 4/2-bit; two do the inverse, wrecking only the
vocab tensors while the body stays at 6-bit. If quality tracks the vocab
tensors rather than the body, the inverse probes will collapse and the
protected ones will not.

Assumes models/gguf/madlad3b-en-dv-f16.gguf exists (scripts/merge_lora.py then
convert_hf_to_gguf.py). Skips artifacts that are already built.

  python scripts/build_gguf_ladder.py
  python scripts/build_gguf_ladder.py --only q4_k_m q2_k
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "models" / "gguf"
F16 = GGUF_DIR / "madlad3b-en-dv-f16.gguf"
QUANTIZE = ROOT / "third_party" / "bin" / "llama-quantize.exe"

# name -> (llama-quantize type, extra per-tensor flags)
LADDER = {
    "q8_0":   ("Q8_0",   []),
    "q6_k":   ("Q6_K",   []),
    "q5_k_m": ("Q5_K_M", []),
    "q4_k_m": ("Q4_K_M", []),
    "q3_k_m": ("Q3_K_M", []),
    "q2_k":   ("Q2_K",   []),
    # vocab tensors protected while the body is quantized hard
    "q4_k_m-emb8": ("Q4_K_M", ["--token-embedding-type", "q8_0",
                               "--output-tensor-type", "q8_0"]),
    "q2_k-emb8":   ("Q2_K",   ["--token-embedding-type", "q8_0",
                               "--output-tensor-type", "q6_K"]),
    # inverse: body stays 6-bit, only a vocab tensor is wrecked
    "q6_k-emb2":   ("Q6_K",   ["--token-embedding-type", "q2_K"]),
    "q6_k-out2":   ("Q6_K",   ["--output-tensor-type", "q2_K"]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, choices=list(LADDER))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if not F16.exists():
        sys.exit(f"missing {F16} - run the converter first")

    names = args.only or list(LADDER)
    sizes = {"f16": round(F16.stat().st_size / 1e6, 1)}
    for name in names:
        qtype, extra = LADDER[name]
        out = GGUF_DIR / f"madlad3b-en-dv-{name}.gguf"
        if out.exists():
            print(f"[skip] {out.name} exists")
        else:
            cmd = [str(QUANTIZE), *extra, str(F16), str(out), qtype, str(args.threads)]
            print(f"\n[build] {out.name}: {' '.join(cmd[1:])}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if proc.returncode != 0 or not out.exists():
                print(proc.stderr[-3000:])
                sys.exit(f"quantize failed for {name}")
        sizes[name] = round(out.stat().st_size / 1e6, 1)
        print(f"  {out.name}: {sizes[name]} MB", flush=True)

    sizes_file = GGUF_DIR / "sizes.json"
    if sizes_file.exists():
        merged = json.loads(sizes_file.read_text(encoding="utf-8"))
        merged.update(sizes)
        sizes = merged
    sizes_file.write_text(json.dumps(sizes, indent=2), encoding="utf-8")
    print(f"\nwrote {sizes_file}")


if __name__ == "__main__":
    main()
