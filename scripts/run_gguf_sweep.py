"""Run eval/eval_gguf.py across every artifact in the GGUF ladder.

Screening pass (--limit 150) first, so a full 1500-row run is only spent on
the interesting points. Tags carry an -s150 suffix so screen outputs never
overwrite full-run outputs.

  python scripts/run_gguf_sweep.py --limit 150
  python scripts/run_gguf_sweep.py --limit 0 --only q8_0 q4_k_m q3_k_m
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "models" / "gguf"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

ORDER = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k",
         "q4_k_m-emb8", "q2_k-emb8", "q6_k-emb2", "q6_k-out2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=150, help="0 = full split")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    args = parser.parse_args()

    names = args.only or ORDER
    suffix = f"-s{args.limit}" if args.limit else ""
    for name in names:
        gguf = GGUF_DIR / f"madlad3b-en-dv-{name}.gguf"
        if not gguf.exists():
            print(f"[skip] missing {gguf.name}")
            continue
        tag = f"gguf-{name}{suffix}"
        out = ROOT / "eval" / "outputs" / f"{tag}_devtest.jsonl"
        if out.exists():
            print(f"[skip] {out.name} exists")
            continue
        cmd = [
            str(PY), str(ROOT / "eval" / "eval_gguf.py"),
            "--gguf", str(gguf), "--tag", tag,
            "--n-gpu-layers", str(args.n_gpu_layers),
            "--checkpoint", str(ROOT / "eval" / f"ckpt_{tag}.jsonl"),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"\n=== {tag} ===", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            sys.exit(f"eval failed for {name}")
        print("\n".join(proc.stdout.strip().splitlines()[-20:]), flush=True)


if __name__ == "__main__":
    main()
