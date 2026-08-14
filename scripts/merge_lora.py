"""Merge the LoRA adapter into the base model and save a standalone checkpoint.

Every quantization track (bitsandbytes, GGUF/llama.cpp, CTranslate2) starts
from a single merged checkpoint rather than base+adapter, so the wiring fix
from madlad_loader has to be baked into the saved files: config must say
tie_word_embeddings=False and lm_head.weight must be stored as its own
tensor. If it is saved tied, every downstream converter reproduces the
transformers 5.x garbage-generation bug in a form we cannot patch at runtime.

Saved as one unsharded safetensors file: transformers 5.15 intermittently
access-violates on Windows when lazily loading large sharded checkpoints,
and the GGUF converter is happier with one file too.

  python scripts/merge_lora.py
  python scripts/merge_lora.py --adapter train/checkpoints/madlad3b-lora-r1/checkpoint-4000
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors import safe_open

from madlad_loader import BASE_CHECKPOINT, load_madlad

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "train" / "checkpoints" / "madlad3b-lora-r1"
DEFAULT_OUT = ROOT / "models" / "madlad3b-lora-r1-merged"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_CHECKPOINT)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {out}")

    print(f"loading base {args.base} (bf16, CPU)...", flush=True)
    tokenizer, model = load_madlad(args.base)

    # the loader rewires only when transformers got the tie wrong; either way
    # the merged checkpoint must go out untied
    assert not torch.equal(
        model.lm_head.weight.data, model.shared.weight.data
    ), "lm_head is tied to the embedding before merge - loader fix did not apply"

    print(f"applying adapter {args.adapter}...", flush=True)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.config.tie_word_embeddings = False
    if hasattr(model, "generation_config"):
        model.generation_config.tie_word_embeddings = None

    assert not torch.equal(
        model.lm_head.weight.data, model.shared.weight.data
    ), "lm_head tied to embedding after merge"

    print(f"saving to {out}...", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), max_shard_size="20GB", safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    # convert_hf_to_gguf.py wants the sentencepiece model, which the fast
    # tokenizer does not always write out
    if not (out / "spiece.model").exists():
        src = _find_spiece(args.base)
        if src:
            shutil.copy2(src, out / "spiece.model")
            print(f"copied spiece.model from {src}")
        else:
            print("WARNING: no spiece.model found - GGUF conversion may fail")

    verify(out)


def _find_spiece(checkpoint: str):
    local = Path(checkpoint)
    if local.is_dir() and (local / "spiece.model").exists():
        return local / "spiece.model"
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    pattern = f"models--{checkpoint.replace('/', '--')}"
    hits = sorted((cache / pattern).glob("snapshots/*/spiece.model")) if (cache / pattern).exists() else []
    return hits[0] if hits else None


def verify(out: Path):
    """Confirm the saved files really carry an untied lm_head."""
    print("\n--- verifying saved checkpoint ---")
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    print(f"config.tie_word_embeddings = {cfg.get('tie_word_embeddings')}")
    assert cfg.get("tie_word_embeddings") is False, "saved config still ties embeddings"

    index = out / "model.safetensors.index.json"
    assert not index.exists(), "checkpoint is sharded; expected a single file"
    with safe_open(out / "model.safetensors", framework="pt") as f:
        names = set(f.keys())
        assert "lm_head.weight" in names, "lm_head.weight missing from saved file"
        embed = next(
            (n for n in ("shared.weight", "encoder.embed_tokens.weight",
                         "decoder.embed_tokens.weight") if n in names),
            None,
        )
        assert embed, "no embedding tensor in saved file"
        lm = f.get_tensor("lm_head.weight")
        emb = f.get_tensor(embed)
        print(f"embedding tensor: {embed} {tuple(emb.shape)}")
        print(f"lm_head.weight    {tuple(lm.shape)}")
        assert not torch.equal(lm, emb), "saved lm_head is identical to the embedding"
    size_gb = (out / "model.safetensors").stat().st_size / 1e9
    print(f"model.safetensors: {size_gb:.2f} GB")

    # reload with plain transformers (no loader fix) - this is what every
    # downstream tool will do
    from transformers import AutoModelForSeq2SeqLM

    print("reloading with plain AutoModelForSeq2SeqLM...", flush=True)
    m = AutoModelForSeq2SeqLM.from_pretrained(str(out), dtype=torch.bfloat16)
    assert not torch.equal(
        m.lm_head.weight.data, m.shared.weight.data
    ), "plain reload re-tied lm_head to the embedding"
    print("OK: plain reload keeps lm_head untied")


if __name__ == "__main__":
    main()
