"""Convert the downloaded fp32 MADLAD 7B shards to bf16 in place.

transformers 5.15's weight loader crashes (access violation in its
mmap-slice materialization) on this checkpoint's fp32 shards under
Windows; plain safetensors reads work fine. Converting the shards to
bf16 with plain reads sidesteps the crashing path (no on-the-fly dtype
conversion at load time) and halves disk/IO. Shard names are kept, so
model.safetensors.index.json stays valid.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DIR = Path(__file__).resolve().parents[1] / "models" / "madlad400-7b-mt-bt"

for shard in sorted(DIR.glob("model-*-of-00007.safetensors")):
    tensors = {}
    with safe_open(str(shard), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            tensors[k] = t.to(torch.bfloat16) if t.dtype == torch.float32 else t
    tmp = shard.with_suffix(".bf16tmp")
    save_file(tensors, str(tmp), metadata={"format": "pt"})
    del tensors
    shard.unlink()
    tmp.rename(shard)
    print(f"converted {shard.name}", flush=True)

cfg_path = DIR / "config.json"
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
cfg["torch_dtype"] = "bfloat16"
cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("ALL_CONVERTED", flush=True)
