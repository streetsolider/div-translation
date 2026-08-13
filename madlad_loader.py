"""Load MADLAD-400 MT checkpoints with correct weight wiring.

The checkpoints ship only decoder.embed_tokens.weight (the true shared
embedding, used by encoder AND decoder) plus a distinct, untied
lm_head.weight — but their config says tie_word_embeddings=True. Transformers
5.x can resolve that conflict by tying shared/encoder.embed/lm_head from the
lm_head tensor, which breaks generation entirely (transformers 4.x happened
to wire it correctly). load_madlad() detects the bad tie and rewires from
the checkpoint file. Handles single-file and sharded checkpoints, and
optional NF4 quantization for the 7B/10B models (quant="nf4").
"""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_CHECKPOINT = "google/madlad400-3b-mt"


def _load_tensor(checkpoint: str, name: str, dtype):
    """Fetch one tensor from a (possibly sharded, possibly local) checkpoint."""
    local = Path(checkpoint)
    if local.is_dir():
        index = local / "model.safetensors.index.json"
        if index.exists():
            shard = json.loads(index.read_text(encoding="utf-8"))["weight_map"][name]
            path = local / shard
        else:
            path = local / "model.safetensors"
    else:
        try:
            path = hf_hub_download(checkpoint, "model.safetensors")
        except Exception:
            index = hf_hub_download(checkpoint, "model.safetensors.index.json")
            with open(index, encoding="utf-8") as f:
                shard = json.load(f)["weight_map"][name]
            path = hf_hub_download(checkpoint, shard)
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name).to(dtype)


def load_madlad(checkpoint: str = BASE_CHECKPOINT, dtype=torch.bfloat16, quant=None):
    """Return (tokenizer, model) with verified-correct embedding/lm_head wiring.

    quant="nf4" loads 4-bit NF4 via bitsandbytes with device_map (do NOT call
    .to("cuda") on the result); quant=None loads plain bf16 on CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        # NOTE: transformers 5.15's lazy weight loading intermittently
        # access-violates on Windows with large sharded checkpoints
        # (crash is in its mmap-slice materialization; plain safetensors
        # reads of the same files are reliable). The state_dict= loading
        # strategy skips on-the-fly quantization, so it is not a usable
        # bypass; callers should retry this load on hard crashes.
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            dtype=dtype,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            ),
        )
    elif quant is not None:
        raise ValueError(f"unknown quant: {quant}")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dtype=dtype)

    # shared embedding and lm_head stay unquantized under bnb, so this
    # comparison is valid in both load modes
    if torch.equal(
        model.lm_head.weight.data.cpu(), model.shared.weight.data.cpu()
    ):
        device = model.lm_head.weight.device
        true_embed = _load_tensor(checkpoint, "decoder.embed_tokens.weight", dtype)
        true_lm_head = _load_tensor(checkpoint, "lm_head.weight", dtype)
        model.config.tie_word_embeddings = False
        emb = torch.nn.Embedding.from_pretrained(
            true_embed.to(device), freeze=False
        )
        model.shared = emb
        model.encoder.embed_tokens = emb
        model.decoder.embed_tokens = emb
        lm_head = torch.nn.Linear(
            true_lm_head.shape[1], true_lm_head.shape[0], bias=False, dtype=dtype
        )
        lm_head.weight.data = true_lm_head
        model.lm_head = lm_head.to(device)

    return tokenizer, model
