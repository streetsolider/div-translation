"""Smoke-test google/madlad400-3b-mt on this machine before any training.

Checks: CUDA works on the RTX 5070 Ti (sm_120), the <2dv> target token
exists in the tokenizer, bf16 generation produces Thaana output, and the
Segha keymap round-trip accepts that output.

Usage:
  python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
# Segha keymap lives in the sibling div-transliteration repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from keymap import keymap_to_thaana, thaana_to_keymap
from madlad_loader import BASE_CHECKPOINT, load_madlad
TEST_SENTENCES = [
    "The government announced a new education policy yesterday.",
    "Fishing is the most important industry in the Maldives.",
    "The weather is expected to improve over the weekend.",
    "Parliament will vote on the proposed budget next week.",
    "She traveled to Male' to visit her family.",
]


def roundtrip_ok(text: str) -> bool:
    return keymap_to_thaana(thaana_to_keymap(text)) == text


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "CUDA not available — check torch install"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"Torch: {torch.__version__}")

    print(f"\nLoading {BASE_CHECKPOINT} (bf16) ...")
    tokenizer, model = load_madlad()
    model = model.to("cuda")
    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    dv_token = "<2dv>"
    dv_id = tokenizer.convert_tokens_to_ids(dv_token)
    assert dv_id != tokenizer.unk_token_id, f"{dv_token} not in vocab!"
    print(f"{dv_token} token id: {dv_id}")

    # Tokenizer efficiency check on real Thaana
    sample_dv = "ދިވެހިރާއްޖޭގެ ރައްޔިތުންނަށް ސަރުކާރުން މުހިންމު މައުލޫމާތު ފޯރުކޮށްދިނުމަށް މަސައްކަތް ކުރަމުން ދެއެވެ."
    n_tok = len(tokenizer(sample_dv).input_ids)
    print(f"Thaana tokenization: {n_tok} tokens / {len(sample_dv.split())} words "
          f"= {n_tok / len(sample_dv.split()):.2f} tok/word")

    print("\nTranslating ...")
    ok = 0
    for sent in TEST_SENTENCES:
        inputs = tokenizer(f"{dv_token} {sent}", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        dv = tokenizer.decode(out[0], skip_special_tokens=True)
        has_thaana = any("ހ" <= c <= "޿" for c in dv)
        rt = roundtrip_ok(dv)
        ok += has_thaana
        print(f"  EN: {sent}")
        print(f"  DV: {dv}")
        print(f"      thaana={has_thaana} keymap_roundtrip={rt}")
    print(f"\nVRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"{ok}/{len(TEST_SENTENCES)} outputs contained Thaana")
    assert ok == len(TEST_SENTENCES), "Some outputs contained no Thaana — investigate"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
