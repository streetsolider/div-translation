# div-translation

**📄 Paper: [Pretraining Coverage Beats Scale](https://streetsolider.github.io/div-translation/paper/paper.html)**
([PDF](paper/pretraining-coverage-beats-scale.pdf)) · **🤗 Adapter: [str33t/madlad400-3b-mt-en-dv-news-v1](https://huggingface.co/str33t/madlad400-3b-mt-en-dv-news-v1)**

English → Dhivehi (Thaana) sentence translation, fine-tuned locally on an
RTX 5070 Ti (16GB). Companion project to
[div-transliteration](https://github.com/streetsolider/div-transliteration),
whose Segha keymap (`keymap.py`) is imported for Dhivehi text normalization
and output QA (round-trip = pure-Thaana check).

## Model

Base: [`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt)
(T5 encoder-decoder, Apache-2.0) — the only strong pretrained MT model
genuinely trained on Dhivehi (~3.5M clean sentences). Its tokenizer covers
Thaana at ~2.1 tokens/word vs ~7–21 for general LLMs. Fine-tuned with LoRA
(r=32, attention + FFN, bf16). Prompt format: `<2dv> {english}`.

Note: NLLB-200 does **not** support Dhivehi, and FLORES-200 has no Dhivehi
split — hence the custom eval set below.

## Data (round 1)

| Source | Pairs | Role |
|---|---|---|
| `alakxender/dhivehi-english-translations` | ~82k train | News sentence pairs (MIT) |
| `google/smol` `gatitos__en_dv` | ~4k ×3 | Professionally translated terminology (CC-BY-4.0) |

Eval: `devtest` (1.5k, frozen, report-only) and `dev` (1k, checkpoint
selection), topic-stratified from the source test split; train is deduped
against both. Headline metric: **chrF++** (BLEU is unreliable on Thaana).

Round 2 (if quality plateaus): filtered `alakxender/dhivehi-english-parallel`
(484k), then back-translation from `alakxender/dhivehi-sentences-extended`.

## Results (round 1, 2026-08-11)

Frozen 1,500-pair devtest, beam-4, identical settings for both rows:

| Model | chrF++ | char-BLEU | keymap round-trip |
|---|---|---|---|
| madlad400-3b-mt (out of box) | 42.59 | 48.36 | 94.6% |
| + LoRA r=32, 3 epochs (~10.3h on RTX 5070 Ti) | **61.82** | **72.21** | **100%** |

Best checkpoint by dev chrF++ (61.45 @ step 4000/4374). Adapter:
`train/checkpoints/madlad3b-lora-r1` (360 MB).

### Comparison vs open models (same frozen devtest)

| System | chrF++ | char-BLEU | keymap round-trip |
|---|---|---|---|
| **3B + LoRA (ours)** | **61.82** | **72.21** | **100%** |
| madlad400-3b-mt (base) | 42.59 | 48.36 | 94.6% |
| madlad400-7b-mt-bt (NF4)\* | 17.83 | 8.78 | 80.3% |
| Flan-T5 en2dv 250M (alakxender) | 24.90 | 11.09 | 53.8% |
| Neobe mT5-large 1.2B (synthetic targets) | 55.75 | 67.15 | 100% |
| Neobe ByT5-large 1.2B (synthetic targets) | 49.61 | 60.07 | 100% |
| TranslateGemma 12B (q4)† | 33.18 | 40.83 | 96.3% |
| Gemma 4 12B (qat)† | 23.22 | 25.52 | 48.3% |
| Qwen3 14B (q4)† | 3.89 | 1.35 | 92.3% |
| NLLB-200 | — | no Dhivehi | support — |

\* Degenerates into repetition loops on ~half of inputs under NF4 (chrF++
49.0 on the clean half vs 40.7 for the 3B base on the same rows); the 3B
under identical NF4 loses only 0.3 chrF++, so the instability is specific
to that checkpoint. † Zero-shot via Ollama, fixed seeded 400-pair subset.

Full paper: [web version](https://streetsolider.github.io/div-translation/paper/paper.html)
· [PDF](paper/pretraining-coverage-beats-scale.pdf). Fine-tuned adapter:
[str33t/madlad400-3b-mt-en-dv-news-v1](https://huggingface.co/str33t/madlad400-3b-mt-en-dv-news-v1).

Note: transformers 5.x mis-ties MADLAD's embeddings and produces garbage —
all scripts load via `madlad_loader.py`, which rewires the checkpoint
correctly. See that module's docstring. Its lazy weight loader also
intermittently access-violates on Windows with large sharded checkpoints
(see `scripts/robust_download.py` + `scripts/convert_7b_bf16.py` for the
workaround chain we used for the 7B row).

## Usage

```powershell
.venv\Scripts\activate
python scripts\smoke_test.py        # verify GPU + model + <2dv> token
python data\build_dataset.py        # build data/processed/round1
python eval\evaluate.py             # out-of-box baseline chrF++ on devtest
python train\finetune.py --smoke    # 50-step pipeline check
python train\finetune.py            # full LoRA fine-tune (~3 epochs)
python eval\evaluate.py --adapter train\checkpoints\madlad3b-lora-r1
```
