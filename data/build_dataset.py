"""Build the round-1 English->Dhivehi training corpus and frozen eval sets.

Sources:
  - alakxender/dhivehi-english-translations (91.8k news sentence pairs, MIT)
  - google/smol config gatitos__en_dv (~4k professionally translated phrases,
    CC-BY-4.0) — upsampled as terminology grounding

Normalization (Dhivehi side): NFC + ASCII ,;? -> Arabic ،؛؟ (matching the
div-transliteration corpus convention) + whitespace collapse.

Eval sets are carved from the source test split, stratified by topic:
devtest (frozen, report-only) and dev (checkpoint selection). Train is
exact-deduped against both on either side of the pair.

Usage:
  python data/build_dataset.py

Output: data/processed/round1 (HF DatasetDict: train/dev/devtest)
"""

import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from keymap import keymap_to_thaana, thaana_to_keymap  # noqa: F401  (roundtrip QA)

OUT_DIR = Path(__file__).resolve().parent / "processed" / "round1"
SEED = 42
DEVTEST_SIZE = 1500
DEV_SIZE = 1000
GATITOS_UPSAMPLE = 3

# Arabic punctuation normalization on the Thaana side — same convention the
# transliteration corpus used to reach 0 round-trip failures.
_ARABIC_PUNCT = str.maketrans({",": "،", ";": "؛", "?": "؟"})
_WS = re.compile(r"\s+")


def has_thaana(s: str) -> bool:
    return any("ހ" <= c <= "޿" for c in s)


def thaana_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha() or "ހ" <= c <= "޿"]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ހ" <= c <= "޿") / len(letters)


def norm_en(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", s)).strip()


def norm_dv(s: str) -> str:
    s = _WS.sub(" ", unicodedata.normalize("NFC", s)).strip()
    if has_thaana(s):
        s = s.translate(_ARABIC_PUNCT)
    return s


def keep_pair(en: str, dv: str) -> bool:
    if not en or not dv or not has_thaana(dv):
        return False
    if len(en) < 3 or len(en) > 600 or len(dv) > 800:
        return False
    # Mostly-Thaana target (allows the odd Latin acronym, rejects mixed junk)
    if thaana_ratio(dv) < 0.7:
        return False
    # Char-length ratio sanity — catches gross misalignments
    ratio = len(dv) / max(len(en), 1)
    return 0.25 <= ratio <= 4.0


def load_news() -> DatasetDict:
    ds = load_dataset("alakxender/dhivehi-english-translations")
    print(f"news source: {({k: len(v) for k, v in ds.items()})}")
    print(f"news columns: {ds['train'].column_names}")
    return ds


def to_pairs(ds, en_col: str, dv_col: str, source: str, topic_col: str | None = None):
    rows = {"en": [], "dv": [], "topic": [], "source": []}
    for ex in ds:
        en, dv = norm_en(str(ex[en_col])), norm_dv(str(ex[dv_col]))
        if not keep_pair(en, dv):
            continue
        rows["en"].append(en)
        rows["dv"].append(dv)
        rows["topic"].append(str(ex.get(topic_col, "")) if topic_col else "")
        rows["source"].append(source)
    return Dataset.from_dict(rows)


def load_gatitos() -> Dataset:
    ds = load_dataset("google/smol", "gatitos__en_dv", split="train")
    print(f"gatitos columns: {ds.column_names}, rows: {len(ds)}")
    rows = {"en": [], "dv": [], "topic": [], "source": []}
    for ex in ds:
        en = norm_en(str(ex.get("src", "")))
        # GATITOS carries one-or-more target renderings per source
        trgs = ex.get("trgs") or ([ex["trg"]] if "trg" in ex else [])
        for trg in trgs:
            dv = norm_dv(str(trg))
            if not en or not dv or not has_thaana(dv):
                continue
            rows["en"].append(en)
            rows["dv"].append(dv)
            rows["topic"].append("gatitos")
            rows["source"].append("gatitos")
    return Dataset.from_dict(rows)


def stratified_indices(topics: list[str], n: int, taken: set[int]) -> list[int]:
    """Sample n indices spread proportionally across topics, skipping taken."""
    by_topic = defaultdict(list)
    for i, t in enumerate(topics):
        if i not in taken:
            by_topic[t].append(i)
    total = sum(len(v) for v in by_topic.values())
    picked = []
    for t, idxs in sorted(by_topic.items()):
        k = max(1, round(n * len(idxs) / total))
        picked.extend(idxs[:k])
    return picked[:n]


def main():
    news = load_news()
    train_pairs = to_pairs(news["train"], "english", "dhivehi", "news", "topic")
    test_pairs = to_pairs(news["test"], "english", "dhivehi", "news", "topic")
    print(f"after filters: train={len(train_pairs)}, test={len(test_pairs)}")

    # Carve frozen eval sets from the source test split, stratified by topic.
    test_pairs = test_pairs.shuffle(seed=SEED).flatten_indices()
    topics = test_pairs["topic"]
    devtest_idx = stratified_indices(topics, DEVTEST_SIZE, set())
    dev_idx = stratified_indices(topics, DEV_SIZE, set(devtest_idx))
    devtest = test_pairs.select(devtest_idx)
    dev = test_pairs.select(dev_idx)

    gatitos = load_gatitos()
    print(f"gatitos usable pairs: {len(gatitos)}")

    # Exact dedup: drop train rows sharing EITHER side with any eval row,
    # then dedup train on the full pair.
    blocked_en = set(devtest["en"]) | set(dev["en"])
    blocked_dv = set(devtest["dv"]) | set(dev["dv"])
    train_all = concatenate_datasets(
        [train_pairs] + [gatitos] * GATITOS_UPSAMPLE
    )
    seen = set()

    def train_filter(ex):
        if ex["en"] in blocked_en or ex["dv"] in blocked_dv:
            return False
        key = (ex["en"], ex["dv"])
        if key in seen and ex["source"] != "gatitos":  # gatitos upsample is deliberate
            return False
        seen.add(key)
        return True

    train = train_all.filter(train_filter)
    train = train.shuffle(seed=SEED).flatten_indices()

    # Keymap round-trip QA on the Dhivehi side (informational)
    sample = train.select(range(min(5000, len(train))))
    rt_fail = sum(
        1 for dv in sample["dv"] if keymap_to_thaana(thaana_to_keymap(dv)) != dv
    )
    print(f"keymap round-trip failures in 5k train sample: {rt_fail}")

    out = DatasetDict({"train": train, "dev": dev, "devtest": devtest})
    print({k: len(v) for k, v in out.items()})
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(OUT_DIR))
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
