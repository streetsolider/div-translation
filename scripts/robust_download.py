"""Byte-range resuming downloader for the MADLAD 7B checkpoint.

huggingface_hub's downloader kept stalling on this connection and its
.incomplete files use per-attempt names, so a stall loses the whole shard.
This fetches with HTTP Range resume: a stall costs one read-timeout, not
the file. Downloads into models/madlad400-7b-mt-bt for local loading.
"""

import time
from pathlib import Path

import requests

REPO = "google/madlad400-7b-mt-bt"
DEST = Path(__file__).resolve().parents[1] / "models" / "madlad400-7b-mt-bt"
FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "tokenizer.json",
    "model.safetensors.index.json",
] + [f"model-0000{i}-of-00007.safetensors" for i in range(1, 8)]


def download(name):
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    dest = DEST / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"have  {name}", flush=True)
        return
    while True:
        try:
            head = requests.head(url, allow_redirects=True, timeout=30)
            head.raise_for_status()
            total = int(head.headers["Content-Length"])
            break
        except Exception as e:
            print(f"head retry {name}: {type(e).__name__}", flush=True)
            time.sleep(10)
    attempt = 0
    while True:
        have = tmp.stat().st_size if tmp.exists() else 0
        if have >= total:
            break
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
                r.raise_for_status()
                with open(tmp, "ab") as f:
                    t0, n = time.time(), 0
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
                        n += len(chunk)
                        if time.time() - t0 > 30:
                            print(
                                f"  {name}: {(have + n) / 1e9:.2f}/{total / 1e9:.2f} GB",
                                flush=True,
                            )
                            t0 = time.time()
        except Exception as e:
            attempt += 1
            print(f"  stall on {name} (attempt {attempt}): {type(e).__name__}", flush=True)
            time.sleep(min(60, 5 * attempt))
    tmp.rename(dest)
    print(f"done  {name}", flush=True)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        download(f)
    print("ALL_FILES_DOWNLOADED", flush=True)
