"""
Download FLORES-200 data directly from HuggingFace.

Uses direct HTTP download since the datasets library has compatibility issues.
"""

import os
import json
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# FLORES-200 devtest split URLs (from HuggingFace)
# We'll download the all_languages config
BASE_URL = "https://huggingface.co/datasets/facebook/flores/resolve/main/devtest"

LANGUAGES = {
    "en": "eng_Latn",
    "id": "ind_Latn",
    "zh": "zho_Hans",
    "ms": "zsm_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "tl": "tgl_Latn",
    "jv": "jav_Latn",
}


def download_flores():
    """Download FLORES-200 devtest data for target languages."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Try the parquet format from open-language-data/flores-plus
    print("Downloading FLORES-200 data...")
    
    for lang_code, flores_code in LANGUAGES.items():
        filepath = os.path.join(DATA_DIR, f"{lang_code}_flores.txt")
        if os.path.exists(filepath):
            print(f"  {lang_code} already exists, skipping")
            continue
        
        # Try multiple URL patterns
        urls = [
            f"https://huggingface.co/datasets/open-language-data/flores-plus/resolve/main/devtest/{flores_code}.txt",
            f"https://huggingface.co/datasets/facebook/flores/resolve/main/devtest/{flores_code}.devtest",
        ]
        
        downloaded = False
        for url in urls:
            try:
                print(f"  Trying {url}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content = resp.read().decode("utf-8")
                    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
                    with open(filepath, "w", encoding="utf-8") as f:
                        for line in lines:
                            f.write(line + "\n")
                    print(f"  Downloaded {lang_code} ({flores_code}): {len(lines)} sentences")
                    downloaded = True
                    break
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        if not downloaded:
            print(f"  WARNING: Could not download {lang_code} ({flores_code})")

    print("Done!")


def load_flores():
    """Load previously downloaded FLORES data."""
    texts = {}
    for lang_code in LANGUAGES:
        filepath = os.path.join(DATA_DIR, f"{lang_code}_flores.txt")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                texts[lang_code] = [line.strip() for line in f if line.strip()]
    return texts


if __name__ == "__main__":
    download_flores()
    
    texts = load_flores()
    for lang, t in texts.items():
        print(f"  {lang}: {len(t)} sentences, avg length {sum(len(x) for x in t)/max(len(t),1):.0f} chars")