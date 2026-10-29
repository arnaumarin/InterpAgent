#!/usr/bin/env python3
"""
Build a parallel multilingual prompt set from FLORES-200 (dev + devtest).

Every language gets the same sentences in the same order, so language is the only
difference between categories. The files in multilingual_prompts/ come from different
sources per language (English instruction prompts vs. subtitle / UN-document text), which
confounds language with genre; prefer this set for language experiments.

FLORES-200 is licensed CC BY-SA 4.0. It is an evaluation benchmark, so the generated files
are not committed to this repository (see .gitignore); run this script to create them.

Usage:
  python workspace/data/build_parallel_prompts.py [--out DIR] [--tarball flores200_dataset.tar.gz] [--languages english,french,...]

Then run FeatureFinder with prompts_dir=workspace/data/parallel_prompts.
"""
import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import urllib.request

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SPLITS = ("dev", "devtest")
LANGUAGES = {
    "english": "eng_Latn", "german": "deu_Latn", "dutch": "nld_Latn",
    "french": "fra_Latn", "italian": "ita_Latn", "spanish": "spa_Latn", "portuguese": "por_Latn",
    "russian": "rus_Cyrl", "polish": "pol_Latn", "czech": "ces_Latn",
    "arabic": "arb_Arab", "hebrew": "heb_Hebr", "turkish": "tur_Latn", "hindi": "hin_Deva",
    "chinese": "zho_Hans", "japanese": "jpn_Jpan", "korean": "kor_Hang",
    "indonesian": "ind_Latn", "vietnamese": "vie_Latn", "thai": "tha_Thai",
}


def read_split(tar, code, split):
    member = tar.getmember(f"./flores200_dataset/{split}/{code}.{split}")
    return tar.extractfile(member).read().decode("utf-8").split("\n")[:-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "parallel_prompts"))
    parser.add_argument("--tarball", help="Local copy of flores200_dataset.tar.gz (downloaded if omitted)")
    parser.add_argument("--languages", default=",".join(LANGUAGES), help="Comma-separated subset of: " + ", ".join(LANGUAGES))
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    unknown = [lang for lang in languages if lang not in LANGUAGES]
    if unknown:
        parser.error(f"Unknown language(s): {unknown}")

    with tempfile.TemporaryDirectory() as tmp:
        path = args.tarball
        if path is None:
            path = os.path.join(tmp, "flores200_dataset.tar.gz")
            print(f"Downloading {URL} ...")
            urllib.request.urlretrieve(URL, path)
        with open(path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        with tarfile.open(path, "r:gz") as tar:
            texts = {lang: [s for split in SPLITS for s in read_split(tar, LANGUAGES[lang], split)] for lang in languages}

    lengths = {len(lines) for lines in texts.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"FLORES splits are not aligned across languages: {lengths}")
    # FeatureFinder drops empty lines, so keep only indices that are non-empty in every language
    keep = [i for i in range(lengths.pop()) if all(texts[lang][i].strip() for lang in languages)]

    os.makedirs(args.out, exist_ok=True)
    for lang in languages:
        with open(os.path.join(args.out, f"prompts_{lang}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(" ".join(texts[lang][i].split()) for i in keep) + "\n")
    manifest = {
        "source": URL, "sha256": sha256, "splits": list(SPLITS), "license": "CC BY-SA 4.0",
        "n_sentences": len(keep), "languages": {lang: LANGUAGES[lang] for lang in languages},
        "note": "Line i is the same FLORES sentence in every file.",
    }
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(languages)} languages x {len(keep)} parallel sentences to {args.out}")


if __name__ == "__main__":
    main()
