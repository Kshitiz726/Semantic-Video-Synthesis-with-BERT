"""Fetch the TransNetV2 PyTorch weights (converted from the official release).

The weights are taken from the `transnetv2-pytorch` wheel on PyPI and checked
against a known SHA-256.

    python scripts/download_weights.py
"""

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "weights" / "transnetv2-pytorch-weights.pth"
SHA256 = "a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de"
PACKAGE, VERSION = "transnetv2-pytorch", "1.0.5"


def main():
    if OUT.exists() and hashlib.sha256(OUT.read_bytes()).hexdigest() == SHA256:
        print(f"Already present: {OUT}")
        return
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{PACKAGE}/{VERSION}/json"))
    url = next(u["url"] for u in meta["urls"] if u["filename"].endswith(".whl"))
    print(f"Downloading {url}")
    wheel = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
    data = wheel.read("transnetv2_pytorch/transnetv2-pytorch-weights.pth")
    if hashlib.sha256(data).hexdigest() != SHA256:
        sys.exit("Checksum mismatch - refusing to write weights")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(data)
    print(f"Saved {OUT} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
