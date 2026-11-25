"""Fetch the pinned public LoCoMo file and verify its content hash.

Only the upstream dataset is downloaded; the package does not redistribute it.
See NOTICE.md for dataset attribution and CC BY-NC 4.0 terms.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
URL = f"https://raw.githubusercontent.com/snap-research/locomo/{REVISION}/data/locomo10.json"
EXPECTED_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/locomo10.json"))
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        if digest(destination) != EXPECTED_SHA256:
            raise SystemExit(f"Refusing to overwrite a different file: {destination}")
        print(f"Verified existing dataset: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with urllib.request.urlopen(URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                temp_name = handle.name
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        temporary = Path(temp_name)
        if digest(temporary) != EXPECTED_SHA256:
            raise SystemExit("Downloaded data does not match pinned SHA256; no output committed.")
        # Do not overwrite a file created by another process while downloading.
        os.link(temporary, destination)
        print(f"Downloaded and verified: {destination}")
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
