#!/usr/bin/env python3
"""Restore omitted public source projections for auditing an archived report.

This does not rerun retrieval or load tests and does not alter recorded results.
The raw dataset and each restored file must match the saved SHA256 exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from public_memory_bench.locomo import convert_locomo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "dataset.json").read_text())
    payload = args.locomo.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["raw_sha256"]:
        raise ValueError("Raw LoCoMo file does not match the recorded SHA256")
    records = json.loads(payload)
    raw = []
    for sample in records:
        scope = "locomo:" + sample["sample_id"]
        conversation = sample["conversation"]
        sessions = sorted((key for key in conversation if re.fullmatch(r"session_\d+", key)),
                          key=lambda key: int(key.removeprefix("session_")))
        for key in sessions:
            for turn in conversation[key]:
                raw.append({"id": scope + ":" + turn["dia_id"], "scope": scope, "text": turn["text"]})
    projections = {"raw_documents": raw,
                   "enriched_documents": convert_locomo(records, include_captions=True)["documents"]}
    pending = []
    for name, rows in projections.items():
        expected = manifest["files"][name]
        data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                       for row in rows).encode("utf-8")
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Restored projection SHA256 mismatch: {name}")
        path = args.run_dir / expected["path"]
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite a different projection: {path}")
        pending.append((path, data))
    for path, data in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    print(json.dumps({"restored_or_verified": [str(path) for path, _ in pending],
                      "all_sha256_match": True, "benchmark_rerun": False}))


if __name__ == "__main__":
    main()
