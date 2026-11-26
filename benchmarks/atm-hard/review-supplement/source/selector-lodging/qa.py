"""Evaluate the independent relevance selector with the existing paired reader and judge."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "p0-full")]
from evaluate import read_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "official", "cache", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.ranks = args.output / "qa-input-rankings.jsonl"
    with (args.output / "rankings.jsonl").open() as source, args.ranks.open("w") as target:
        for row in map(json.loads, source):
            target.write(json.dumps(dict(id=row["id"], previous=row["reference"],
                                         combined_arrival=row["combined_arrival"])) + "\n")
    args.arm = "combined_arrival"
    args.workers = 6
    args.stage_prefix = "hard-graph-order"
    paired = read_module("lodging_paired_qa", ROOT / "selector-trip/paired_qa.py")
    paired.main(args)


if __name__ == "__main__":
    main()
