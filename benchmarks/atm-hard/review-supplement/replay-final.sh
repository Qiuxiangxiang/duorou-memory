#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
  printf 'Usage: bash replay-final.sh PYTHON DUOROU_ROOT ATM_DATA ATM_OFFICIAL OUTPUT\n' >&2
  exit 2
fi
bench_dir=$(cd "$(dirname "$0")" && pwd)
"$1" "$bench_dir/source/selector-weather-sequence/run.py" "$2" "$3" "$4" "$5/weather"
cmp "$5/weather/rankings.jsonl" "$bench_dir/source/selector-weather-sequence/rankings.jsonl"
# Lodging consumes the frozen day_only file just verified above.
"$1" "$bench_dir/source/selector-lodging/run.py" "$2" "$3" "$4" "$5/lodging"
cmp "$5/lodging/rankings.jsonl" "$bench_dir/submitted/selector-rankings.jsonl"
"$1" "$bench_dir/verify_reader.py"
