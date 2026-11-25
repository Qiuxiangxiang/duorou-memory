# duorou-memory on ATM-Bench Hard

This bundle records one experimental duorou v1.0.0-based configuration on all 31 Hard questions. The evaluated selector uses prepared corpus descriptions and original source IDs. The final selection stage groups lodging records by venue and dates, keeps reservation confirmations alongside relevant cancellation or post-stay records, and adds earlier arrival observations for separate visits.

| Metric | Value |
|---|---:|
| Recall@10, official macro definition | 80.7566848079% |
| Questions with all gold sources retrieved | 18 / 31 |
| Questions with at least one gold source retrieved | 31 / 31 |
| MRR@10 | 0.8020 |
| Source hits across questions | 143 / 194 |
| QS, GPT-6.2 Sol on open questions | 57.93824% |
| QS, gpt-5-mini on open questions | **45.04%** |
| gpt-5-mini judge tokens | 8,968 |
| GPT-6.1 Terra answer generation tokens | 227,486 |
| GPT-6.2 Sol judging tokens | 32,166 |

Question types are graded with the ATM evaluator's deterministic number and list routines. The Sol and GPT-5 mini rows use their respective models to judge open-ended answers. The two token figures include cached input and cover the saved reader and judge receipts. The pipeline reuses earlier candidate rankings and relevance scores. Build and upstream retrieval costs are outside those figures. An earlier geo rerank used 89,378 tokens, as logged in the experiment report.

Hard questions and gold-label error analysis informed development. These are in-sample results. At inference the final selector calls use the question and source corpus. Gold IDs are used for measurement. The selector is an experimental harness built on the v1.0.0 core.

## Files

- `evidence/predictions.jsonl` contains the 31 frozen Terra answers in the official evaluator's format.
- `evidence/retrieval_recall_details.json` contains each submitted ranking and its public gold IDs.
- `evidence/official-retrieval-output.json` contains the pinned official retrieval evaluator's output.
- `evidence/sol-qa-summary.json` and `evidence/sol-per-question.jsonl` preserve the alternate-judge result.
- `atm/` contains the final selector and the fixed upstream artifacts it consumes, including the predecessor rankings. The bundled score file retains the needed numeric score columns. Source text is loaded from the pinned public dataset.
- The repository root contains the original v1.0.0 Python core used by the selector.
- `rejudge/` contains the official GPT-5 mini summary, per-question grades and usage receipts.

## Reproduce

Commands below run from the `benchmarks/atm-hard` directory. The two URLs identify the benchmark's own files and evaluator.

```bash
git clone https://github.com/JingbiaoMei/ATM-Bench.git upstream
git -C upstream checkout ef4e5dff1a47ec71213a06e359f02753defa8fb1
mkdir -p data
data_url=https://huggingface.co/datasets/Jingbiao/ATM-Bench/resolve/78e826dc07e97466b2f54443831ef9a83ab8b27c/data
for name in atm-bench atm-bench-hard emails image_batch_results video_batch_results; do
  curl -fL "$data_url/$name.json" -o "data/$name.json"
done
python3 -m venv .venv
.venv/bin/python -m pip install numpy onnxruntime tokenizers requests tqdm openai
.venv/bin/python atm/selector-lodging/run.py ../.. data upstream rerun
.venv/bin/python atm/selector-lodging/verify.py data atm/selector-weather-sequence/rankings.jsonl rerun
.venv/bin/python upstream/memqa/utils/evaluator/evaluate_retrieval/comprehensive_eval.py \
  --details evidence/retrieval_recall_details.json --output evidence/official-retrieval-output.json
```

The fixed upstream artifacts and original data reproduce the final selector and its evidence packets. Rebuilding all development variants from raw files is outside this snapshot's reproduction scope. This also preserves the original reader answers for independent judging.

Official judge command, for an environment with access to `gpt-5-mini`:

```bash
PYTHONPATH=upstream .venv/bin/python upstream/memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth data/atm-bench-hard.json --predictions evidence/predictions.jsonl \
  --output-dir official-qa --metrics atm --judge-provider openai \
  --judge-model gpt-5-mini --judge-reasoning-effort minimal
```

The saved GPT-5 mini run completed all 13 model judgments with zero fallbacks.
