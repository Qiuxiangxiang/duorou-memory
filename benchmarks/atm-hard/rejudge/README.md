# duorou-memory ATM-Bench Hard rejudge

| Metric | Result |
|---|---:|
| Recall@10, published rank list | 80.76% |
| QS, official evaluator and gpt-5-mini | 45.04% |
| Number | 1/6 |
| List recall (mean Jaccard) | 66.34% |
| Open-ended | 5/13 |
| Judge API input tokens | 7,775 |
| Judge API output tokens | 1,193 |
| Judge API total tokens | 8,968 |

Scores cover all 31 Hard questions. The grader reused the exact frozen predictions from the [original submission](https://github.com/atmbench/atmbench.github.io/issues/13). The OpenAI API returned 13 completed responses from the gpt-5-mini-2025-08-07 snapshot. The reasoning effort was minimal, with zero fallbacks.

The Sol and gpt-5-mini runs agree on the deterministic number and list grades. Of the open-ended grades, 4 moved from pass to fail and 0 moved from fail to pass. Earlier Sol QS was 57.94%.

The reader used 227,486 tokens in the original run. Reader and gpt-5-mini judge usage sum to 236,454 tokens. The reused upstream retrieval and build stages have separate usage.

See the JSON files in the evidence archive for per-question grades, evaluator summary, and model usage. The archive contains no API credential.
