# duorou-memory Hard review evidence

This supplement responds to the [ATM-Bench maintainer's review](https://github.com/atmbench/atmbench.github.io/issues/13#issuecomment-7181441101). It describes the submitted 31-question run. The [evaluation release](https://github.com/Qiuxiangxiang/duorou-memory/releases/tag/atm-bench-hard-v1.0.0) freezes Recall@10 **80.76%** and official GPT-5 mini QS **45.04%**.

## Original answer run

- The entry point was [selector-lodging/qa.py](source/selector-lodging/qa.py), which invokes [selector-trip/paired_qa.py](source/selector-trip/paired_qa.py) with arm `combined_arrival`. It uses the batch helper in [event-graph-full/benchmark.py](source/event-graph-full/benchmark.py) and the Codex wrapper in [p0-full/cloud.py](source/p0-full/cloud.py). The prior arm was a comparison.
- The submitted arm used GPT-6.1 Terra at low reasoning effort. There were **six successful batches**, capped at six questions and 150,000 bytes of serialized rows. The generic `answer_and_score` function in `p0-full/evaluate.py` supplied neither the entry point nor the batching parameters for the submission. The paired script imported its `ANSWER` text and schema helpers.
- **Yes, the benchmark's `qtype` was supplied as the request's `format` field.** Each reader item contained the question ID and text, that format label, and ten ordered evidence IDs and text excerpts. Reference answers and gold IDs were used for evaluation. The reader payloads and their cache-key digests show exactly which fields it received.
- [Per-question request fields and responses](reader/per-question.jsonl) contain the exact excerpt text and normalized model answer for each question. [Batch manifest](reader/manifest.json) maps each question to the full batch prompt and original saved response and usage record. The prompt files reproduce the application input to Codex CLI. The saved responses are the parsed model outputs written by the original wrapper. The underlying service transcript, hidden instructions and failed-call receipts were outside what the harness saved.
- The six reconstructed request digests match the original cache filenames. Cached input tokens are already included in input token counts. The saved successful reader calls total **227,486 tokens**. All 31 responses match the official predictions.

Run `python3 verify_reader.py` from the supplement for an offline integrity check. For the historical reproduction, `export_reader.py` takes the original ATM work tree, corpus, predictions and cache as inputs.

## Upstream rankings and model usage

The immediate chain is `facet-shortlist/model-rankings:coverage_guard` plus proof scores → `weather-sequence:day_only` → `lodging:combined_arrival`. The supplement includes the code for both final deterministic steps and their frozen inputs. We replayed both steps and compared the original ranking and evidence-packet files byte-for-byte. The reduced proof checkpoint retains the same numeric inputs for the five questions routed by those two stages. The upstream proof-score generator and the geographic rerank source are also included. Reconstructing the entire upstream experiment tree from raw data requires additional checkpoints and receipt attribution.

| Stage or recorded scope | Model | Saved tokens | Attribution |
| --- | --- | ---: | --- |
| Hard alternate query rewrites | GPT-6.1 Terra | 25,364 | Saved rewrite input consumed by scope/corpus stages |
| Geographic facets shortlist rerank | GPT-6.2 Sol | 89,378 | Three saved facets requests. The coverage guard used the reranked list on one question |
| Submitted final answers | GPT-6.1 Terra | 227,486 | Six batches, exact request digests and predictions verified |
| Geographic flat shortlist rerank | GPT-6.2 Sol | 89,300 | Paired comparison |
| Early Hard P0 rerank | GPT-6.1 Terra | 345,215 | Historical cache receipts. Inclusion in the submitted configuration is unverified |
| Visual stage | GPT-6.1 Terra | 1,385,628 | 56 receipts reconcile to the visual stage summary. Allocation to the submitted chain is unresolved |
| Official scoring | GPT-5 mini | 8,968 | Evaluation-only usage |

[Stage usage](upstream-model-usage.csv) and [call inventory](model-call-inventory.jsonl) list **539** saved successful or candidate receipts and the SHA-256 of each cached record. The inventory includes parallel alternatives and repeated development-stage artifacts. Several stages reused inputs from earlier experiments. The ledger therefore supports inspecting calls and their recorded usage, but does not establish a deduplicated end-to-end total for the submitted configuration. **Leave the leaderboard's Total Tokens field unreported.** The final answer-token count remains available in the evidence.

## Inference inputs and tuning

The submitted configuration used public processed corpus descriptions, original emails, source metadata and, in upstream visual experiments, benchmark media. It inherited the following benchmark labels and artifacts:

| Boundary | Confirmed use |
| --- | --- |
| Final Terra reader | Passes the original `qtype` as `format` for all 31 questions |
| Geographic candidate route | `coverage_route` reads `qtype` to gate open-ended geographic requests |
| Public iterative route | `slots_from` uses `qtype` for list/aggregate handling |
| Upstream witness, answer-trace, recheck, compact union and visual stages | Use benchmark labels in routing, list handling, and/or model request formatting, as shown by the [source index](qtype-source-index.txt) |
| Final weather and lodging selection rules | Project and rank from question text and saved corpus data/inputs. These rules inherit the upstream artifacts described above |

The source index also includes scoring-only references to `qtype`. Development consulted Hard questions, reference answers and gold-driven diagnostics. The final lodging and visit variants were compared on those same 31 questions. The final stage changed four question lists, each adding one gold source. **Benchmark-tuned** is the appropriate board footnote. Reported QS and recall are in-sample and may overestimate unseen-question performance.

## Board row materials

Use type **Memory**, name **duorou-memory**, model **GPT-6.1 Terra**, Recall@10 **80.76**, QS **45.04**, and leave end-to-end total tokens and cost unreported. The publication includes the official judge summary and ranked source IDs. The benchmark-tuned legend should disclose the upstream and reader use of `qtype`, and the same-Hard-set error analysis. The upstream board example is [Graphon Hard PR #9](https://github.com/atmbench/atmbench.github.io/pull/9).
