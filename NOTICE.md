# duorou: sources and data terms

This package contains independently authored design documents, a SQLite schema,
demonstration scripts, and validation reports. It does not incorporate the
Rolling Memory source repository and does not redistribute the complete LoCoMo
or LongMemEval raw datasets.

## LoCoMo

- Authors/project: Maharana et al., SNAP Research, *Evaluating Very Long-Term
  Conversational Memory of LLM Agents* (2024).
- Repository: https://github.com/snap-research/locomo
- Pinned revision: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.
- Source path: `data/locomo10.json`.
- SHA256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- Dataset license: CC BY-NC 4.0; upstream terms apply to its use and to derived
  redistribution. https://creativecommons.org/licenses/by-nc/4.0/

The import demonstration uses the published conversation fields and allowed
source metadata. It does not index evaluation questions, reference answers,
gold evidence labels, generated summaries, or `has_answer` labels. Any source
snippets in the generated report are attributed to the above dataset. The
generated database is excluded from this distribution and can be reproduced.

## Previous benchmark snapshot

The included baseline snapshot is copied from our earlier independent public
memory benchmark results. LongMemEval's source repository is
https://github.com/xiaowu0162/LongMemEval and its public dataset is
https://huggingface.co/datasets/xiaowu0162/longmemeval (MIT). The prior cleaned
dataset revision was `98d7416c24c778c2fee6e6f3006e7a073259d48f`.
The snapshot is historical context, not a new evaluation of this schema.

## Research references

Primary papers and the limits of their engineering interpretation are listed
in `docs/SCIENTIFIC_BASIS.zh.md`. No paper, dataset author, or institution is
claimed to endorse this design. Database fields, indexes, and API proposals
are engineering decisions to be tested independently.

## This release benchmark artifacts

The LoCoMo-derived evaluation queries, gold evidence identifiers and rankings are
provided for reproducibility under the upstream CC BY-NC 4.0 conditions. Full
raw conversations and the generated SQLite database are omitted from the release
archive; the pinned downloader and runner regenerate them. The upstream license
is retained in `licenses/locomo-LICENSE.txt`.

The `vendor/` directory contains our independently authored earlier benchmark
engine and lexical retrieval control, preserved without source edits. The engine
license is retained in `vendor/MEMORY_QUERY_BENCH_LICENSE`. The new MCP adapter
uses the installed MCP SDK; third-party runtime packages are listed in
`requirements.txt` and are not vendored as source.
