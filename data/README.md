# Data Directory

- `humaneval/humaneval.jsonl` — the HumanEval benchmark (164 problems), committed.

ClassEval, LiveCodeBench, and HumanEval+ are downloaded on demand (HuggingFace /
the `evalplus` package); run `bash scripts/download_datasets.sh` to fetch and
verify them. The legacy DevEval dataset is not part of this package (see
`archive/`).
