# archive/ — superseded & not-in-paper code

These scripts are **not part of the published replication package**. They are kept
for provenance only and are excluded from the documented reproduction workflow.
They may import modules or reference datasets (e.g. DevEval) that are no longer
shipped, so they are not guaranteed to run.

| File | Why archived |
|------|--------------|
| `run_deveval_eval.py`, `convert_to_deveval.py`, `fix_indentation.py`, `devEvalTools.py` | DevEval benchmark exploration — produced 0% Pass@1 and is **not used in the paper**. The 3.1 GB DevEval dataset is not shipped. |
| `run_ablation.py`, `run_ablation_v2.py`, `run_multimodel_ablation.py` | Early ablation runners, superseded by `run_humaneval_ablation.py` (supplementary §E1–E4). |
| `run_experiment.py`, `run_full_experiment.py`, `main.py` | Early/legacy single-model runners, superseded by the per-benchmark runners (`run_tcgp_vs_cot.py`, `run_classeval.py`, `run_livecodebench.py`, `run_humaneval_plus.py`). |
| `run_humaneval_plus_v2.py` | Superseded by `run_humaneval_plus.py`. |

For the active, paper-relevant code and the reproduction workflow, see the
top-level `README.md` and `REPRODUCE.md`.
