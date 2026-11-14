# Knowledge-Constrained Identification of FN-MABAC Decision Components: code and results

This repository contains the code, data preparation scripts and result files for the paper
*Knowledge-Constrained Identification of Explainable, Rank-Reversal-Free Decision Components in Hybrid Intelligent Information Systems* (submitted to CAiSE 2027).

## Setup

```bash
pip install -r requirements.txt          # Python 3.13 was used
git clone https://github.com/c3nk/THE-World-University-Rankings data/THE-World-University-Rankings
cd scripts
```

All commands below are run from the `scripts/` folder. Data paths are relative to it.

## Data

* **THE World University Rankings 2023-2027** are obtained from the public repository
  `c3nk/THE-World-University-Rankings` (CSV files in `outputs/csv`). Only institutions with an exact
  overall score are used (`experiment.load`). The data are not redistributed here.
* **Graphics cards** (30 cards, 8 criteria, Tom's Hardware reference ranking) are transcribed in
  `gpu_data.py` from Tables 8-10 of Kizielewicz, Wieckowski and Salabun (2025), Engineering Applications of
  Artificial Intelligence 141:109736.

Data checks reported in the paper:

```bash
python3 verify_the.py                     # weighted sum vs. overall score, FN-MABAC vs. THE ranking
python3 gpu_experiment.py CHECK ../results 1   # reproduces the published rw = 0.92074
```

## Experiments

Labels in the paper and the corresponding commands and result files (the file names use internal labels):

| Paper | Description | Command | Result file(s) |
|---|---|---|---|
| E1 | Cross-edition identification (THE) | `python3 full_experiments.py E1 ../data/THE-World-University-Rankings/outputs/csv ../results 2` | `E1_cross_edition.csv` |
| E1 (all cores, JADE) | All-cores variants re-run with JADE | `python3 rerun_allcores.py E1 2` | `E1_allcores_jade.csv` |
| E1 (LP baseline) | Preference-disaggregation LP | `python3 extra_experiments.py E6 ../data/THE-World-University-Rankings/outputs/csv ../results 2` | `E6_lp_baseline.csv` |
| E1 (partial knowledge) | Partial, vague and mixed knowledge | `python3 extra_experiments.py E7 ... ../results 2` | `E7_partial_knowledge.csv` |
| E2 | Semi-synthetic non-monotonic criteria | `python3 full_experiments.py E2 ... ../results 2`, `python3 extra_experiments.py E2b ... ../results 2`, `python3 rerun_allcores.py E2 2` | `E2_nonmonotonic.csv`, `E2b_flag_plus_constraints.csv`, `E2_allcores_jade.csv` |
| E3 | Fuzzy estimates (weights, cores) | `python3 full_experiments.py E3 ... ../results 2`, `python3 extra_experiments.py E3b ... ../results 2` | `E3_fuzzy.csv`, `E3b_fuzzy_cores.csv` |
| E4 | Penalty strength | `python3 full_experiments.py E4 ... ../results 2` | `E4_lambda.csv` |
| Sec. 3 (rank-reversal check) | Rank reversal | `python3 full_experiments.py E5 ... ../results 1` | `E5_rank_reversal.csv` |
| E5 | Graphics cards | `python3 gpu_experiment.py SPLIT ../results 2`, `python3 gpu_experiment.py FULL ../results 2`, `python3 rerun_allcores.py GPU_SPLIT 2`, `python3 rerun_allcores.py GPU_FULL 2` | `E8_gpu_split.csv`, `E8_gpu_full.csv`, `E8_allcores_jade_split.csv`, `E8_allcores_jade_full.csv` |
| E1, E2, E5 (knowledge only) | Centroid of the weight region admitted by the knowledge, no decisions | `python3 median_deploy.py KNONLY ... ../results 1` | `knowledge_only.csv` |
| E1 (median deployment) | Median of 15 runs for SITW and SITW + knowledge (Table 2 rows marked median, repetitions 0-26 used) | `python3 median_deploy.py E1 ... ../results 2` (resumable) | `E1_median_deploy.csv` |
| E5 (median deployment) | Median of 15 runs for SITW, SITW + knowledge, flag data bus with and without knowledge (Table 4 rows marked median) | `python3 median_deploy.py GPU ... ../results 2` (resumable) | `E6_median_deploy.csv` |
| Runtime | Timing of one identification | `python3 extra_experiments.py TIME ... ../results 1` | `TIME.csv` |

The first column follows the paper. Script arguments and result file names keep the numbering of the development version.

`...` stands for `../data/THE-World-University-Rankings/outputs/csv`. The last argument is the number of worker processes.

Variants that learn *all* TFN cores are reported with JADE (300 iterations, 50 individuals), because PSO did not converge
reliably in the enlarged search space. The corresponding PSO rows in `E1_cross_edition.csv`, `E2_nonmonotonic.csv` and
`E8_gpu_*.csv` are superseded by the `*_allcores_jade*.csv` files.

## Summaries and figures

```bash
python3 summarize_full.py ../results      # E1-E4 tables, rank-reversal check and Wilcoxon tests
python3 summarize_extra.py ../results     # LP baseline, partial knowledge, flag + constraints, fuzzy cores, runtime
python3 summarize_gpu.py ../results       # graphics cards
python3 summarize_median.py ../results    # knowledge-only baseline and median deployment
python3 figures.py ../results <figure_dir> # Fig. 2 and Fig. 3
```

The outputs used for the paper are included as `results/summary*.txt`.

## Main components

* `experiment.py`: data loading, FN-MABAC, weighted Spearman correlation
* `full_experiments.py`: knowledge-constrained identification (`identify`), expert knowledge as hinge penalties, experiments E1-E5
* `extra_experiments.py`: LP baseline, partial knowledge, flag + constraints, fuzzy cores, runtime
* `gpu_data.py`, `gpu_experiment.py`: graphics-card case (general TFN bounds and criterion types)
* `rerun_allcores.py`: all-cores variants with JADE
