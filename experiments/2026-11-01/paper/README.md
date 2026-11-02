# Paper products of the 2026-11-01 results (manuscript comparison set)

Regenerated from `../result/` (the complete evaluation archive) with the manuscript's comparison set, i.e. without
the SPT rule, which was not part of the original submission's five dispatching rules (MOR, FIFO, MWKR, Random, RRC).
The archive in `../result/` is untouched and keeps every method, including SPT.

Commands (repository root; `$S` = a copy of `../result/`):

```
FSHGRL_EXCLUDE_METHODS=spt python -m result.aggregate --eval $S/eval_results.csv --out-dir $S
P=experiments/2026-11-01/paper
FSHGRL_EXCLUDE_METHODS=spt FSHGRL_RESULT_DIR=$S FSHGRL_FIG_DIR=$P/figures python paper_assets/figures/src/fig_train_curve.py   # and the other four figure scripts
FSHGRL_EXCLUDE_METHODS=spt FSHGRL_RESULT_DIR=$S FSHGRL_TABLE_DIR=$P/tables python paper_assets/scripts/make_tables.py
```

- `result/`: the aggregates with 12 paired comparators (`paired_tests.csv`, Holm family of 12), `generalization.csv`
  (best rule among the five), `ablation_results.csv`, `advantage_table.csv`, `friedman.csv`, `integer_check.csv`
  (693 groups, all consistent) and the validation-curve aggregates `conv_<method>.csv`;
- `tables/`: the booktabs bodies of the revised manuscript (Tables 5-13 and the appendix tables: generator settings,
  instance statistics, hyperparameters, ablation, rule and DRL comparisons (flat and grouped headers), Wilcoxon,
  arrival regimes, sensitivity with the greedy column, MILP check (full and compact), sampled decoding) with their
  CSV twins;
- `figures/`: Fig. 7 (`train_curve`, three panels with the selected checkpoint), the paired ablation differences
  with their test records, the decision-time panels (ms), the single-panel quality-time trade-off, the advantage
  heat map with a mean column, the mask-threshold sensitivity figure and the convergence curves of every learned
  method (repository product, not used in the manuscript), PDF + PNG.
