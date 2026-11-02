# Paper products of the 2026-11-01 results (manuscript comparison set)

Regenerated from `../result/` (the complete evaluation archive) with the manuscript's comparison set, i.e. without
the SPT rule, which was not part of the original submission's five dispatching rules (MOR, FIFO, MWKR, Random, RRC).
The archive in `../result/` is untouched and keeps every method, including SPT.

Commands (repository root; `$S` = a copy of `../result/`):

```
FSHGRL_EXCLUDE_METHODS=spt python -m result.aggregate --eval $S/eval_results.csv --out-dir $S
FSHGRL_EXCLUDE_METHODS=spt FSHGRL_RESULT_DIR=$S FSHGRL_FIG_DIR=figures python paper_assets/figures/src/fig_train_curve.py   # and the other three figure scripts
FSHGRL_EXCLUDE_METHODS=spt FSHGRL_RESULT_DIR=$S FSHGRL_TABLE_DIR=tables python paper_assets/scripts/make_tables.py
```

- `result/`: the aggregates with 12 paired comparators (`paired_tests.csv`, Holm family of 12), `generalization.csv`
  (best rule among the five), `ablation_results.csv`, `advantage_table.csv`, `friedman.csv`, `integer_check.csv`
  (726 groups, all consistent) and the validation-curve aggregates `conv_<method>.csv`;
- `tables/`: the booktabs bodies of the revised manuscript (Tables 6-11, the arrival, small-instance, sensitivity,
  MILP and instance-statistics tables) with their CSV twins;
- `figures/`: Fig. 7 (`train_curve_*`), the convergence curves, the paired ablation differences, the decision-time
  panels, the quality-time trade-off and the advantage heat map, PDF + PNG.
