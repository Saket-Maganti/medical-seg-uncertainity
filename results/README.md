# Results Guide

This directory contains generated outputs from the main experiments in the
repository. The goal is to make the repository useful immediately without
including the paper sources yet.

## Main subdirectories

- `drive/`: primary DRIVE benchmark results for MC Dropout, deterministic, TTA,
  and ensemble-style evaluations
- `cross_dataset/`: generalization results on STARE and CHASE_DB1
- `crossval/`: fold-wise outputs for cross-validation runs
- `ablations/`: results from controlled sweeps over uncertainty-related design
  choices
- `summaries/`: condensed CSV and JSON tables for reporting and quick inspection
- `paper_artifacts/`: exported figures generated from the experiment outputs

## Useful files

- `summaries/drive_main_table.csv`: main benchmark summary table
- `summaries/cross_dataset_table.csv`: cross-dataset performance summary
- `summaries/crossval_table.csv`: fold-wise aggregate metrics
- `summaries/runtime_table.csv`: runtime comparison table
- `summaries/deferral_operating_points.csv`: deferral thresholds and tradeoffs
- `summaries/stats_table.csv`: statistical comparison summary
- `cross_dataset/cross_dataset_results.json`: zero-shot results in JSON form
- `crossval/crossval_summary.json`: aggregate cross-validation summary

## Notes

- Large raw training artifacts such as local checkpoints, datasets, and tracking
  runs are not intended to be versioned as part of a clean repository snapshot.
- Some figure files are retained here because they are direct renderings of the
  experiment outputs, even though the manuscript itself is not being included
  yet.
