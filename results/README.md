# Selected experiment exports

These are downloaded from the completed run in Drive, not invented dashboard data. `artifact_manifest.json` records source IDs, download time and SHA-256 checksums. The full source run remains in Drive; this repository keeps only artifacts needed to show, verify and use the final model.

- `model_comparison_validation.csv`: five models × four stages, evaluated against BART references.
- `final_test_metrics.csv`, `final_test_per_category.csv`: frozen model versus BART test labels.
- `human_audit_metrics.csv`, `human_audit_bootstrap_intervals.csv`: unweighted audit and weighted group-population estimates.
- `human_audit_joined_predictions.csv`: 300 reviewed labels and model predictions, retaining annotation notes for provenance. No full article corpus is included.
- `frozen_selection.json`: selection rule, category order, versions and dataset hash.
- `coverage_reliability_proxies.csv`, `audit_refinement_changes.csv`: supporting refinement evidence.
- `ML_Assignment_Results_Report.md`, `run_manifest.json`, `model_requirements.txt`: original report and run metadata. The run manifest describes the full Drive run, including files intentionally omitted here. Figure paths in the original report refer to that full run.

The audit is a supplied reviewed-label set; notes retain AI-assisted provenance. BART-reference metrics must not be reported as independently verified accuracy.
