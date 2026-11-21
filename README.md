# Ensemble Learning for Sri Lankan News Classification

Interactive Python dashboard comparing Logistic Regression, Random Forest (bagging), XGBoost (boosting), out-of-fold stacking, an ANN, and a validation-selected stacking/ANN probability blend.

Author: R. K. Edirisinghe, 224048G. Code generated and revised with assistance from ChatGPT. Review and understand the implementation before submission.

## Run locally

```bash
python -m venv .venv
.venv/bin/python -m pip install -r dashboard/requirements.txt
cd dashboard
../.venv/bin/python -m streamlit run app.py
```

Without artifacts the app displays its methodology and clear pending states. It does not fabricate predictions or metrics.

## Export trained models from Colab

Run the training notebook through its frozen-model test section. Then run the final dashboard export cell, also supplied in `scripts/export_dashboard_section.py`. It exports `dashboard_export.zip` containing:

- A compressed `hybrid_bundle.joblib` with all three base pipelines, stack meta-model, ANN pipeline, label mapping, base order and selected weight.
- Matching validation/test article-level predictions, category support, ANN history and validation blend search.
- `run.json` with dataset hash, taxonomy, counts, selected settings, checksums and library versions.
- Exact model dependency versions.

Import your own completed export:

```bash
python scripts/import_dashboard_export.py /absolute/path/to/dashboard_export.zip
.venv/bin/python -m pip install -r dashboard/requirements.txt
```

Only load model bundles produced by your own training notebook: joblib/pickle files can execute Python code. The public dashboard intentionally has no model-upload control. The importer checks filenames and checksums and refuses to overwrite an existing experiment. To use another export, preserve the old artifacts directory and install into an empty one.

`NEWS_ARTIFACT_DIR` optionally points the local app to another installed experiment directory. On the hosted app use `dashboard/artifacts`.

## Deploy from a private GitHub repository

1. Commit the dashboard, scripts, notebooks and completed artifact export. Raw news archives, virtual environments, temporary files and secrets are excluded by `.gitignore`.
2. Inspect model file size before pushing. If it exceeds GitHub's ordinary file limit, use an appropriate model artifact distribution method rather than committing it directly.
3. In Streamlit Community Cloud, choose the repository and branch and set the entrypoint to `dashboard/app.py`.
4. Choose a Python version compatible with the export's `run.json`; the import step pins the model library versions from training.
5. Deploy, verify startup and classify an article. Check access settings and provide examiner access as needed; a private repository alone does not define the application's audience.

Deployment requires the owner's GitHub/Streamlit sign-in and completed training artifacts. The dashboard does not retrain on startup and does not need BART at inference time.

## Evaluation contract

- Source: Sri Lankan news corpus scraped by Nuwan, https://github.com/nuuuwan/lk_news/tree/data.
- Input: English headline, newline, first 200 whitespace-separated article-body words.
- Reference labels: BART zero-shot generated topics; scores measure agreement with these labels.
- Grouped train/validation/test splits keep normalised identical headlines or bodies together. Semantic near-duplicates can remain.
- Every base model refits preprocessing within its OOF training fold. ANN TF-IDF/SVD/scaling use training data only.
- ANN epoch and blend weight are chosen on validation. Test reports retain those frozen settings.
- Models in the dashboard comparison use identical eligible categories and articles. The original pilot excludes singleton training categories from the stacking experiment; original split counts and comparison counts are recorded separately.
- Overall macro-F1 includes all experiment categories using zero for undefined F1, as in the notebook. Category tables show no-support categories as blank.
- A slider changes only exploratory new-article blending. It cannot change recorded evaluation scores.
- A selected endpoint (0% or 100% stacking) means no hybrid benefit has been demonstrated.
- v7 pilot results and final v8 results must remain distinct. A pilot is a workflow check and was previously used during taxonomy development.

## Verify

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests/test_dashboard.py -q
```

The tests exercise class/base ordering, probability blending, text truncation, all-category metrics, archive paths and all dashboard views with missing and evaluation-only artifacts.
